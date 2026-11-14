// trace_demo.rs — Metatron Dynamics, Inc.
// ABR Relational Attention — Single-Passage Relational Trace
//
// Implementation revision: abr-relational-attention  branch: declaration-v5-0
// (see `git log --oneline -1` for exact commit at run time)
//
// Passage (attributed, exact):
//   Darwin, C. (1859). On the Origin of Species by Means of Natural Selection.
//   John Murray, London.  Chapter II, "Variation under Nature", first sentence.
//   Text: "When we look to the"  (first 5 tokens of DARWIN_PARAGRAPH corpus)
//   Text preserved exactly: capitalisation, punctuation, boundaries as-is.
//
// Corpus for M: the full DARWIN_PARAGRAPH (233 tokens) is the M-building corpus.
//   This is the same corpus used in wo_demo.rs.
//   M is PPMI co-occurrence within sentence boundaries (OC-RA-10 CLOSED).
//   dim=10, window=5, min_count=1.
//   Normalisation: min-max per dimension to [0,1]^dim at load time.
//
// Declared parameters (all CLOSED):
//   η  = 0.1    (OC-RA-2)
//   θ_min = 0.05  (OC-RA-1)
//   ρ_base = 1.0 (language domain M-declared)
//   χ₀ = 1.0    (V9 general; one unit of [0,1]^dim distance)
//   W  = 64     (OC-RA-5)
//
// This trace:
//   - processes each token ONCE, in order (single-pass streaming)
//   - reports all intermediate quantities for every token arrival
//   - uses θ=0.0 for the structural trace (all accumulated pairs admitted)
//   - reports admission status under declared θ_min=0.05 separately
//   - preserves signed R values without clipping or normalisation
//   - distinguishes: measured zero / unavailable information / no admitted edge
//
// Antisymmetric circulation:
//   The causal streaming field admits only directed edges from earlier to later
//   positions (RelationalPair::declare enforces src.position < tgt.position).
//   No directed cycles exist.  Antisymmetric circulation is therefore
//   structurally unresolved for this passage — not measured zero.
//   The antisymmetry of A[e] = x[src] − x[tgt] under direction reversal is a
//   structural property (A[rev(e)] = −A[e]); reverse edges are inadmissible.
//
// ι (iota / identity observable):
//   SemanticM::from_corpus does not set iota weights for a plain PPMI build.
//   Energy ε[e] = |ι[src] − ι[tgt]| is therefore UNAVAILABLE (not measured zero).
//   StreamingField::epsilon_for returns 0.0 when iota is empty.
//   Accumulation proceeds at base rate η (ambient accumulation, OC-RA-12).
//
// RUN:
//   cargo run --release --bin trace_demo

use std::collections::HashSet;

use abr_relational_attention::{
    declaration::{ETA, THETA_MIN, COHERENCE_WINDOW, Locus},
    operators::{DeclaredGraph, compute_rho, operator_a, operator_b, operator_r},
    semantic::{SemanticM, tokenize_text},
    streaming::StreamingField,
};

/// ρ_base — declared by M for the language domain.
/// V9 general value: 1.0.  Declared here because the declaration module
/// does not export a RHO_BASE constant (it is passed as a parameter to
/// compute_rho, not stored as a crate-level constant).
const RHO_BASE: f64 = 1.0;

// ── Passage and corpus ────────────────────────────────────────────────────────

/// Corpus for M-building: Darwin (1859), On the Origin of Species, Ch. II.
/// Exactly the same text used in wo_demo.rs.
const DARWIN_PARAGRAPH: &str = "
When we look to the individuals of the same variety or sub-variety of our
older cultivated plants and animals, one of the first points which strikes
us, is, that they generally differ much more from each other, than do the
individuals of any one species or variety in a state of nature. When we
reflect on the vast diversity of the plants and animals which have been
cultivated, and which have varied during all ages under the most different
climates and treatment, I think we are driven to conclude that this greater
variability is simply due to our domestic varieties having been raised under
conditions of life not so uniform as, and somewhat different from, those to
which the parent-species have been exposed under nature. There is, also, I
think, some probability in the view propounded by Andrew Knight, that this
variability may be partly connected with excess of food. It seems pretty
clear that organic beings must be exposed for several generations to the new
conditions of life to cause any appreciable amount of variation; and that
when the organisation has once begun to vary, it generally continues to vary
for many generations. No case is on record of a variable being ceasing to be
variable after it has once begun to vary.
";

/// The 5-token passage to trace.
/// Exact text, capitalisation, and boundary preserved.
const TRACE_PASSAGE: &str = "When we look to the";

/// Number of tokens to trace.
const N_TRACE: usize = 5;

/// PPMI embedding dimension for the trace M.
const DIM: usize = 10;

/// PPMI co-occurrence window.
const PPMI_WINDOW: usize = 5;

/// PPMI min-count (1 = include all tokens).
const MIN_COUNT: usize = 1;

// ── Helper functions ──────────────────────────────────────────────────────────

fn vec_summary(v: &[f64], label: &str) {
    if v.is_empty() { println!("  {}: (empty)", label); return; }
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // Print all components (DIM=10 is small enough)
    let cpts: Vec<String> = v.iter().map(|x| format!("{:+.6}", x)).collect();
    println!("  {}: norm={:.6}  range=[{:+.6}, {:+.6}]", label, norm, min, max);
    println!("    components: [{}]", cpts.join(", "));
}

fn rho_n_from_a(a: &[f64], chi0: f64, rho_base: f64) -> (f64, f64) {
    // Returns (chi, rho_n)
    let chi: f64 = a.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let rho_n = rho_base * chi / (chi0 + chi);
    (chi, rho_n)
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    // ── Print trace header ─────────────────────────────────────────────────────
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("ABR RELATIONAL TRACE — Darwin (1859), On the Origin of Species, Ch. II");
    println!("Implementation: abr-relational-attention  branch: declaration-v5-0");
    // Print commit at run time
    let commit = std::process::Command::new("git")
        .args(["log", "--oneline", "-1", "--format=%H %s"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "(git unavailable)".to_string());
    println!("Commit: {}", commit);
    println!("Kernel: V9  ·  Declaration: V5.0 (2026-11-13)");
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("Passage (exact text, capitalisation, boundaries preserved):");
    println!("  \"{}\"", TRACE_PASSAGE);
    println!("  Source: Darwin (1859), On the Origin of Species, Ch. II, John Murray, London.");
    println!("          First {} tokens. Public domain.", N_TRACE);
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("M declaration:");
    println!("  SemanticM::from_corpus — PPMI co-occurrence within sentence boundaries");
    println!("  Corpus: DARWIN_PARAGRAPH (same text as wo_demo.rs, {} chars)", DARWIN_PARAGRAPH.len());
    println!("  dim={}, PPMI_window={}, min_count={}", DIM, PPMI_WINDOW, MIN_COUNT);
    println!("  Normalisation: min-max per dimension to [0,1]^dim at load time (OC-RA-10)");
    println!("  Connection level: M-word (token/sentence boundary) — single-level for this trace");
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("Declared parameters:");
    println!("  η = {:.4}  (OC-RA-2 CLOSED)", ETA);
    println!("  θ_min = {:.4}  (OC-RA-1 CLOSED)", THETA_MIN);
    println!("  ρ_base = {:.4}  (language domain M-declared)", RHO_BASE);
    println!("  χ₀ = 1.0  (V9 general; one unit of [0,1]^dim distance)");
    println!("  W = {} positions  (OC-RA-5 CLOSED)", COHERENCE_WINDOW as usize);
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("ι (identity observable): UNAVAILABLE — SemanticM::from_corpus does not");
    println!("  set iota weights.  ε[e] = |ι[src]−ι[tgt]| is unavailable (not measured zero).");
    println!("  Accumulation proceeds at base rate η (ambient; OC-RA-12 path not activated).");
    println!("─────────────────────────────────────────────────────────────────────────────");
    println!("Antisymmetric circulation: STRUCTURALLY UNRESOLVED for this passage.");
    println!("  Causal field: all edges directed from earlier to later position.");
    println!("  RelationalPair::declare enforces src.position < tgt.position.");
    println!("  No directed cycles possible.  Circulation is not measured zero;");
    println!("  the loop Σ_cycle R[e] has no instances to evaluate.");
    println!("  A[e] = x[src]−x[tgt] is structurally antisymmetric: A[rev(e)] = −A[e].");
    println!("  Reverse edges are inadmissible; this antisymmetry cannot be measured as");
    println!("  circulation here.  Correspondence not yet established.");
    println!("═══════════════════════════════════════════════════════════════════════════════");

    // ── Build M ────────────────────────────────────────────────────────────────
    println!();
    println!("BUILDING M from corpus...");
    let sem = SemanticM::from_corpus(DARWIN_PARAGRAPH, DIM, PPMI_WINDOW, MIN_COUNT);
    let vocab = sem.to_vocab_table();
    let vocab_size = vocab.len();
    println!("  Vocabulary size: {} unique tokens", vocab_size);
    println!("  Embedding dim:   {}", DIM);

    // ── Tokenise the trace passage ─────────────────────────────────────────────
    let trace_tokens: Vec<String> = tokenize_text(TRACE_PASSAGE);
    assert!(trace_tokens.len() >= N_TRACE,
        "TRACE_PASSAGE has {} tokens, need >= {}", trace_tokens.len(), N_TRACE);
    let trace_tokens = &trace_tokens[..N_TRACE];

    // Resolve token IDs
    let trace_ids: Vec<usize> = trace_tokens.iter().map(|w| {
        *sem.vocab.get(w.as_str())
            .unwrap_or_else(|| panic!("Token \"{}\" not in SemanticM vocabulary", w))
    }).collect();

    println!();
    println!("TRACE TOKENS (resolved against M vocabulary):");
    for (i, (word, &tid)) in trace_tokens.iter().zip(trace_ids.iter()).enumerate() {
        println!("  position {} : \"{}\"  token_id={}", i, word, tid);
    }

    println!();
    println!("M VECTORS (observable information at each token):");
    println!("  M[token] ∈ [0,1]^dim — PPMI embedding, min-max normalised.");
    println!("  This is the phonemic-semantic state: the declared observable for each type.");
    for (word, &tid) in trace_tokens.iter().zip(trace_ids.iter()) {
        vec_summary(&vocab[tid], &format!("M[\"{}\"] (id={})", word, tid));
    }

    // ── Streaming field ────────────────────────────────────────────────────────
    let mut field = StreamingField::new();
    // No iota set → ε unavailable → ambient accumulation

    // We maintain a parallel sequence to build DeclaredGraph outside StreamingField
    // (stream field is private in StreamingField)
    let mut our_sequence: Vec<(usize, &str)> = Vec::new(); // (token_id, word)

    // ── Token-by-token trace ───────────────────────────────────────────────────
    for (position, (word, &token_id)) in trace_tokens.iter().zip(trace_ids.iter()).enumerate() {

        println!();
        println!("╔═══════════════════════════════════════════════════════════════════════════╗");
        println!("║  TOKEN ARRIVAL  position={}  word=\"{}\"  token_id={}",
                 position, word, token_id);
        println!("╚═══════════════════════════════════════════════════════════════════════════╝");

        // ── Accumulation phase ─────────────────────────────────────────────────
        println!();
        println!("▶ ACCUMULATION");
        println!("  Coherence window: positions [{}, {}]",
                 position.saturating_sub(COHERENCE_WINDOW as usize), position.max(1)-1);

        if position == 0 {
            println!("  No predecessors in window (position 0).  No edges created.");
            println!("  EdgeStore: empty.  frontier advances to 1.");
        } else {
            // For each predecessor in window: show what accumulation computes
            let window_start = position.saturating_sub(COHERENCE_WINDOW as usize);
            println!("  {} predecessor(s) in window:", position - window_start);
            for pred_pos in window_start..position {
                let (pred_tid, pred_word) = our_sequence[pred_pos];
                // Compute A for this pair
                let src_locus = Locus::new(pred_pos, pred_tid);
                let tgt_locus = Locus::new(position, token_id);
                let a_vec = operator_a(&src_locus, &tgt_locus, &vocab);
                let (chi, rho_n) = rho_n_from_a(&a_vec, 1.0, RHO_BASE);

                // ρ_acc BEFORE this arrival
                let rho_before = field.store.rho_acc_for_pair(pred_tid, token_id).unwrap_or(0.0);
                // Expected ρ_acc after: ρ_before + η·ρₙ·(1−ρ_before)
                let delta = ETA * rho_n * (1.0 - rho_before);
                let rho_expected = (rho_before + delta).min(1.0);

                println!();
                println!("  EDGE  ({}, \"{}\", id={}) → ({}, \"{}\", id={})",
                         pred_pos, pred_word, pred_tid,
                         position, word, token_id);
                vec_summary(&a_vec, &format!("A[{}→{}]", pred_pos, position));
                println!("    χ[e]  = max|A[e][k]| = {:.6}  (source: operator_a_to_rho_n)", chi);
                println!("    ρₙ    = ρ_base·χ/(χ₀+χ) = {:.4}·{:.6}/(1.0+{:.6}) = {:.6}",
                         RHO_BASE, chi, chi, rho_n);
                println!("    ε[e]  = |ι[src]−ι[tgt]|  →  UNAVAILABLE (ι not set)");
                println!("           accumulation uses base rate η (ambient)");
                println!("    ρ_acc before: {:.6}", rho_before);
                println!("    ρ_acc update: {:.4} + {:.4}·{:.6}·(1−{:.6}) = {:.4} + {:.6}",
                         rho_before, ETA, rho_n, rho_before, rho_before, delta);
                println!("    ρ_acc after (expected): {:.6}", rho_expected);

                let admitted_after = rho_expected >= THETA_MIN;
                if admitted_after {
                    println!("    admission (θ_min={:.2}): YES  ← ADMITTED TO TOPOLOGY",
                             THETA_MIN);
                } else {
                    println!("    admission (θ_min={:.2}): NO   ρ_acc={:.6} < θ_min={:.4}  [no admitted edge]",
                             THETA_MIN, rho_expected, THETA_MIN);
                }
            }
        }

        // Actually call arrive() — this performs the accumulation
        let locus = Locus::new(position, token_id);
        field.arrive(locus, &vocab);
        our_sequence.push((token_id, word.as_str()));

        // Verify ρ_acc values match expectations
        if position > 0 {
            println!();
            println!("  EDGE STORE after arrive() — verification:");
            let window_start = position.saturating_sub(COHERENCE_WINDOW as usize);
            for pred_pos in window_start..position {
                let (pred_tid, pred_word) = our_sequence[pred_pos];
                let actual_rho = field.store.rho_acc_for_pair(pred_tid, token_id).unwrap_or(0.0);
                let admitted = actual_rho >= THETA_MIN;
                println!("    ρ_acc(\"{}\",\"{}\") = {:.6}  admitted(θ=0.05): {}",
                         pred_word, word, actual_rho,
                         if admitted { "YES" } else { "NO [no admitted edge]" });
            }
        }

        // ── Prior binding record for all pairs in window ───────────────────────
        println!();
        println!("▶ PRIOR BINDING RECORD (full EdgeStore at frontier={})", position + 1);
        let all_pairs: Vec<(usize, usize)> = field.store.all_admitted_pairs_at_threshold(0.0);
        if all_pairs.is_empty() {
            println!("  EdgeStore: empty (novel token; no prior binding)");
        } else {
            println!("  {} total pair(s) in EdgeStore:", all_pairs.len());
            // Build reverse lookup from token_id to word
            let id_to_word: std::collections::HashMap<usize, &str> =
                trace_tokens.iter().zip(trace_ids.iter())
                    .map(|(w, &id)| (id, w.as_str()))
                    .collect();
            let mut pair_list: Vec<_> = all_pairs.iter().map(|&(s, t)| {
                let rho = field.store.rho_acc_for_pair(s, t).unwrap_or(0.0);
                (s, t, rho)
            }).collect();
            pair_list.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            for (s, t, rho) in &pair_list {
                let sw = id_to_word.get(s).copied().unwrap_or("?");
                let tw = id_to_word.get(t).copied().unwrap_or("?");
                let admitted = *rho >= THETA_MIN;
                println!("    ρ_acc(\"{}\",\"{}\") = {:.6}  admitted(θ=0.05): {}",
                         sw, tw, rho,
                         if admitted { "YES" } else { "NO [no admitted edge]" });
            }
        }

        // ── Structural inference at this position (θ=0.0) ─────────────────────
        println!();
        println!("▶ STRUCTURAL INFERENCE at position={} (θ=0.0, all accumulated pairs admitted)",
                 position);

        // Build Loci sequence for DeclaredGraph
        let loci: Vec<Locus> = our_sequence.iter().enumerate()
            .map(|(pos, &(tid, _))| Locus::new(pos, tid))
            .collect();

        let admitted_all = field.store.all_admitted_pairs_at_threshold(0.0);
        let graph = DeclaredGraph::build(&loci, &vocab, &admitted_all);

        if graph.n_edges() == 0 {
            println!("  DeclaredGraph (θ=0.0): empty — no pairs accumulated yet.");
            println!("  Response (θ=0.0): zero vector  [novel token; declared admissible state]");
            println!("  Response (θ=0.05): zero vector [no admitted edge; same declared state]");
        } else {
            println!("  DeclaredGraph (θ=0.0): {} edge(s)  {} node(s)",
                     graph.n_edges(), loci.len());

            // Print edges with their A values
            println!();
            println!("  OPERATOR A — directed difference  A[e] = M[src] − M[tgt]");
            for (ei, &(sp, tp)) in graph.edges.iter().enumerate() {
                let sw = our_sequence[sp].1;
                let tw = our_sequence[tp].1;
                vec_summary(&graph.a_field[ei],
                    &format!("A[pos{}→pos{} (\"{}\",\"{}\")]", sp, tp, sw, tw));
            }

            // ρ per node
            let rho_nodes = compute_rho(&graph, loci.len(), RHO_BASE);
            println!();
            println!("  OPERATOR ρ — per-node binding magnitude");
            println!("    χ[i] = max|A[e][k]| over incident edges at node i");
            println!("    ρ[i] = ρ_base·χ[i]/(χ₀+χ[i])  (source: operators.rs compute_rho)");
            for (i, (_, word_i)) in our_sequence.iter().enumerate() {
                // Compute chi for this node manually for the trace
                let mut chi_i = 0.0_f64;
                for (ei, &(sp, tp)) in graph.edges.iter().enumerate() {
                    if sp == i || tp == i {
                        for c in &graph.a_field[ei] {
                            chi_i = chi_i.max(c.abs());
                        }
                    }
                }
                println!("    node {} \"{}\":  χ={:.6}  ρ={:.6}",
                         i, word_i, chi_i, rho_nodes[i]);
            }

            // Operator B
            let bg = operator_b(&graph);
            println!();
            println!("  OPERATOR B — successor accumulation  B[e] = A[e] + Σ_succ A[f]");
            println!("    succ(e = src→tgt) = edges whose source is tgt  (kernel_reference.md §11)");
            for (ei, &(sp, tp)) in graph.edges.iter().enumerate() {
                let sw = our_sequence[sp].1;
                let tw = our_sequence[tp].1;
                let n_succ = graph.succ[ei].len();
                let n_pred = graph.pred[ei].len();
                vec_summary(&bg[ei],
                    &format!("B[pos{}→pos{} (\"{}\",\"{}\")] succ={} pred={}",
                             sp, tp, sw, tw, n_succ, n_pred));
            }

            // Operator R
            let rg = operator_r(&bg, &graph, &rho_nodes);
            println!();
            println!("  OPERATOR R — antisymmetric resolution");
            println!("    R[e] = B[e] + ρ[src(e)]·(Σ_succ B[f] − Σ_pred B[f])");
            println!("    R[e] ∈ ℝ^dim.  Signed values preserved; no clipping or normalisation.");
            println!("    Note: ρ_base=1.0 and multiple successors in B make R[e] components");
            println!("    well-bounded (finite, in D) but NOT bounded to [0,1]^dim.");
            println!("    Streaming.rs line 302 comment 'mean is bounded' means O(1) growth,");
            println!("    not [0,1] confinement.  See correction in DECLARATION.md.");
            for (ei, &(sp, tp)) in graph.edges.iter().enumerate() {
                let sw = our_sequence[sp].1;
                let tw = our_sequence[tp].1;
                vec_summary(&rg[ei],
                    &format!("R[pos{}→pos{} (\"{}\",\"{}\")] ρ_node[{}]={:.6}",
                             sp, tp, sw, tw, sp, rho_nodes[sp]));
                // Check if any component goes outside [-1,1]
                let outside: Vec<(usize, f64)> = rg[ei].iter().enumerate()
                    .filter(|(_, &v)| v < -1.0 || v > 1.0)
                    .map(|(k, &v)| (k, v))
                    .collect();
                if !outside.is_empty() {
                    println!("    ⚠  components outside [−1,1]: {}",
                             outside.iter().map(|(k,v)| format!("[{}]={:+.6}", k, v))
                                     .collect::<Vec<_>>().join(", "));
                    println!("       This is CORRECT.  R[e] ∈ ℝ^dim.  The original [0,1]^dim");
                    println!("       bound claim applies to M (the measurement domain), not to");
                    println!("       derived operators.  No clipping or normalisation applied.");
                }
            }

            // Response: mean R over edges whose tgt == current position
            println!();
            println!("  RESPONSE at position={} — aggregation rule:", position);
            println!("    output = (1/k) · Σ R[e]  for all edges e with tgt = position={}",
                     position);
            println!("    (mean, not sum; O(1) with k; source: streaming.rs inference_at)");

            let mut output = vec![0.0_f64; DIM];
            let mut n_at_target = 0usize;
            let n_loci = loci.len();
            for (ei, &(_, tp)) in graph.edges.iter().enumerate() {
                if tp == n_loci - 1 {   // last locus = current position
                    for d in 0..DIM {
                        output[d] += rg[ei][d];
                    }
                    n_at_target += 1;
                }
            }
            if n_at_target > 1 {
                for v in &mut output { *v /= n_at_target as f64; }
            }

            if n_at_target == 0 {
                println!("    No edges ending at position {} in DeclaredGraph (θ=0.0).", position);
                println!("    Response: zero vector [no admitted edge at target]");
            } else {
                println!("    k={} edge(s) aggregated at position={}", n_at_target, position);
                vec_summary(&output, "response (mean R, θ=0.0)");
                println!("    Meaning: relational attention output at \"{}\" given context.",
                         word);
                println!("    This is the vector that W_O would project to vocabulary logits.");

                // Check outside [0,1] for mean too
                let out_range: Vec<(usize, f64)> = output.iter().enumerate()
                    .filter(|(_, &v)| v < 0.0 || v > 1.0)
                    .map(|(k, &v)| (k, v))
                    .collect();
                if !out_range.is_empty() {
                    println!("    ⚠  mean-R components outside [0,1]: {}",
                             out_range.iter().map(|(k,v)| format!("[{}]={:+.6}", k, v))
                                      .collect::<Vec<_>>().join(", "));
                    println!("       Correct.  Mean R ∈ ℝ^dim; signed values preserved.");
                }
            }

            // ── Declared θ_min=0.05 admission ─────────────────────────────────
            println!();
            let admitted_05 = field.store.all_admitted_pairs_at_threshold(THETA_MIN);
            let admitted_05_set: HashSet<(usize, usize)> = admitted_05.into_iter().collect();
            // Count how many edges in the CURRENT WINDOW are admitted at θ_min
            let n_admitted_05 = graph.edges.iter().filter(|&&(sp, tp)| {
                let stid = our_sequence[sp].0;
                let ttid = our_sequence[tp].0;
                admitted_05_set.contains(&(stid, ttid))
            }).count();

            if n_admitted_05 == 0 {
                println!("  RESPONSE under declared θ_min={:.2}: zero vector  [no admitted edge]",
                         THETA_MIN);
                println!("    All ρ_acc < θ_min.  Novel-token signature: declared admissible state.");
                println!("    Accumulation continues.  Topology not yet established.");
            } else {
                println!("  RESPONSE under declared θ_min={:.2}: {} admitted edge(s)",
                         THETA_MIN, n_admitted_05);
            }
        }

        // ── Primitive variable readings ────────────────────────────────────────
        println!();
        println!("▶ PRIMITIVE VARIABLE READINGS at position={}", position);

        println!();
        println!("  ENERGY  ε[e] = |ι[src] − ι[tgt]|");
        println!("  Source: streaming.rs epsilon_for(); accumulation.rs OC-RA-12");
        println!("  Status: UNAVAILABLE — ι not set via StreamingField::with_iota().");
        println!("          Not measured zero.  SemanticM::from_corpus (plain PPMI)");
        println!("          does not assign iota weights.  epsilon_for returns 0.0 as");
        println!("          the ambient fallback, but this is not a measured value;");
        println!("          the ι observable has not been declared for these tokens.");

        println!();
        println!("  CONNECTION — topology hierarchy");
        println!("  Source: semantic.rs SemanticM (M-word level, sentence boundaries, OC-RA-10)");
        println!("  This trace uses a single M level (M-word).  Multi-scale hierarchy");
        println!("  (M_char, M_word, M_sent, M_para) is declared but not activated here.");
        let n_total = field.store.admitted_edges_at_threshold(0.0);
        let n_admitted = field.store.admitted_edges_at_threshold(THETA_MIN);
        println!("  Current topology: {} pair(s) accumulated;  {} admitted at θ_min=0.05",
                 n_total, n_admitted);
        if n_admitted == 0 {
            println!("  Declared graph is empty under θ_min=0.05.  [no admitted edge]");
        } else {
            println!("  Declared graph has {} edge(s) under θ_min=0.05.", n_admitted);
        }

        println!();
        println!("  DEGREES OF FREEDOM — nested confinement hierarchy");
        println!("  Source: declaration.rs COHERENCE_WINDOW, THETA_MIN, RHO_BASE, χ₀");
        let w_start = position.saturating_sub(COHERENCE_WINDOW as usize);
        let positions_in_window = position - w_start + 1;
        let max_edges = positions_in_window * (positions_in_window - 1) / 2;
        println!("  frontier position: {}  coherence window: {} positions",
                 position, COHERENCE_WINDOW as usize);
        println!("  positions in window: {}  max possible directed edges: {}",
                 positions_in_window, max_edges);
        println!("  pairs accumulated (θ=0): {}  admitted (θ=0.05): {}",
                 n_total, n_admitted);
        println!("  confinement levels:");
        println!("    W={}: window suppresses pairs > {} positions apart", COHERENCE_WINDOW, COHERENCE_WINDOW);
        println!("    θ_min={}: admits only pairs with ρ_acc ≥ {}", THETA_MIN, THETA_MIN);
        println!("    ρ_base=1.0, χ₀=1.0: bound ρₙ ∈ [0, ρ_base]=[0, 1.0]");
        println!("    accumulation ceiling: ρ_acc ∈ [0, 1.0] by (1−ρ_acc) factor");

        println!();
        println!("  ANTISYMMETRIC CIRCULATION");
        println!("  STRUCTURALLY UNRESOLVED — see trace header for declaration.");
        println!("  At position {}: {} directed edges, all causal (earlier→later).", position, n_total);
        println!("  No cycles.  Circulation Σ_cycle R[e] has no instances to evaluate.");

        println!();
    } // end token-by-token loop

    // ── Trace summary ──────────────────────────────────────────────────────────
    println!();
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("TRACE COMPLETE — {} tokens processed", N_TRACE);
    println!();
    println!("SUMMARY:");
    println!("  Total pairs accumulated (single pass): {}",
             field.store.admitted_edges_at_threshold(0.0));
    println!("  Pairs admitted at θ_min=0.05: {}",
             field.store.admitted_edges_at_threshold(THETA_MIN));
    println!("  Mean ρ_acc across all pairs: {:.6}", field.store.mean_rho_acc());
    println!();
    println!("DOMAIN BOUND FINDING:");
    println!("  R[e] ∈ ℝ^dim — not bounded to [0,1]^dim or [−1,1]^dim.");
    println!("  streaming.rs line 302 comment 'the mean is bounded' is correct in the");
    println!("  sense that mean-R grows O(1) with k (not O(k) as sum would).  It is");
    println!("  INCORRECT if read as a [0,1]^dim claim.  R components can and do");
    println!("  exceed [−1,1] when succ/pred B-flow magnitudes are large.  This is");
    println!("  expected: M ∈ [0,1]^dim is the measurement domain; derived operators");
    println!("  operate in ℝ^dim.  The domain D = {{x ∈ ℝⁿ | n < ∞, |x[i]| < ∞ ∀i}}");
    println!("  bounds finiteness, not [0,1] confinement for derived quantities.");
    println!();
    println!("NOVEL-TOKEN STATE:");
    println!("  Under single-pass streaming with η=0.1 and 5 tokens, most pairs");
    println!("  accumulate exactly once: ρ_acc = η·ρₙ·1 = 0.1·ρₙ.");
    println!("  For ρ_acc ≥ θ_min=0.05: need ρₙ ≥ 0.5, i.e. χ ≥ 1.0.");
    println!("  Since M ∈ [0,1]^dim, χ = max|A[k]| ≤ 1.0.  χ=1.0 is the maximum");
    println!("  achievable when M[src][k]=1.0 and M[tgt][k]=0.0 (or vice versa).");
    println!("  Whether any pair reaches θ_min after one exposure depends on the");
    println!("  actual PPMI vectors.  Pairs below θ_min: [no admitted edge] state.");
    println!("  This is declared admissible — not an error.  Accumulation continues.");
    println!("═══════════════════════════════════════════════════════════════════════════════");
}
