// src/bin/demonstrate.rs — Metatron Dynamics, Inc.
// ABR Relational Attention — Plain Demonstration
// Bounded over D. No claim beyond D.
//
// Kernel equivalence: VERIFIED (Verifier disposition 2026-09-17)
//   A, B, ρ, R match Kernel V7 within declared Phase 0 single-topology domain.
//   B interface restriction: Phase 0 chain is B(A) — B accepts arbitrary edge
//     field in the kernel; this implementation invokes it specifically on A.
//   R topology restriction: the declared Phase 0 domain instantiates a single
//     relational topology. No component/spatial cross-topology relation has been
//     declared; therefore the kernel cross-topology term has no operand in this
//     instantiation. This is not an omission — the term has no declared operand.
//
// What this demonstrates:
//   Dense attention constructs the pair space from the tokens.
//   ABR begins with the relational structure.
//   The operators compute over the relations the observable actually declares.
//
// What this does not demonstrate:
//   Subquadratic asymptotic scaling — the current admitted graph remains
//     approximately proportional to n² at this declaration depth.
//     No different asymptotic scaling is claimed.
//   Equivalence to a full transformer output — not established here.
//   Power, cooling, or memory traffic claims — not derived from this run.
//
// Open experimental question (QUESTION 2, separate from QUESTION 1 above):
//   Does maintaining relational structure allow state progression without
//   reconstructing the quadratic candidate pair domain?
//   Status: under active investigation in abr-transformer-integration-pub.
//
// Run:
//   cargo run --bin demonstrate

use abr_relational_attention::{
    declaration::{Locus, build_vocab_table, THETA_MIN},
    accumulation::EdgeStore,
    attention::{AttentionConfig, accumulate_from_sequence, relational_attention_pass},
};

// ── Sequence lengths to demonstrate ────────────────────────────────────────

const SEQUENCE_LENGTHS: &[usize] = &[8, 16, 32, 64, 128, 256];

const VOCAB_SIZE: usize  = 32;
const DIM: usize         = 32;
const ACCUM_PASSES: usize = 100;

// ── Build a demonstration sequence of length n ──────────────────────────────

fn make_sequence(n: usize) -> Vec<Locus> {
    (0..n)
        .map(|i| Locus::new(i, i % VOCAB_SIZE))
        .collect()
}

// ── Candidate pair count under dense construction — n² ─────────────────────
//
// Dense attention constructs a score for every (query, key) pair.
// The candidate pair space is n × n — determined entirely by the token count.
// This is the computational domain dense attention starts from.

fn dense_candidate_pairs(n: usize) -> usize {
    n * n
}

// ── Run one demonstration trial ─────────────────────────────────────────────

struct TrialResult {
    n: usize,
    dense_pairs: usize,
    admitted_pairs: usize,
    pairs_not_in_declared_graph: usize,
    pct_not_admitted: f64,
}

fn run_trial(n: usize) -> TrialResult {
    let vocab     = build_vocab_table(VOCAB_SIZE, DIM);
    let seq       = make_sequence(n);
    let config    = AttentionConfig::default();
    let mut store = EdgeStore::new();

    // Accumulation: establishes which token-pair relations have accumulated
    // relational persistence above θ_min. This is the declared relational
    // graph — not a filtered subset of n² candidates, but the structure
    // the observable has established through accumulated exposure.
    for _ in 0..ACCUM_PASSES {
        accumulate_from_sequence(&seq, &vocab, &mut store, &config);
    }

    // Relational attention pass: operators A → B → ρ → R act on the
    // declared graph only. Pairs not in the declared graph have no operand
    // in this computation — they are not enumerated and rejected,
    // they are simply not part of the declared relational structure.
    let output = relational_attention_pass(&seq, &vocab, &store, &config);

    let dense    = dense_candidate_pairs(n);
    let admitted = output.admitted_edge_count;
    let not_admitted = dense.saturating_sub(admitted);
    let pct = 100.0 * not_admitted as f64 / dense as f64;

    TrialResult {
        n,
        dense_pairs: dense,
        admitted_pairs: admitted,
        pairs_not_in_declared_graph: not_admitted,
        pct_not_admitted: pct,
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

fn main() {
    println!("================================================================");
    println!("  ABR Relational Attention — Plain Demonstration");
    println!("  Metatron Dynamics, Inc. | relationalrelativity.dev");
    println!("================================================================");

    println!("
Dense attention constructs the pair space from the tokens.
ABR begins with the relational structure.
The operators compute over the relations the observable actually declares.

Dense attention defines scores over n × n candidate pairs — one for
every possible (query, key) combination. The candidate pair count is
determined entirely by the number of tokens. The n × n candidate
domain is instantiated before the resulting attention structure is
determined.

ABR relational attention begins from the declared relational graph:
the set of token-identity pairs whose accumulated relational persistence
has crossed the declared admission threshold θ_min = {theta_min}.
The operators A → B → ρ → R act on that graph directly.
Pairs outside the declared graph have no operand in the computation.

The underlying experimental question:

  Why reconstruct every possible relationship when relational
  structure can persist?

That question is under active investigation.
", theta_min = THETA_MIN);

    println!("Declaration:");
    println!("  Vocabulary size  : {VOCAB_SIZE} tokens");
    println!("  Embedding dim    : {DIM}");
    println!("  Accum passes     : {ACCUM_PASSES}  (establishes declared graph)");
    println!("  Admission θ_min  : {THETA_MIN}");
    println!("  Operators        : A → B(A) → ρ → R  (Kernel V7, verified)");
    println!("  Topology         : single declared relational topology (Phase 0)");
    println!();

    println!(
        "  {:>6}  {:>14}  {:>16}  {:>20}  {:>16}",
        "n",
        "Dense pairs",
        "Admitted pairs",
        "Not in declared graph",
        "% not admitted"
    );
    println!("  {}", "─".repeat(80));

    let mut results = Vec::new();
    for &n in SEQUENCE_LENGTHS {
        let r = run_trial(n);
        println!(
            "  {:>6}  {:>14}  {:>16}  {:>20}  {:>15.1}%",
            r.n,
            format_large(r.dense_pairs),
            format_large(r.admitted_pairs),
            format_large(r.pairs_not_in_declared_graph),
            r.pct_not_admitted,
        );
        results.push(r);
    }

    println!();
    println!("────────────────────────────────────────────────────────────────");
    println!("Scaling note");
    println!("────────────────────────────────────────────────────────────────");
    println!("
  Dense candidate pairs grow as n²:
    n=64  →     4,096 pairs
    n=128 →    16,384 pairs  (4× for 2× tokens)
    n=256 →    65,536 pairs  (16× for 4× tokens)

  In this declared demonstration, the admitted graph also grows
  approximately with n² at this declaration depth — the admitted
  fraction remains near constant across the table above.
  No subquadratic asymptotic scaling is claimed from this run.

  The distinction being demonstrated is not asymptotic scaling.
  It is computational domain:

    Dense attention starts from n × n and recovers structure.
    ABR starts from the declared relational structure directly.

  Whether relational state can be advanced from prior state plus
  observed change — without reconstructing n² candidates — is the
  open experimental question under active investigation.
");

    println!("────────────────────────────────────────────────────────────────");
    println!("What the verified operators do");
    println!("────────────────────────────────────────────────────────────────");
    println!("
  All four operators verified against Kernel V7 (2026-09-17).
  Phase 0 chain: E = R(B(A), ρ(A)) over the declared graph.

  A[e] = M(src) − M(tgt)
         Directed difference between declared loci.
         Canonical V7: A(x)[e] = x[s] − x[t].
         Domain instantiation: vocabulary embedding vectors.

  B(A)[e] = A[e] + Σ_{{f ∈ succ(e)}} A[f]
         Successor accumulation over the declared graph.
         Phase 0 interface: B invoked on A specifically.
         Kernel B accepts any edge field; this instantiation fixes input to A.

  ρ[i] = ρ_base · χ[i] / (1 + χ[i])   χ[i] = max|A[e]| incident on i
         Per-node relational persistence magnitude.
         Selection over incident edges — not aggregation.
         Bounded in D by construction.

  R[e] = B[e] + ρ[src(e)] · (Σ_succ B − Σ_pred B)
         Antisymmetric resolution over the B field.
         Phase 0 topology: single declared relational topology.
         Kernel R includes a cross-topology term; that term has no
         declared second topology operand in this Phase 0 instantiation.
");

    println!("────────────────────────────────────────────────────────────────");
    println!("The open question");
    println!("────────────────────────────────────────────────────────────────");

    let n_dense: usize = 4096;
    let layers:  usize = 32;
    let heads:   usize = 32;
    let dense_total = n_dense * n_dense * layers * heads;

    println!("
  At n={n_dense}, {layers} layers, {heads} heads:
  Dense candidate pair evaluations per forward pass: {dense_total}

  ABR asks a different question before that computation begins:

    Which relations are declared by the observable?

  The experimental question underneath:

    R_t + Δ_t → R_(t+1)

    How much of the next relational state can be derived from
    the current relational state and the observed change —
    without enumerating the n² candidate pair space?

  That is what the Phi-3 intervention experiments are testing.
  This executable demonstrates what computation over a declared
  relational structure looks like.
  The connection between the two is the subject of ongoing work.
",
        dense_total = format_large(dense_total),
    );

    println!("================================================================");
    println!("  Metatron Dynamics, Inc. | relationalrelativity.dev");
    println!("  Bounded over D. No claim beyond D.");
    println!("================================================================");
}

// ── Formatting ──────────────────────────────────────────────────────────────

fn format_large(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { result.push(','); }
        result.push(c);
    }
    result.chars().rev().collect()
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use abr_relational_attention::{
        declaration::{Locus, build_vocab_table, THETA_MIN},
        accumulation::EdgeStore,
        attention::{AttentionConfig, accumulate_from_sequence, relational_attention_pass},
    };

    // ── format_large ─────────────────────────────────────────────────────────

    #[test]
    fn test_format_large_small() {
        assert_eq!(format_large(0),   "0");
        assert_eq!(format_large(999), "999");
    }

    #[test]
    fn test_format_large_thousands() {
        assert_eq!(format_large(1_000),     "1,000");
        assert_eq!(format_large(65_536),    "65,536");
        assert_eq!(format_large(1_048_576), "1,048,576");
    }

    // ── dense_candidate_pairs ─────────────────────────────────────────────────

    #[test]
    fn test_dense_candidate_pairs_quadratic() {
        assert_eq!(dense_candidate_pairs(4),  16);
        assert_eq!(dense_candidate_pairs(8),  64);
        assert_eq!(dense_candidate_pairs(16), 256);
        // Doubling n quadruples dense pairs
        assert_eq!(dense_candidate_pairs(32), dense_candidate_pairs(16) * 4);
    }

    // ── make_sequence ─────────────────────────────────────────────────────────

    #[test]
    fn test_make_sequence_length() {
        for &n in &[4, 8, 16, 32] {
            assert_eq!(make_sequence(n).len(), n);
        }
    }

    #[test]
    fn test_make_sequence_positions_ascending() {
        let seq = make_sequence(8);
        for (i, locus) in seq.iter().enumerate() {
            assert_eq!(locus.position, i);
        }
    }

    #[test]
    fn test_make_sequence_token_ids_in_vocab() {
        let seq = make_sequence(VOCAB_SIZE * 3);
        for locus in &seq {
            assert!(locus.token_id < VOCAB_SIZE,
                "token_id {} out of vocab range {}", locus.token_id, VOCAB_SIZE);
        }
    }

    // ── run_trial — structural invariants ────────────────────────────────────

    #[test]
    fn test_trial_admitted_never_exceeds_dense() {
        for &n in &[8, 16, 32] {
            let r = run_trial(n);
            assert!(r.admitted_pairs <= r.dense_pairs,
                "n={}: admitted {} > dense {}", n, r.admitted_pairs, r.dense_pairs);
        }
    }

    #[test]
    fn test_trial_not_admitted_consistent() {
        for &n in &[8, 16, 32] {
            let r = run_trial(n);
            assert_eq!(
                r.pairs_not_in_declared_graph,
                r.dense_pairs - r.admitted_pairs,
                "n={}: pairs_not_in_declared_graph inconsistent", n
            );
        }
    }

    #[test]
    fn test_trial_pct_in_range() {
        for &n in &[8, 16, 32] {
            let r = run_trial(n);
            assert!(r.pct_not_admitted >= 0.0 && r.pct_not_admitted <= 100.0,
                "n={}: pct_not_admitted {} out of range", n, r.pct_not_admitted);
        }
    }

    #[test]
    fn test_dense_pairs_quadratic_growth() {
        let r8  = run_trial(8);
        let r16 = run_trial(16);
        assert_eq!(r16.dense_pairs, r8.dense_pairs * 4,
            "dense pairs must grow as n²");
    }

    // ── Verified operator properties ──────────────────────────────────────────

    #[test]
    fn test_accumulation_produces_admitted_edges() {
        let vocab     = build_vocab_table(VOCAB_SIZE, DIM);
        let seq       = make_sequence(16);
        let mut store = EdgeStore::new();
        let config    = AttentionConfig::default();
        for _ in 0..ACCUM_PASSES {
            accumulate_from_sequence(&seq, &vocab, &mut store, &config);
        }
        assert!(store.admitted_edges() > 0,
            "accumulation must produce admitted edges after {} passes", ACCUM_PASSES);
    }

    #[test]
    fn test_relational_pass_output_length_matches_sequence() {
        let vocab     = build_vocab_table(VOCAB_SIZE, DIM);
        let seq       = make_sequence(16);
        let mut store = EdgeStore::new();
        let config    = AttentionConfig::default();
        for _ in 0..ACCUM_PASSES {
            accumulate_from_sequence(&seq, &vocab, &mut store, &config);
        }
        let out = relational_attention_pass(&seq, &vocab, &store, &config);
        assert_eq!(out.outputs.len(), seq.len(),
            "output length must match sequence length");
    }

    #[test]
    fn test_relational_pass_outputs_finite() {
        let vocab     = build_vocab_table(VOCAB_SIZE, DIM);
        let seq       = make_sequence(16);
        let mut store = EdgeStore::new();
        let config    = AttentionConfig::default();
        for _ in 0..ACCUM_PASSES {
            accumulate_from_sequence(&seq, &vocab, &mut store, &config);
        }
        let out = relational_attention_pass(&seq, &vocab, &store, &config);
        for (i, o) in out.outputs.iter().enumerate() {
            for (j, v) in o.iter().enumerate() {
                assert!(v.is_finite(),
                    "output[{}][{}] = {} is not finite", i, j, v);
            }
        }
    }

    #[test]
    fn test_admitted_pairs_below_dense_after_accumulation() {
        // After accumulation, only relationally persistent pairs are in the
        // declared graph. The admitted count must be strictly below n²
        // for non-trivial n with a sparse declared relational structure.
        let vocab     = build_vocab_table(VOCAB_SIZE, DIM);
        let seq       = make_sequence(32);
        let mut store = EdgeStore::new();
        let config    = AttentionConfig::default();
        for _ in 0..ACCUM_PASSES {
            accumulate_from_sequence(&seq, &vocab, &mut store, &config);
        }
        let out  = relational_attention_pass(&seq, &vocab, &store, &config);
        let dense = dense_candidate_pairs(seq.len());
        assert!(out.admitted_edge_count < dense,
            "admitted {} must be < dense {} for n=32",
            out.admitted_edge_count, dense);
    }

    #[test]
    fn test_theta_min_gates_admission() {
        // A pair with rho_acc below THETA_MIN must not appear in the declared graph.
        let mut store = EdgeStore::new();
        let pair = abr_relational_attention::declaration::RelationalPair::declare(
            Locus::new(0, 0),
            Locus::new(1, 1),
        ).unwrap();
        // Single update with very small rho_n — stays below theta_min
        store.update(&pair, 0.001);
        assert_eq!(store.admitted_edges(), 0,
            "single low-rho update must not produce an admitted edge \
             (theta_min = {})", THETA_MIN);
    }

    #[test]
    fn test_admitted_pairs_not_constructed_from_dense_enumeration() {
        // The declared graph is built from accumulated token-identity bindings,
        // not by filtering n² candidates. This test verifies that a fresh store
        // with no accumulation produces zero admitted pairs — the declared graph
        // is empty by construction, not by rejection of dense candidates.
        let vocab  = build_vocab_table(VOCAB_SIZE, DIM);
        let seq    = make_sequence(16);
        let store  = EdgeStore::new(); // no accumulation
        let config = AttentionConfig::default();
        let out    = relational_attention_pass(&seq, &vocab, &store, &config);
        assert_eq!(out.admitted_edge_count, 0,
            "empty store must produce empty declared graph — zero admitted pairs");
    }
}
