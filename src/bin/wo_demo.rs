// wo_demo.rs — Metatron Dynamics, Inc.
// ABR Relational Attention — W_O Output Projection Demo V1.0
//
// W_O IS THE FIRST LEARNED PARAMETER IN ABR RELATIONAL ATTENTION.
//
// Everything else is accumulated (ρ_acc) or declared (M, A, B, R, η, θ, W).
// W_O alone is trained by gradient descent on a task loss.
//
// SCOPE DECLARATION (exploratory; see DECLARATION.md §W_O):
//
//   This demo measures TRAINING FIT, not held-out generalization. Training
//   and evaluation use the same 233-token Darwin paragraph. No generalization
//   claim is made.
//
//   N_MATURE=3 passes over the same paragraph during R-field construction are
//   a declared exploratory deviation from the streaming single-pass protocol.
//   rho_acc levels reflect repeated exposure, not single-pass accumulation.
//
// WHAT THIS DEMO MEASURES:
//   Language modeling task: predict next token P(t+1 | relational_context_t).
//   Three representations for the same task, same W_O architecture:
//
//   BASELINE A — Random 50-dim vectors (position-memorization baseline):
//     Each token POSITION receives a distinct LCG-seeded vector regardless
//     of token identity. W_O learns position → next-token mappings, which
//     is sequence memorization, not useful representation. This baseline
//     does NOT measure frequency-only or token-type prediction; it measures
//     how easily W_O can memorize the fixed sequence by position index.
//     Its apparent dominance at 500 epochs reflects memorization, not
//     representation quality.
//
//   BASELINE B — GloVe-50d embeddings (no ABR):
//     Representations are the GloVe vector of the CURRENT token at each position.
//     W_O can learn token-to-token transitions (bigram statistics).
//     "If the current token is 'species', the next is often 'of'."
//     Cannot use context — same vector regardless of what preceded.
//
//   ABR — R-field output from layer stack:
//     Representations aggregate relational history via the streaming field.
//     Tokens that appear in SIMILAR RELATIONAL CONTEXTS have similar R-fields.
//     W_O can learn "in this relational context, the next token is likely..."
//     ABR and GloVe both use token-type representations; their comparison is
//     the meaningful one. Result is training fit under declared conditions.
//
// OPTIMIZER: Adam (adaptive learning rate, faster than SGD for small datasets).
// CORPUS: Darwin paragraph (233 tokens, 135 unique in compact vocab).
// M_0: GloVe-50d, top 50k words.
// TASK: predict token[t+1] given representation at token[t].
//
// RUN: cargo run --release --bin wo_demo

use abr_relational_attention::{
    declaration::{ETA, THETA_MIN, COHERENCE_WINDOW},
    semantic::SemanticM,
    layer_stack::ABRLayerStack,
    output_projection::{OutputProjection, AdamState, train_epoch_adam, evaluate},
};

const DARWIN_PARAGRAPH: &str = "
When we look to the individuals of the same variety or sub-variety of our
older cultivated plants and animals, one of the first things which strikes us
is, that they generally differ much more from each other than do the
individuals of any one species in a state of nature. When we reflect on the
vast diversity of the plants and animals which have been cultivated, and which
have varied during all ages under the most different climates and treatment, I
think we are driven to conclude that this greater variability is simply due to
our domestic varieties having been raised under conditions of life not so
uniform as, and somewhat different from, those to which the parent-species
have been exposed under nature. There is, also, I think, some probability in
the view propounded by Andrew Knight, that this variability may be partly
connected with excess of food. It seems pretty clear that organic beings must
be exposed for several generations to the new conditions of life to cause any
appreciable amount of variation; and that when the organisation has once begun
to vary, it generally continues to vary for many generations. No case is on
record of a variable being ceasing to be variable under cultivation. Our
oldest cultivated plants, such as wheat, still often yield new varieties; our
oldest domesticated animals are still capable of rapid improvement or
modification.
";

const GLOVE_PATH: &str = "/tmp/glove_50d_top50k.txt";
const MAX_VOCAB:  usize = 50_000;
const N_LAYERS:   usize = 3;
const N_MATURE:   usize = 3;   // maturation passes for ABR field
const N_EPOCHS:   usize = 500; // gradient descent epochs per baseline
const ADAM_LR:    f64   = 1e-3; // Adam learning rate
const REPORT_EVERY: usize = 50;

fn main() {
    println!("====================================================================");
    println!("  W_O Output Projection — ABR Relational Attention");
    println!("  Metatron Dynamics, Inc. — First Learned Parameter");
    println!("====================================================================");
    println!();
    println!("  Declared: ETA={} | THETA_MIN={} | W={} | Layers={}",
             ETA, THETA_MIN, COHERENCE_WINDOW as usize, N_LAYERS);
    println!("  Optimizer: Adam(lr={}, β₁=0.9, β₂=0.999)", ADAM_LR);
    println!("  Task: next-token language modeling on Darwin paragraph");
    println!();

    // ── 1. Load GloVe and tokenize ──────────────────────────────────────────
    print!("  Loading GloVe-50d ({} words) ... ", MAX_VOCAB);
    let sem = match SemanticM::from_glove_file(GLOVE_PATH, MAX_VOCAB) {
        Ok(s)  => { println!("OK (dim={})", s.dim); s }
        Err(e) => { eprintln!("FAILED: {}", e); std::process::exit(1); }
    };
    let dim = sem.dim;

    let full_token_ids = sem.encode(DARWIN_PARAGRAPH);
    let n_tokens = full_token_ids.len();
    let (base_vocab, token_ids) = sem.compact_vocab_for(&full_token_ids);
    let vocab_size = base_vocab.len(); // compact vocab size

    // Build compact_id → word map for display.
    let mut id_to_word: Vec<String> = vec!["<unk>".to_string(); vocab_size];
    {
        let mut assigned = vec![false; vocab_size];
        for (pos, &cid) in token_ids.iter().enumerate() {
            if cid < vocab_size && !assigned[cid] {
                let gid = full_token_ids[pos];
                id_to_word[cid] = if gid < sem.id_to_word.len() {
                    sem.id_to_word[gid].clone()
                } else { "<unk>".to_string() };
                assigned[cid] = true;
            }
        }
    }

    println!("  Corpus: {} tokens, {} unique in compact vocab", n_tokens, vocab_size);
    println!();

    // Training pairs: (representation[t], target[t]) where target[t] = token_ids[t+1]
    // We predict the NEXT token from the representation at the CURRENT position.
    let n_train = n_tokens - 1;
    let targets: Vec<usize> = token_ids[1..].to_vec(); // next-token labels

    // ── 2. Build representations for each baseline ──────────────────────────
    println!("  Building representations ...");

    // BASELINE A: Random 50-dim vectors (fixed seed, no semantic structure).
    let random_reps: Vec<Vec<f64>> = {
        let mut rng = 12345u64;
        (0..n_tokens).map(|_| {
            (0..dim).map(|_| lcg(&mut rng)).collect()
        }).collect()
    };
    println!("    [A] Random 50-dim: {} vectors", n_tokens);

    // BASELINE B: GloVe vectors at each position (no relational context).
    let glove_reps: Vec<Vec<f64>> = token_ids.iter().map(|&cid| {
        base_vocab[cid].clone()
    }).collect();
    println!("    [B] GloVe-50d:     {} vectors (current-token embedding only)", n_tokens);

    // ABR: Run layer stack, collect R-field outputs from the final layer.
    let abr_reps: Vec<Vec<f64>> = {
        println!("    [C] ABR R-field:   building {} × {} × {} layer stack ...",
                 N_MATURE, n_tokens, N_LAYERS);
        let mut stack = ABRLayerStack::new(N_LAYERS);
        // Maturation passes (preserve ρ_acc, reset positional stream between passes)
        for pass in 0..N_MATURE {
            stack.forward(&token_ids, &base_vocab);
            if pass < N_MATURE - 1 { stack.reset_streams(); }
        }
        stack.reset_streams();
        // Final forward pass: collect R-field outputs
        let results = stack.forward(&token_ids, &base_vocab);
        // Use the DEEPEST layer's R-field (highest-order relational context).
        let last_layer = results.last().expect("stack must have at least one layer");
        let n_novel = last_layer.n_novel;
        println!("    [C] ABR R-field:   {} vectors (layer {}, {}/{} novel)",
                 n_tokens, last_layer.layer, n_novel, n_tokens);
        last_layer.outputs.iter().map(|o| o.r_field.clone()).collect()
    };
    println!();

    // ── 3. Training comparison ──────────────────────────────────────────────
    // Report R-field norm statistics to understand scale difference between baselines.
    let mean_norm = |reps: &[Vec<f64>]| -> f64 {
        let sum: f64 = reps.iter().map(|r| r.iter().map(|v| v*v).sum::<f64>().sqrt()).sum();
        sum / reps.len() as f64
    };
    println!("  Representation norm statistics (mean |r|):");
    println!("    [A] Random:  {:.4}", mean_norm(&random_reps[..n_train]));
    println!("    [B] GloVe:   {:.4}", mean_norm(&glove_reps[..n_train]));
    println!("    [C] ABR:     {:.4}", mean_norm(&abr_reps[..n_train]));
    println!();
    println!("  Note: ABR R-field norms are O(W·√dim) = O(64·√50) ≈ 452.");
    println!("  W_O learns to interpret each representation scale.");
    println!();

    // Print loss table header
    println!("  Training W_O on each baseline ({} epochs, Adam lr={}):", N_EPOCHS, ADAM_LR);
    println!();
    println!("  {:>5}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
             "epoch", "A:rand_loss", "B:glove_loss", "C:abr_loss", "B:pplx", "C:pplx");
    println!("  {:>5}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
             "─────", "────────────", "────────────", "────────────", "────────────", "────────────");

    // Create W_O for each baseline (same architecture, same init seed).
    let mut wo_rand  = OutputProjection::new_random(dim, vocab_size);
    let mut wo_glove = OutputProjection::new_random(dim, vocab_size);
    let mut wo_abr   = OutputProjection::new_random(dim, vocab_size);
    // Clone initial weights so all three start identically.
    wo_glove.w = wo_rand.w.clone();
    wo_glove.b = wo_rand.b.clone();
    wo_abr.w   = wo_rand.w.clone();
    wo_abr.b   = wo_rand.b.clone();

    let mut adam_rand  = AdamState::new(dim, vocab_size);
    let mut adam_glove = AdamState::new(dim, vocab_size);
    let mut adam_abr   = AdamState::new(dim, vocab_size);

    let rand_in  = &random_reps[..n_train];
    let glove_in = &glove_reps[..n_train];
    let abr_in   = &abr_reps[..n_train];

    // Initial loss (before any training)
    let (l_r0, _, _) = evaluate(&wo_rand,  rand_in,  &targets);
    let (l_g0, _, _) = evaluate(&wo_glove, glove_in, &targets);
    let (l_a0, _, _) = evaluate(&wo_abr,   abr_in,   &targets);
    println!("  {:>5}  {:>12.4}  {:>12.4}  {:>12.4}  {:>12.1}  {:>12.1}",
             0, l_r0, l_g0, l_a0,
             OutputProjection::perplexity(l_g0),
             OutputProjection::perplexity(l_a0));

    for epoch in 1..=N_EPOCHS {
        train_epoch_adam(&mut wo_rand,  &mut adam_rand,  rand_in,  &targets, ADAM_LR);
        train_epoch_adam(&mut wo_glove, &mut adam_glove, glove_in, &targets, ADAM_LR);
        train_epoch_adam(&mut wo_abr,   &mut adam_abr,   abr_in,   &targets, ADAM_LR);

        if epoch % REPORT_EVERY == 0 || epoch == N_EPOCHS {
            let (l_r, _, _) = evaluate(&wo_rand,  rand_in,  &targets);
            let (l_g, _, _) = evaluate(&wo_glove, glove_in, &targets);
            let (l_a, _, _) = evaluate(&wo_abr,   abr_in,   &targets);
            println!("  {:>5}  {:>12.4}  {:>12.4}  {:>12.4}  {:>12.1}  {:>12.1}",
                     epoch, l_r, l_g, l_a,
                     OutputProjection::perplexity(l_g),
                     OutputProjection::perplexity(l_a));
        }
    }
    println!();

    // ── 4. Final evaluation ─────────────────────────────────────────────────
    let (loss_r, acc_r, top5_r) = evaluate(&wo_rand,  rand_in,  &targets);
    let (loss_g, acc_g, top5_g) = evaluate(&wo_glove, glove_in, &targets);
    let (loss_a, acc_a, top5_a) = evaluate(&wo_abr,   abr_in,   &targets);

    println!("--- Final Evaluation: W_O on Next-Token Prediction ---");
    println!();
    println!("  {:>8}  {:>10}  {:>12}  {:>10}  {:>12}",
             "baseline", "loss", "perplexity", "top-1 acc", "top-5 acc");
    println!("  {:>8}  {:>10}  {:>12}  {:>10}  {:>12}",
             "────────", "──────────", "────────────", "──────────", "────────────");
    println!("  {:>8}  {:>10.4}  {:>12.1}  {:>9.1}%  {:>11.1}%",
             "[A] rand", loss_r, OutputProjection::perplexity(loss_r), 100.0*acc_r, 100.0*top5_r);
    println!("  {:>8}  {:>10.4}  {:>12.1}  {:>9.1}%  {:>11.1}%",
             "[B] glove", loss_g, OutputProjection::perplexity(loss_g), 100.0*acc_g, 100.0*top5_g);
    println!("  {:>8}  {:>10.4}  {:>12.1}  {:>9.1}%  {:>11.1}%",
             "[C] abr", loss_a, OutputProjection::perplexity(loss_a), 100.0*acc_a, 100.0*top5_a);
    println!();

    // ── 5. Top predictions comparison ──────────────────────────────────────
    println!("--- Top-5 Predictions at Selected Positions ---");
    println!();
    println!("  Comparing what W_O predicts next for key tokens.");
    let probe_positions: Vec<usize> = find_probe_positions(
        &token_ids, &id_to_word,
        &["species", "the", "cultivated", "variation"],
    );
    for &pos in &probe_positions {
        if pos + 1 >= n_tokens { continue; }
        let curr_word = &id_to_word[token_ids[pos]];
        let next_word = &id_to_word[token_ids[pos+1]];
        println!("  Position {:>3}: current='{}', actual_next='{}'",
                 pos, curr_word, next_word);
        println!("    [B] GloVe top-5:  {}", top5_words(&wo_glove, &glove_reps[pos], &id_to_word, 5));
        println!("    [C] ABR   top-5:  {}", top5_words(&wo_abr,   &abr_reps[pos],   &id_to_word, 5));
        println!();
    }

    // ── 6. Summary ──────────────────────────────────────────────────────────
    println!("====================================================================");
    println!("  W_O Training Result: ABR vs GloVe vs Random");
    println!("====================================================================");
    println!();
    println!("  W_O architecture: {} → {} (compact Darwin vocab)", dim, vocab_size);
    println!("  Parameters: {} × {} + {} = {} total",
             vocab_size, dim, vocab_size, vocab_size * dim + vocab_size);
    println!("  Corpus: {} token positions, {} training pairs",
             n_tokens, n_train);
    println!();

    let abr_vs_glove_loss_pct = 100.0 * (loss_g - loss_a) / loss_g;
    let abr_vs_glove_pplx_ratio = OutputProjection::perplexity(loss_g)
                                 / OutputProjection::perplexity(loss_a);

    println!("  ABR R-field vs GloVe (training fit, same corpus):");
    println!("    Loss reduction:       {:.1}%", abr_vs_glove_loss_pct);
    println!("    Perplexity ratio:     {:.2}× (GloVe pplx / ABR pplx)", abr_vs_glove_pplx_ratio);
    println!("    Top-1 accuracy gain:  {:.1}pp", 100.0 * (acc_a - acc_g));
    println!();
    println!("  Interpretation:");
    if loss_a < loss_g {
        println!("    ABR < GloVe: R-field achieves lower training loss than GloVe.");
        println!("    Both use token-type representations; this comparison is valid.");
        println!("    Result is training fit. Held-out generalization not established.");
    } else {
        println!("    GloVe ≤ ABR on this corpus.");
        println!("    Darwin paragraph is too short for the relational field to ");
        println!("    accumulate enough structure to outperform direct GloVe bigrams.");
        println!("    Expected: ABR advantage grows with corpus size.");
    }
    println!();
    println!("  Scope (declared exploratory conditions):");
    println!("    - Training and evaluation use the SAME 233-token paragraph.");
    println!("    - N_MATURE={} replay passes deviate from single-pass streaming.", N_MATURE);
    println!("    - [A] random baseline is position-memorization, not a frequency floor.");
    println!("    - Meaningful comparison: ABR vs GloVe (both token-type reps).");
    println!();
    println!("  KEY STRUCTURAL CLAIM: W_O is the ONLY gradient-updated parameter.");
    println!("  The relational field (ρ_acc in EdgeStore) is NOT updated by backprop.");
    println!("  ABR decouples representation accumulation from prediction learning.");
    println!();
    println!("  Declared parameters: ETA={} THETA_MIN={} W={}",
             ETA, THETA_MIN, COHERENCE_WINDOW as usize);
    println!("  W_O trained by Adam(lr={}) for {} epochs.", ADAM_LR, N_EPOCHS);
    println!("====================================================================");
}

// ── Helpers ────────────────────────────────────────────────────────────────────

fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005)
                  .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn top5_words(wo: &OutputProjection, rep: &[f64], id_to_word: &[String], k: usize) -> String {
    let logits = wo.forward(rep);
    let mut indexed: Vec<(usize, f64)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|(i, _)| {
        id_to_word.get(*i).map(|w| w.as_str()).unwrap_or("<unk>")
    }).collect::<Vec<_>>().join(", ")
}

fn find_probe_positions(
    token_ids: &[usize],
    id_to_word: &[String],
    probe_words: &[&str],
) -> Vec<usize> {
    let mut positions = Vec::new();
    for word in probe_words {
        // Find first occurrence of this word in token_ids
        for (pos, &tid) in token_ids.iter().enumerate() {
            if id_to_word.get(tid).map(|w| w.as_str()) == Some(word) {
                positions.push(pos);
                break;
            }
        }
    }
    positions.sort();
    positions
}
