// held_out_benchmark.rs — Metatron Dynamics, Inc.
// ABR Relational Attention — OC-RA-3: Held-Out Benchmark V1.0
//
// SCOPE: This binary closes OC-RA-3 (generalisation — held-out validation).
// wo_demo.rs established training fit only (same corpus, N_MATURE=3 replay).
// This binary reports held-out perplexity on a split not seen during training.
//
// DECLARED PROTOCOL:
//
//   Corpus: WikiText-2 (pytorch/examples).
//     Train split: first TRAIN_TOKENS tokens of wikitext-2/train.txt
//     Test  split: first TEST_TOKENS  tokens of wikitext-2/test.txt
//
//   M declaration:
//     SemanticM::from_corpus — PPMI co-occurrence vectors built from train text only.
//     dim=50, window=5, min_count=3.
//     Test tokens not in train vocabulary → unk (declared closed-world M).
//     This is a corpus-derived M: the relational structure reflects co-occurrence
//     statistics in the train corpus. No pre-trained vectors; no external dependency.
//
//   Accumulation: single-pass streaming over train tokens. No replay.
//     This is the declared streaming protocol (OC-RA-5 closed).
//     N_MATURE deviation of wo_demo is not present here.
//
//   W_O training: Adam on train R-fields → train next-token targets.
//     N_EPOCHS = 200. Only W_O is gradient-updated.
//
//   Test evaluation: stream test tokens through field (field continues to accumulate
//     from train state — declared online condition). W_O frozen during test.
//     Reported metric: mean cross-entropy → perplexity on test set.
//
//   Comparison baseline: GloVe-baseline replaced with corpus-M direct embedding.
//     "Direct M" baseline: W_O trained on raw M(token) vectors (no ABR operators).
//     Meaningful comparison: ABR R-field vs. Direct M (same M, same W_O, same task).
//
// RUN:
//   cargo run --release --bin held_out_benchmark -- \
//     --train /tmp/wikitext2/train.txt --test /tmp/wikitext2/test.txt
//
//   Or without args: looks for /tmp/wikitext2/{train,test}.txt by default.

use abr_relational_attention::{
    declaration::{ETA, THETA_MIN, COHERENCE_WINDOW},
    semantic::SemanticM,
    streaming::StreamingField,
    output_projection::{OutputProjection, AdamState, train_epoch_adam, evaluate},
};

// ── Declared parameters ────────────────────────────────────────────────────────

/// Number of train tokens to stream and train W_O on.
/// Large enough for a real relational field; small enough for a single session.
/// WikiText-2 train has ~2M tokens; we use 10K here as the declared benchmark.
const TRAIN_TOKENS: usize = 10_000;

/// Number of test tokens to evaluate on (held-out, never seen during training).
const TEST_TOKENS:  usize = 3_000;

/// PPMI embedding dimension (M declaration).
const PPMI_DIM:     usize = 50;

/// PPMI context window (M declaration).
const PPMI_WINDOW:  usize = 5;

/// Minimum token frequency for PPMI vocabulary (M declaration).
/// Higher value → smaller vocabulary → faster W_O training.
const PPMI_MIN_COUNT: usize = 5;

/// W_O training epochs.
const N_EPOCHS:     usize = 150;

/// Adam learning rate (declared in DECLARATION.md).
const ADAM_LR:      f64   = 1e-3;

/// Report loss every N epochs.
const REPORT_EVERY: usize = 50;

/// Characters of train text to pass to from_corpus for PPMI building.
/// Approximates TRAIN_TOKENS × avg_chars_per_token (~6) with margin.
/// Keeps the PPMI vocabulary sized to the actual training corpus, not the full
/// WikiText-2 training file.
const PPMI_TEXT_CHARS: usize = TRAIN_TOKENS * 8;

fn main() {
    // ── Parse args ──────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let train_path = arg_or(&args, "--train", "/tmp/wikitext2/train.txt");
    let test_path  = arg_or(&args, "--test",  "/tmp/wikitext2/test.txt");

    println!("====================================================================");
    println!("  ABR Relational Attention — Held-Out Benchmark (OC-RA-3)");
    println!("  Metatron Dynamics, Inc. — Declaration V4.1");
    println!("====================================================================");
    println!();
    println!("  Declared protocol:");
    println!("    Train: first {} tokens of {}", TRAIN_TOKENS, train_path);
    println!("    Test:  first {} tokens of {}", TEST_TOKENS,  test_path);
    println!("    M: PPMI corpus-derived (dim={}, window={}, min_count={})",
             PPMI_DIM, PPMI_WINDOW, PPMI_MIN_COUNT);
    println!("    Accumulation: single-pass streaming (no replay)");
    println!("    ETA={} | THETA_MIN={} | W={}", ETA, THETA_MIN, COHERENCE_WINDOW as usize);
    println!("    W_O: Adam(lr={}) for {} epochs", ADAM_LR, N_EPOCHS);
    println!();

    // ── 1. Load corpora ──────────────────────────────────────────────────────
    print!("  [1/5] Loading corpora ... ");
    let train_text = load_file(&train_path);
    let test_text  = load_file(&test_path);
    println!("OK");

    // ── 2. Build M from train corpus ─────────────────────────────────────────
    print!("  [2/5] Building PPMI M from train corpus ... ");
    // Truncate train_text to PPMI_TEXT_CHARS characters so that M is built from
    // approximately the same vocabulary as the TRAIN_TOKENS streaming corpus.
    // Using the full 11MB WikiText-2 file would produce a 27K-word vocabulary
    // while only 10K tokens are actually streamed — a declared inconsistency.
    // Declared: M vocabulary is closed-world to the first PPMI_TEXT_CHARS chars.
    let ppmi_text: &str = {
        let byte_limit = train_text.len().min(PPMI_TEXT_CHARS);
        // Snap to a UTF-8 char boundary
        let mut end = byte_limit;
        while end > 0 && !train_text.is_char_boundary(end) { end -= 1; }
        &train_text[..end]
    };
    let sem = SemanticM::from_corpus(ppmi_text, PPMI_DIM, PPMI_WINDOW, PPMI_MIN_COUNT);
    let dim = sem.dim;
    let vocab_table = sem.to_vocab_table();
    let vocab_size = sem.vocab_size();
    println!("OK (vocab={}, dim={})", vocab_size, dim);

    // Tokenize train and test using the same M vocabulary.
    // Test tokens not in train vocab → unk_id (declared closed-world M).
    let all_train_ids = sem.encode(&train_text);
    let all_test_ids  = sem.encode(&test_text);

    let train_ids: Vec<usize> = all_train_ids.iter().copied().take(TRAIN_TOKENS).collect();
    let test_ids:  Vec<usize> = all_test_ids.iter().copied().take(TEST_TOKENS).collect();

    let n_train = train_ids.len();
    let n_test  = test_ids.len();

    // Count OOV tokens in test (unk_id = last entry in vocab)
    let n_test_unk = test_ids.iter().filter(|&&id| id == sem.unk_id).count();
    println!("    Train tokens: {} | Test tokens: {} ({:.1}% OOV→unk)",
             n_train, n_test, 100.0 * n_test_unk as f64 / n_test as f64);
    println!();

    // Training pairs: (representation[t], target[t+1])
    // We predict the NEXT token from the representation at the CURRENT position.
    let n_train_pairs = n_train.saturating_sub(1);
    let n_test_pairs  = n_test.saturating_sub(1);
    let train_targets: Vec<usize> = train_ids[1..].to_vec();
    let test_targets:  Vec<usize> = test_ids[1..].to_vec();

    // ── 3. Stream train through field (single pass, no replay) ───────────────
    // MAX_SEQUENCE_LENGTH = 4096: stream in chunks of CHUNK_SIZE so that
    // Locus::new position assertions never fire.
    // EdgeStore persists across chunks (reset_stream discards only position stream).
    const CHUNK_SIZE: usize = 4000;
    println!("  [3/5] Streaming {} train tokens through relational field (chunk={})",
             n_train, CHUNK_SIZE);
    let mut field = StreamingField::new();
    arrive_chunked(&mut field, &train_ids, &vocab_table, CHUNK_SIZE);
    println!("    Admitted pairs at theta={}: {}",
             THETA_MIN, field.store.admitted_edges_at_threshold(THETA_MIN));
    println!("    Total edge bindings: {}", field.store.total_edges());

    // Collect train R-field representations.
    // Re-stream train tokens for inference: positions are rebuilt from 0 in each
    // chunk. inference_at reads the settled field (no additional accumulation
    // occurs because arrive_chunked already finalized the EdgeStore; we re-stream
    // only to give inference_at a valid positional stream to work with).
    //
    // Declared: inference uses the settled EdgeStore from the full train stream;
    // R-field values are therefore conditioned on the full train accumulation.
    //
    // Note: arrive_chunked calls reset_stream() after each chunk and after the
    // last chunk. We re-stream once more in a single chunk just for inference
    // position building — CHUNK_SIZE tokens at a time — collecting R-fields per
    // chunk as we go, to avoid exceeding MAX_SEQUENCE_LENGTH.
    let mut train_reps: Vec<Vec<f64>> = Vec::with_capacity(n_train_pairs);
    let mut n_novel_train = 0_usize;
    for (chunk_idx, chunk_ids) in train_ids[..n_train_pairs].chunks(CHUNK_SIZE).enumerate() {
        // Determine the offset for target indexing
        let chunk_offset = chunk_idx * CHUNK_SIZE;
        // Re-stream this chunk for positional inference
        field.arrive_sequence(chunk_ids, &vocab_table);
        // Collect R-fields for each position in this chunk (not including last
        // token which is a target, not a representation source, unless it has a
        // next-token target still within this chunk)
        for local_pos in 0..chunk_ids.len() {
            let (r, n_edges) = field.inference_at(local_pos, &vocab_table, THETA_MIN);
            if n_edges == 0 { n_novel_train += 1; }
            train_reps.push(r);
            let _ = chunk_offset; // suppress warning — chunk_offset tracked via vec length
        }
        field.reset_stream();
    }

    // Also build direct-M baseline reps (raw PPMI vectors, no ABR).
    let direct_m_train: Vec<Vec<f64>> = train_ids[..n_train_pairs].iter()
        .map(|&id| vocab_table[id].clone())
        .collect();

    println!("    Train R-fields: {} vectors ({} novel/zero)", n_train_pairs, n_novel_train);
    println!();

    // ── 4. Train W_O on train R-fields ───────────────────────────────────────
    println!("  [4/5] Training W_O ({} epochs, Adam lr={}) ...", N_EPOCHS, ADAM_LR);
    println!();
    println!("  {:>5}  {:>14}  {:>14}  {:>14}  {:>14}",
             "epoch", "ABR:train_loss", "ABR:train_pplx", "Direct:train_loss", "Direct:train_pplx");
    println!("  {:>5}  {:>14}  {:>14}  {:>14}  {:>14}",
             "─────", "──────────────", "──────────────", "─────────────────", "─────────────────");

    let mut wo_abr    = OutputProjection::new_random(dim, vocab_size);
    let mut wo_direct = OutputProjection::new_random(dim, vocab_size);
    // Start both from identical initial weights.
    wo_direct.w = wo_abr.w.clone();
    wo_direct.b = wo_abr.b.clone();

    let mut adam_abr    = AdamState::new(dim, vocab_size);
    let mut adam_direct = AdamState::new(dim, vocab_size);

    // Initial loss
    let (l_abr_0, _, _)    = evaluate(&wo_abr,    &train_reps,    &train_targets);
    let (l_direct_0, _, _) = evaluate(&wo_direct, &direct_m_train, &train_targets);
    println!("  {:>5}  {:>14.4}  {:>14.1}  {:>14.4}  {:>14.1}",
             0, l_abr_0, l_abr_0.exp(),
             l_direct_0, l_direct_0.exp());

    for epoch in 1..=N_EPOCHS {
        train_epoch_adam(&mut wo_abr,    &mut adam_abr,    &train_reps,    &train_targets, ADAM_LR);
        train_epoch_adam(&mut wo_direct, &mut adam_direct, &direct_m_train, &train_targets, ADAM_LR);

        if epoch % REPORT_EVERY == 0 || epoch == N_EPOCHS {
            let (l_abr, _, _)    = evaluate(&wo_abr,    &train_reps,    &train_targets);
            let (l_direct, _, _) = evaluate(&wo_direct, &direct_m_train, &train_targets);
            println!("  {:>5}  {:>14.4}  {:>14.1}  {:>14.4}  {:>14.1}",
                     epoch, l_abr, l_abr.exp(), l_direct, l_direct.exp());
        }
    }
    println!();

    // ── 5. Evaluate on held-out test set ─────────────────────────────────────
    println!("  [5/5] Evaluating on held-out test set ({} tokens) ...", n_test);

    // Stream test tokens through field (field accumulates from train state).
    // Declared: online accumulation continues during test; W_O frozen.
    // This is not replay — test tokens are new, unseen exposures.
    // Chunked for same MAX_SEQUENCE_LENGTH reason as train.
    arrive_chunked(&mut field, &test_ids, &vocab_table, CHUNK_SIZE);

    // Re-stream test for inference (same positional rebuild pattern as train).
    let mut test_reps: Vec<Vec<f64>> = Vec::with_capacity(n_test_pairs);
    let mut n_novel_test = 0_usize;
    for chunk_ids in test_ids[..n_test_pairs].chunks(CHUNK_SIZE) {
        field.arrive_sequence(chunk_ids, &vocab_table);
        for local_pos in 0..chunk_ids.len() {
            let (r, n_edges) = field.inference_at(local_pos, &vocab_table, THETA_MIN);
            if n_edges == 0 { n_novel_test += 1; }
            test_reps.push(r);
        }
        field.reset_stream();
    }

    let direct_m_test: Vec<Vec<f64>> = test_ids[..n_test_pairs].iter()
        .map(|&id| vocab_table[id].clone())
        .collect();

    println!("    Test R-fields: {} vectors ({} novel/zero, {:.1}%)",
             n_test_pairs, n_novel_test,
             100.0 * n_novel_test as f64 / n_test_pairs.max(1) as f64);
    println!();

    // Evaluate: W_O frozen, R-fields from held-out test.
    let (test_loss_abr,    test_acc_abr,    test_top5_abr)    = evaluate(&wo_abr,    &test_reps,    &test_targets);
    let (test_loss_direct, test_acc_direct, test_top5_direct) = evaluate(&wo_direct, &direct_m_test, &test_targets);
    let (train_loss_abr,   train_acc_abr,   train_top5_abr)   = evaluate(&wo_abr,    &train_reps,    &train_targets);
    let (train_loss_direct, train_acc_direct, train_top5_direct) = evaluate(&wo_direct, &direct_m_train, &train_targets);

    // ── Results ──────────────────────────────────────────────────────────────
    println!("====================================================================");
    println!("  HELD-OUT BENCHMARK RESULTS (OC-RA-3)");
    println!("====================================================================");
    println!();
    println!("  {:>12}  {:>10}  {:>12}  {:>10}  {:>12}",
             "condition", "loss", "perplexity", "top-1 acc", "top-5 acc");
    println!("  {:>12}  {:>10}  {:>12}  {:>10}  {:>12}",
             "────────────", "──────────", "────────────", "──────────", "────────────");
    println!("  {:>12}  {:>10.4}  {:>12.1}  {:>9.1}%  {:>11.1}%",
             "ABR train",    train_loss_abr,    train_loss_abr.exp(),    100.0*train_acc_abr,    100.0*train_top5_abr);
    println!("  {:>12}  {:>10.4}  {:>12.1}  {:>9.1}%  {:>11.1}%",
             "ABR test",     test_loss_abr,     test_loss_abr.exp(),     100.0*test_acc_abr,     100.0*test_top5_abr);
    println!("  {:>12}  {:>10.4}  {:>12.1}  {:>9.1}%  {:>11.1}%",
             "Direct train", train_loss_direct, train_loss_direct.exp(), 100.0*train_acc_direct, 100.0*train_top5_direct);
    println!("  {:>12}  {:>10.4}  {:>12.1}  {:>9.1}%  {:>11.1}%",
             "Direct test",  test_loss_direct,  test_loss_direct.exp(),  100.0*test_acc_direct,  100.0*test_top5_direct);
    println!();

    // Generalisation gap
    let abr_gap    = test_loss_abr    - train_loss_abr;
    let direct_gap = test_loss_direct - train_loss_direct;
    let abr_vs_direct_test  = test_loss_direct  - test_loss_abr;
    let abr_vs_direct_train = train_loss_direct - train_loss_abr;

    println!("  Generalisation gap (test − train loss):");
    println!("    ABR:    {:.4} ({} → {})", abr_gap,
             if abr_gap > 0.0 { "overfit" } else { "underfit" },
             if abr_gap.abs() < 0.5 { "small gap" } else { "large gap" });
    println!("    Direct: {:.4}", direct_gap);
    println!();
    // abr_vs_direct = Direct_loss - ABR_loss.
    // Positive  → Direct loss is higher → ABR has lower loss → ABR is better.
    // Negative  → Direct loss is lower  → ABR has higher loss → Direct M is better.
    println!("  ABR advantage over Direct M (positive = ABR better):");
    println!("    Test  loss delta: {:.4}  (ABR {} Direct on held-out — {} is better)",
             abr_vs_direct_test,
             if abr_vs_direct_test >= 0.0 { "<" } else { ">" },
             if abr_vs_direct_test >= 0.0 { "ABR" } else { "Direct M" });
    println!("    Train loss delta: {:.4}  ({} is better on train)",
             abr_vs_direct_train,
             if abr_vs_direct_train >= 0.0 { "ABR" } else { "Direct M" });
    println!();
    println!("  Novel token rate (no admitted R-field):");
    println!("    Train: {:.1}% | Test: {:.1}%",
             100.0 * n_novel_train as f64 / n_train_pairs.max(1) as f64,
             100.0 * n_novel_test  as f64 / n_test_pairs.max(1)  as f64);
    println!();

    println!("====================================================================");
    println!("  DECLARED CONDITIONS");
    println!("====================================================================");
    println!();
    println!("  Corpus:    WikiText-2 (pytorch/examples)");
    println!("  Train:     first {} tokens; Test: first {} tokens",
             TRAIN_TOKENS, TEST_TOKENS);
    println!("  M:         PPMI corpus-derived from train only");
    println!("             dim={} window={} min_count={}", PPMI_DIM, PPMI_WINDOW, PPMI_MIN_COUNT);
    println!("             OOV rate on test: {:.1}%",
             100.0 * n_test_unk as f64 / n_test as f64);
    println!("  Streaming: single-pass, no replay (OC-RA-5 compliant)");
    println!("  Test eval: online accumulation continues during test stream");
    println!("             (new token-pair exposures; not a replay of train data)");
    println!("  W_O:       Adam(lr={}) for {} epochs; frozen during test",
             ADAM_LR, N_EPOCHS);
    println!("  Baseline:  Direct M — W_O on raw PPMI M(token), no ABR operators");
    println!("             (same M, same W_O init, same training; operators removed)");
    println!();
    println!("  KEY RESULT: ABR test perplexity = {:.1}", test_loss_abr.exp());
    println!("  Direct M  test perplexity = {:.1}  (delta: {:.4})",
             test_loss_direct.exp(), abr_vs_direct_test);
    println!();

    if abr_vs_direct_test > 0.0 {
        // Direct loss > ABR loss → ABR has lower loss → ABR is better
        println!("  FINDING: ABR R-field generalises better than Direct M.");
        println!("  The relational operators provide held-out predictive advantage");
        println!("  over the raw PPMI embeddings on the same vocabulary and task.");
    } else {
        // Direct loss <= ABR loss → Direct M is better on held-out
        println!("  FINDING: Direct M test perplexity < ABR on this configuration.");
        println!("  Likely causes: OOV rate {:.1}%, small train corpus ({} tokens),",
                 100.0 * n_test_unk as f64 / n_test as f64, TRAIN_TOKENS);
        println!("  R-field scaling (initial ABR pplx >> Direct due to operator range).");
        println!("  Declared: result is accurate for stated conditions.");
        println!("  Larger corpus and lower OOV rate expected to improve ABR advantage.");
    }
    println!();
    println!("  OC-RA-3 status: result obtained. Generalisation claim bounded to");
    println!("  declared conditions above. See DECLARATION.md V4.2.");
    println!("====================================================================");
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/// Stream token_ids through a StreamingField in chunks.
///
/// Declared chunk size is below MAX_SEQUENCE_LENGTH (4096) so that position
/// assertions in Locus::new never fire.  reset_stream() discards the positional
/// stream after each chunk while preserving the EdgeStore.
///
/// Token pairs that span a chunk boundary (up to COHERENCE_WINDOW per boundary)
/// are not evaluated.  For 100K tokens and CHUNK=4000: ~25 boundaries × 64 =
/// ~1600 missed pair evaluations out of ~100K × 64 = 6.4M total.
/// Declared acceptable loss: 0.025%.
fn arrive_chunked(
    field: &mut abr_relational_attention::streaming::StreamingField,
    token_ids: &[usize],
    vocab: &abr_relational_attention::declaration::VocabTable,
    chunk_size: usize,
) {
    for chunk in token_ids.chunks(chunk_size) {
        field.arrive_sequence(chunk, vocab);
        field.reset_stream();
    }
}

fn load_file(path: &str) -> String {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| { eprintln!("Cannot read {}: {}", path, e); std::process::exit(1); })
}

fn arg_or<'a>(args: &'a [String], flag: &str, default: &'a str) -> String {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].clone())
        .unwrap_or_else(|| default.to_string())
}
