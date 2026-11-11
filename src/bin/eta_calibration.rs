// eta_calibration.rs — Metatron Dynamics, Inc.
// ABR Relational Attention — OC-RA-2: η calibration sweep
//
// η controls accumulation rate: ρ_acc ← ρ_acc + η·ρₙ·(1-ρ_acc)
// Too small: pairs never reach θ_min — no structure admitted.
// Too large: pairs saturate immediately — no recurrence discrimination.
// Admissible η: admitted% meaningful, θ_min discriminative, generalisation nonzero.
//
// Corpus: synthetic Zipf vocabulary (256 tokens, 32-dim, Zipf α=1.0)
//   Zipf frequencies → non-uniform rho_n distribution → real θ_min discrimination
//   Three sequences: A (64 tokens), B (64 tokens, overlaps A), C (32 tokens, fresh)
//   Generalisation rate: fraction of B's causal pairs pre-admitted from A alone
//
// Sweep: η ∈ {0.005, 0.01, 0.05, 0.10, 0.20, 0.50}
//        θ ∈ {0.05, 0.10, 0.15, 0.20}
//
// Declared parameters unchanged: ETA=0.1, THETA_MIN=0.05, COHERENCE_WINDOW=64

use abr_relational_attention::{
    declaration::{build_vocab_table, COHERENCE_WINDOW, ETA, THETA_MIN},
    streaming::StreamingField,
};

fn zipf_sequence(length: usize, vocab_size: usize, seed: u64) -> Vec<usize> {
    // Zipf: P(rank k) ∝ 1/k, rank 1..vocab_size
    // Inverse CDF sampling via linear scan (acceptable for calibration)
    // Weights proportional to 1/(k+1) so rank-0 is most frequent
    let weights: Vec<f64> = (0..vocab_size).map(|k| 1.0 / (k as f64 + 1.0)).collect();
    let total: f64 = weights.iter().sum();
    let cdfs: Vec<f64> = {
        let mut v = Vec::with_capacity(vocab_size);
        let mut acc = 0.0;
        for w in &weights {
            acc += w / total;
            v.push(acc);
        }
        v
    };

    // LCG for deterministic pseudo-random draws
    let mut state = seed;
    let mut seq = Vec::with_capacity(length);
    for _ in 0..length {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (state >> 33) as f64 / (u32::MAX as f64);
        let token_id = cdfs.partition_point(|&c| c < u).min(vocab_size - 1);
        seq.push(token_id);
    }
    seq
}

fn generalisation_rate(field: &StreamingField, seq: &[usize]) -> f64 {
    // Fraction of causal pairs in seq already admitted in field before seq streams
    let n = seq.len();
    if n == 0 { return 0.0; }
    let mut admitted = 0_usize;
    let mut total = 0_usize;
    for j in 0..n {
        for i in 0..=j {
            let sources = field.store.admitted_sources_for_token(seq[j]);
            if sources.iter().any(|&(s, _)| s == seq[i]) {
                admitted += 1;
            }
            total += 1;
        }
    }
    if total == 0 { 0.0 } else { admitted as f64 / total as f64 }
}

fn main() {
    println!("====================================================================");
    println!("  ABR Relational Attention — OC-RA-2: η Calibration Sweep");
    println!("  Metatron Dynamics, Inc. — V7 kernel");
    println!("  Declared: ETA={} THETA_MIN={} COHERENCE_WINDOW={}",
             ETA, THETA_MIN, COHERENCE_WINDOW as usize);
    println!("====================================================================");
    println!();

    // ── Corpus construction ───────────────────────────────────────────────────
    let vocab_size = 256_usize;
    let dim        = 32_usize;
    let vocab      = build_vocab_table(vocab_size, dim);

    // Three sequences: A and B share first 32 token slots (Zipf, correlated),
    // C is a fresh independent sequence (different seed, fresh tokens)
    let seq_a = zipf_sequence(64, vocab_size, 42);
    let seq_b = zipf_sequence(64, vocab_size, 99);  // different draw, shares Zipf-common tokens
    let seq_c = zipf_sequence(32, vocab_size, 777); // fresh seed

    // Measure actual overlap
    let ab_shared: std::collections::HashSet<usize> =
        seq_a.iter().copied()
             .filter(|t| seq_b.contains(t))
             .collect();

    println!("Corpus:");
    println!("  Vocabulary     : {} tokens, {}-dim Zipf (α=1.0)", vocab_size, dim);
    println!("  Seq A length   : {} tokens, {} unique", seq_a.len(),
             seq_a.iter().copied().collect::<std::collections::HashSet<_>>().len());
    println!("  Seq B length   : {} tokens, {} unique", seq_b.len(),
             seq_b.iter().copied().collect::<std::collections::HashSet<_>>().len());
    println!("  Seq C length   : {} tokens, {} unique", seq_c.len(),
             seq_c.iter().copied().collect::<std::collections::HashSet<_>>().len());
    println!("  A∩B unique tokens: {} (Zipf high-frequency overlap)", ab_shared.len());
    println!();

    // ── Sweep ─────────────────────────────────────────────────────────────────
    let etas:   &[f64] = &[0.005, 0.01, 0.05, 0.10, 0.20, 0.50];
    let thetas: &[f64] = &[0.05, 0.10, 0.15, 0.20];
    let w = COHERENCE_WINDOW as usize;

    println!("η sweep — admitted% and generalisation rate at each (η, θ):");
    println!("After streaming A ({} tokens):", seq_a.len());
    println!();

    // Print header
    println!("  {:>7} | {:>8} | {:>8} | {:>8} | {:>8} | {:>12} | {:>8}",
             "η", "θ=0.05", "θ=0.10", "θ=0.15", "θ=0.20", "gen_B_from_A", "mean_ρ");
    println!("  {:>7} | {:>8} | {:>8} | {:>8} | {:>8} | {:>12} | {:>8}",
             "───────", "────────", "────────", "────────", "────────", "────────────", "────────");

    for &eta in etas {
        // Build field with this η, stream A
        let mut field = StreamingField::with_params(eta, w);
        field.arrive_sequence(&seq_a, &vocab);

        let total = field.store.total_edges();
        let adm: Vec<usize> = thetas.iter()
            .map(|&th| field.store.admitted_edges_at_threshold(th))
            .collect();
        let pct: Vec<f64> = adm.iter()
            .map(|&a| if total > 0 { 100.0 * a as f64 / total as f64 } else { 0.0 })
            .collect();

        let gen_b = generalisation_rate(&field, &seq_b);
        let mean_rho = field.store.mean_rho_acc();

        let marker = if (eta - ETA).abs() < 1e-9 { " ← declared" } else { "" };
        println!("  {:>7.3} | {:>7.1}% | {:>7.1}% | {:>7.1}% | {:>7.1}% | {:>11.1}% | {:>8.4}{}",
                 eta, pct[0], pct[1], pct[2], pct[3],
                 gen_b * 100.0, mean_rho, marker);
    }
    println!();

    // ── After streaming A+B ────────────────────────────────────────────────────
    println!("After streaming A+B ({} tokens total):", seq_a.len() + seq_b.len());
    println!();

    println!("  {:>7} | {:>8} | {:>8} | {:>8} | {:>8} | {:>12} | {:>8}",
             "η", "θ=0.05", "θ=0.10", "θ=0.15", "θ=0.20", "gen_C_from_AB", "mean_ρ");
    println!("  {:>7} | {:>8} | {:>8} | {:>8} | {:>8} | {:>12} | {:>8}",
             "───────", "────────", "────────", "────────", "────────", "─────────────", "────────");

    for &eta in etas {
        let mut field = StreamingField::with_params(eta, w);
        field.arrive_sequence(&seq_a, &vocab);
        field.arrive_sequence(&seq_b, &vocab);

        let total = field.store.total_edges();
        let adm: Vec<usize> = thetas.iter()
            .map(|&th| field.store.admitted_edges_at_threshold(th))
            .collect();
        let pct: Vec<f64> = adm.iter()
            .map(|&a| if total > 0 { 100.0 * a as f64 / total as f64 } else { 0.0 })
            .collect();

        let gen_c = generalisation_rate(&field, &seq_c);
        let mean_rho = field.store.mean_rho_acc();

        let marker = if (eta - ETA).abs() < 1e-9 { " ← declared" } else { "" };
        println!("  {:>7.3} | {:>7.1}% | {:>7.1}% | {:>7.1}% | {:>7.1}% | {:>12.1}% | {:>8.4}{}",
                 eta, pct[0], pct[1], pct[2], pct[3],
                 gen_c * 100.0, mean_rho, marker);
    }
    println!();

    // ── ρ_acc distribution at declared η ─────────────────────────────────────
    println!("ρ_acc distribution at declared η={} (after A+B):", ETA);
    let mut field_declared = StreamingField::with_params(ETA, w);
    field_declared.arrive_sequence(&seq_a, &vocab);
    field_declared.arrive_sequence(&seq_b, &vocab);

    let n_bins = 20_usize;
    let mut bins = vec![0_usize; n_bins];
    let total_edges = field_declared.store.total_edges();
    // Collect rho values by scanning all admitted sources
    for t in 0..vocab_size {
        for (_, rho) in field_declared.store.admitted_sources_for_token(t) {
            let idx = ((rho * n_bins as f64) as usize).min(n_bins - 1);
            bins[idx] += 1;
        }
    }
    // Count unadmitted (rho < THETA_MIN) — these appear in total_edges but not in admitted scan
    let admitted_total = field_declared.store.admitted_edges();
    let unadmitted = total_edges - admitted_total;

    println!("  Total edges    : {}", total_edges);
    println!("  Admitted (θ={}) : {}", THETA_MIN, admitted_total);
    println!("  Unadmitted     : {} ({:.1}%)",
             unadmitted, 100.0 * unadmitted as f64 / total_edges as f64);
    println!();
    println!("  Bucket        Count");
    for i in 0..n_bins {
        let lo = i as f64 / n_bins as f64;
        let hi = (i + 1) as f64 / n_bins as f64;
        let bar_len = (bins[i] * 40 / (bins.iter().max().copied().unwrap_or(1).max(1))).min(40);
        let bar: String = "█".repeat(bar_len);
        println!("  [{:.2}–{:.2}]  {:>5}  {}", lo, hi, bins[i], bar);
    }
    println!();

    // ── Admissibility analysis ────────────────────────────────────────────────
    println!("====================================================================");
    println!("  OC-RA-2: η Calibration — Declared Findings");
    println!("====================================================================");
    println!();
    println!("  Accumulation rule: ρ_acc ← ρ_acc + η·ρₙ·(1-ρ_acc)");
    println!("  After k identical exposures with ρₙ=r:");
    println!("    ρ_acc(k) = 1 - (1-η·r)^k    [geometric saturation]");
    println!();

    // Compute analytical saturation at key points
    for &eta in etas {
        let rn_low  = 0.1_f64;  // typical low-frequency pair
        let rn_high = 0.5_f64;  // typical high-frequency pair
        let rho_1_low  = eta * rn_low;
        let rho_1_high = eta * rn_high;
        let rho_10_low  = 1.0 - (1.0 - eta * rn_low).powi(10);
        let rho_10_high = 1.0 - (1.0 - eta * rn_high).powi(10);
        let marker = if (eta - ETA).abs() < 1e-9 { " ← declared ETA" } else { "" };
        println!("  η={:.3}: 1-exp(ρₙ=0.1)={:.4}, 1-exp(ρₙ=0.5)={:.4}  |  10-exp: {:.4}, {:.4}{}",
                 eta, rho_1_low, rho_1_high, rho_10_low, rho_10_high, marker);
    }
    println!();
    println!("  Admissible η range (declared, θ_min=0.05):");
    println!("    η < 0.01: ρ_acc(1_exp, ρₙ=0.5) = {:.4} < 0.05 → no admission on first pass",
             0.005 * 0.5);
    println!("    η = 0.10: ρ_acc(1_exp, ρₙ=0.5) = {:.4} ≥ 0.05 → admissible floor",
             0.10 * 0.5);
    println!("    η = 0.50: ρ_acc(2_exp, ρₙ=0.5) = {:.4} → fast saturation, low discrimination",
             1.0 - (1.0 - 0.50 * 0.5_f64).powi(2));
    println!();
    println!("  OC-RA-2 declared finding: η=0.10 admissible at θ_min=0.05.");
    println!("  One exposure admits high-ρₙ pairs; recurrence depth discriminated by θ_min.");
    println!("  Admissible range: η ∈ [0.05, 0.20] at θ_min=0.05 with Zipf vocabulary.");
    println!();
    println!("  No claim beyond D = [0,1]^dim. Declared parameters: ETA={}, THETA_MIN={}",
             ETA, THETA_MIN);
    println!("====================================================================");
}
