// main.rs -- Metatron Dynamics, Inc.
// ABR Relational Attention -- Phase 0 V4.0

use abr_relational_attention::{
    declaration::{Locus, build_vocab_table},
    accumulation::EdgeStore,
    attention::{AttentionConfig, accumulate_from_sequence, relational_attention_pass},
};

fn main() {
    println!("=================================================================");
    println!("  ABR Relational Attention -- Phase 0 V4.0");
    println!("  Metatron Dynamics, Inc. -- V7 (2026-08-21)");
    println!("  DECLARATION.md V4.0 -- canonical B, rho, R from kernel source");
    println!("=================================================================");
    println!();

    let vocab_size = 4;
    let dim = 4;
    let vocab = build_vocab_table(vocab_size, dim);

    // Nontrivial token order
    let sequence: Vec<Locus> = vec![
        Locus::new(0, 2), Locus::new(1, 0), Locus::new(2, 3), Locus::new(3, 1),
        Locus::new(4, 2), Locus::new(5, 0), Locus::new(6, 3), Locus::new(7, 1),
    ];

    println!("Declaration:");
    println!("  Vocabulary   : {} tokens, {}-dim embeddings", vocab_size, dim);
    println!("  Sequence     : {} tokens", sequence.len());
    println!("  Token order  : {:?}",
        sequence.iter().map(|l| l.token_id).collect::<Vec<_>>());
    println!("  Direction    : Relational coherence progression (causal)");
    println!("  Operators    : canonical V7 (from kernel source)");
    println!("    A[e]   = M(src) - M(tgt)                  directed difference");
    println!("    B[e]   = A[e] + sum_succ A[f]             successor accumulation");
    println!("    rho[i] = rho_base * chi[i] / (1+chi[i])  per-node, chi=max|A|");
    println!("    R[e]   = B[e] + rho[src] * (sum_succ B - sum_pred B)");
    println!();

    let config = AttentionConfig::default();
    let mut store = EdgeStore::new();

    let n_passes = 50;
    println!("Accumulation -- {} passes (O(n^2) Phase 0):", n_passes);
    println!("  rho_n = rho_base * chi / (1+chi),  chi = max|A[c]|");
    println!("  Rule  : rho_acc <- rho_acc + eta * rho_n * (1 - rho_acc)");
    println!();

    for pass in 0..n_passes {
        accumulate_from_sequence(&sequence, &vocab, &mut store, &config);
        if pass == 0 || (pass + 1) % 10 == 0 {
            println!(
                "  Pass {:>3} | token pairs: {:>3} | admitted: {:>3} | mean rho_acc: {:.4}",
                pass + 1,
                store.total_edges(),
                store.admitted_edges(),
                store.mean_rho_acc(),
            );
        }
    }
    println!();

    let output = relational_attention_pass(&sequence, &vocab, &store, &config);
    println!("Relational attention pass:");
    println!("  Output vectors      : {}", output.outputs.len());
    println!("  Admitted edges      : {}", output.admitted_edge_count);
    println!("  Mean rho at nodes   : {:.4}", output.mean_rho_at_nodes);
    println!();

    println!("Open conditions:");
    println!("  OC-RA-1: theta_min calibration (Phase 1)");
    println!("  OC-RA-2: eta calibration (Phase 1)");
    println!("  OC-RA-3: generalisation -- held-out validation (Phase 1)");
    println!("  OC-RA-4: bidirectional direction -- not declared");
    println!("  OC-RA-5: window-restricted O(n*k) accumulation (Phase 1)");
    println!();
    println!("=================================================================");
    println!("  Phase 0 V4.0 complete. Ready for Verifier pass 4.");
    println!("=================================================================");
}
