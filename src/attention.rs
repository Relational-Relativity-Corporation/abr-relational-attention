// attention.rs -- Metatron Dynamics, Inc.
// ABR Relational Attention -- V4.0
//
// Grounding: operators.rs V7 kernel source, DECLARATION.md V4.0 (2026-08-21)
//
// V4.0: uses canonical B, rho, R over DeclaredGraph.
//   1. EdgeStore (rho_acc >= theta_min) establishes the declared graph topology.
//   2. B acts on declared graph: B[e] = A[e] + sum_succ A[f]
//   3. rho computed per node: rho[i] = rho_base * chi[i] / (1 + chi[i])
//   4. R acts on B field: R[e] = B[e] + rho[src(e)] * (sum_succ B - sum_pred B)
//
// B and R do not perform admissibility selection.
// The declared graph is established before they act.

use crate::declaration::{Locus, RelationalPair, VocabTable};
use crate::operators::{operator_a, operator_b, compute_rho, operator_r,
                       DeclaredGraph, Embedding, RHO_BASE};
use crate::accumulation::EdgeStore;

pub struct AttentionConfig {
    pub rho_base: f64,
}

impl Default for AttentionConfig {
    fn default() -> Self { Self { rho_base: RHO_BASE } }
}

pub struct AttentionOutput {
    pub outputs: Vec<Embedding>,
    pub admitted_edge_count: usize,
    pub mean_rho_at_nodes: f64,
}

/// Build the declared graph for a sequence from the current EdgeStore.
/// admitted_pairs: all (src_token_id, tgt_token_id) with rho_acc >= theta_min.
/// This establishes the topology. B and R then act canonically on it.
fn build_declared_graph(
    sequence: &[Locus],
    vocab: &VocabTable,
    store: &EdgeStore,
) -> DeclaredGraph {
    let admitted_pairs = store.all_admitted_pairs();
    DeclaredGraph::build(sequence, vocab, &admitted_pairs)
}

/// Relational attention pass.
///
/// 1. Build declared graph from EdgeStore admitted pairs.
/// 2. Compute A field (directed differences).
/// 3. Compute B field (canonical successor accumulation).
/// 4. Compute rho per node (canonical chi-based).
/// 5. Compute R field (canonical antisymmetric resolution).
/// 6. Output at each sequence position = sum of R[e] for edges targeting that position.
///
/// Inference cost: O(n * k) admitted neighbourhood.
pub fn relational_attention_pass(
    sequence: &[Locus],
    vocab: &VocabTable,
    store: &EdgeStore,
    config: &AttentionConfig,
) -> AttentionOutput {
    let graph = build_declared_graph(sequence, vocab, store);
    let admitted_edge_count = graph.n_edges();

    if admitted_edge_count == 0 {
        // Cold start: no admitted edges, zero outputs
        let dim = if vocab.is_empty() { 1 } else { vocab[0].len() };
        return AttentionOutput {
            outputs: vec![vec![0.0; dim]; sequence.len()],
            admitted_edge_count: 0,
            mean_rho_at_nodes: 0.0,
        };
    }

    let bg = operator_b(&graph);
    let rho_nodes = compute_rho(&graph, sequence.len(), config.rho_base);
    let r_field = operator_r(&bg, &graph, &rho_nodes);

    let dim = r_field[0].len();
    let mut outputs = vec![vec![0.0_f64; dim]; sequence.len()];

    // Aggregate R outputs at each target position
    for (e, &(_src_pos, tgt_pos)) in graph.edges.iter().enumerate() {
        for (o, r) in outputs[tgt_pos].iter_mut().zip(r_field[e].iter()) {
            *o += r;
        }
    }

    let mean_rho = if sequence.is_empty() {
        0.0
    } else {
        rho_nodes.iter().sum::<f64>() / sequence.len() as f64
    };

    AttentionOutput { outputs, admitted_edge_count, mean_rho_at_nodes: mean_rho }
}

/// Training accumulation pass.
///
/// For each causal pair (i,j): compute A(i,j), update EdgeStore with rho_n.
/// rho_n = ||A(i,j)||_max / (||A(i,j)||_max + eps) -- pre-graph rho for accumulation.
/// (Graph is not yet declared during accumulation -- rho_n bootstraps admission.)
///
/// Cost: O(n^2) Phase 0.
pub fn accumulate_from_sequence(
    sequence: &[Locus],
    vocab: &VocabTable,
    store: &mut EdgeStore,
    _config: &AttentionConfig,
) {
    for j in 0..sequence.len() {
        for i in 0..=j {
            let src = sequence[i];
            let tgt = sequence[j];
            if let Some(pair) = RelationalPair::declare(src, tgt) {
                // A = directed difference
                let a = operator_a(&src, &tgt, vocab);
                // rho_n = max |A[c]| / (max |A[c]| + eps) -- chi form, pre-graph
                let chi: f64 = a.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                let rho_n = RHO_BASE * chi / (1.0 + chi);
                store.update(&pair, rho_n);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accumulation::EdgeStore;
    use crate::declaration::{Locus, build_vocab_table};

    fn make_sequence(n: usize) -> Vec<Locus> {
        (0..n).map(|i| Locus::new(i, (i * 3 + 1) % 4)).collect()
    }

    fn make_vocab() -> VocabTable { build_vocab_table(4, 4) }

    #[test]
    fn test_cold_start_produces_zero_outputs() {
        let seq = make_sequence(4);
        let vocab = make_vocab();
        let store = EdgeStore::new();
        let out = relational_attention_pass(&seq, &vocab, &store, &Default::default());
        assert_eq!(out.outputs.len(), 4);
        assert_eq!(out.admitted_edge_count, 0);
        for o in &out.outputs {
            for v in o { assert!(v.abs() < 1e-10); }
        }
    }

    #[test]
    fn test_accumulation_creates_edges() {
        let seq = make_sequence(4);
        let vocab = make_vocab();
        let mut store = EdgeStore::new();
        accumulate_from_sequence(&seq, &vocab, &mut store, &Default::default());
        assert!(store.total_edges() > 0);
    }

    #[test]
    fn test_rho_n_zero_for_same_token() {
        // A(x,x) = 0 => chi = 0 => rho_n = 0 => same-token pairs never admitted
        let vocab = make_vocab();
        let src = Locus::new(0, 2);
        let tgt = Locus::new(1, 2);
        let a = operator_a(&src, &tgt, &vocab);
        let chi: f64 = a.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(chi < 1e-10, "same token must give chi=0");
    }

    #[test]
    fn test_different_tokens_produce_nonzero_rho_n() {
        let vocab = make_vocab();
        let src = Locus::new(0, 0);
        let tgt = Locus::new(1, 1);
        let a = operator_a(&src, &tgt, &vocab);
        let chi: f64 = a.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(chi > 1e-8, "different tokens must give nonzero chi");
    }

    #[test]
    fn test_repeated_accumulation_increases_mean_rho() {
        let seq = make_sequence(4);
        let vocab = make_vocab();
        let mut store = EdgeStore::new();
        accumulate_from_sequence(&seq, &vocab, &mut store, &Default::default());
        let mean_1 = store.mean_rho_acc();
        accumulate_from_sequence(&seq, &vocab, &mut store, &Default::default());
        let mean_2 = store.mean_rho_acc();
        assert!(mean_2 >= mean_1);
    }

    #[test]
    fn test_attention_after_training_has_admitted_edges() {
        let seq = make_sequence(4);
        let vocab = make_vocab();
        let mut store = EdgeStore::new();
        for _ in 0..50 {
            accumulate_from_sequence(&seq, &vocab, &mut store, &Default::default());
        }
        let out = relational_attention_pass(&seq, &vocab, &store, &Default::default());
        assert!(out.admitted_edge_count > 0);
    }

    #[test]
    fn test_outputs_finite_after_training() {
        let seq = make_sequence(5);
        let vocab = make_vocab();
        let mut store = EdgeStore::new();
        for _ in 0..50 {
            accumulate_from_sequence(&seq, &vocab, &mut store, &Default::default());
        }
        let out = relational_attention_pass(&seq, &vocab, &store, &Default::default());
        for o in &out.outputs {
            for v in o { assert!(v.is_finite(), "output must be finite"); }
        }
    }

    #[test]
    fn test_causal_only_no_future_sources() {
        let seq = make_sequence(5);
        let vocab = make_vocab();
        let mut store = EdgeStore::new();
        for _ in 0..50 {
            accumulate_from_sequence(&seq, &vocab, &mut store, &Default::default());
        }
        let graph = build_declared_graph(&seq, &vocab, &store);
        for &(src_pos, tgt_pos) in &graph.edges {
            assert!(src_pos <= tgt_pos,
                "causal violation: src pos {} > tgt pos {}", src_pos, tgt_pos);
        }
    }

    #[test]
    fn test_token_binding_persists_across_sequences() {
        let seq_a: Vec<Locus> = vec![Locus::new(0,1), Locus::new(1,2)];
        let seq_b: Vec<Locus> = vec![Locus::new(0,3), Locus::new(1,1), Locus::new(2,2)];
        let vocab = build_vocab_table(4, 4);
        let mut store = EdgeStore::new();
        accumulate_from_sequence(&seq_a, &vocab, &mut store, &Default::default());
        let rho_a = store.get(&RelationalPair::declare(
            Locus::new(0,1), Locus::new(1,2)).unwrap());
        accumulate_from_sequence(&seq_b, &vocab, &mut store, &Default::default());
        let rho_b = store.get(&RelationalPair::declare(
            Locus::new(0,1), Locus::new(1,2)).unwrap());
        assert!(rho_b > rho_a, "token binding must accumulate: {rho_a} -> {rho_b}");
    }

    #[test]
    fn test_r_passthrough_preserved() {
        // When succ and pred are empty, R[e] = B[e]
        let vocab = make_vocab();
        let seq = vec![Locus::new(0,0), Locus::new(1,1)];
        let mut store = EdgeStore::new();
        for _ in 0..50 {
            accumulate_from_sequence(&seq, &vocab, &mut store, &Default::default());
        }
        let graph = build_declared_graph(&seq, &vocab, &store);
        if graph.n_edges() > 0 {
            let bg = operator_b(&graph);
            let rho_nodes = compute_rho(&graph, seq.len(), RHO_BASE);
            let r = operator_r(&bg, &graph, &rho_nodes);
            // For the only edge (no succ, no pred), R = B
            assert_eq!(r.len(), bg.len());
            for (ri, bi) in r[0].iter().zip(bg[0].iter()) {
                assert!((ri - bi).abs() < 1e-10);
            }
        }
    }
}
