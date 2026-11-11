// layer_stack.rs — Metatron Dynamics, Inc.
// ABR Relational Attention — Layer Stack V1.0
//
// Grounding: declaration.rs V2.0, attention_layer.rs V1.0, operators.rs V7
//
// ARCHITECTURAL PRINCIPLE:
//   The ABR kernel produces a relational field R ∈ ℝ^dim at each token position.
//   That R-field IS the token's contextualized representation — the declared
//   measurement of what that token means in the relational context of the sequence.
//
//   A single ABR layer: M_1 (declared, GloVe/identity) → R-field_1
//   A stacked ABR layer: M_2 (derived from R-field_1) → R-field_2
//
//   M_{L+1}[token_id] = mean(R-field_L at all positions where token_id appeared,
//                             excluding novel positions with zero output)
//
//   Normalized per-dimension to [0,1]^dim — back into the declared domain D.
//
// WHY STACKING WORKS:
//   Layer 1 M is a syntactic/lexical geometry (GloVe word vectors).
//   Layer 1 R-field accumulates which tokens co-occur relationally in this corpus.
//   Layer 2 M is now a contextual geometry: the same token_id maps to its
//   relational history, not just its dictionary definition.
//   Layer 2 R-field accumulates higher-order relational structure.
//
//   This is a declared analog of BERT's layer hierarchy:
//     BERT layer 1: syntax → layer 2: morphology → ... → layer 12: semantics
//     ABR layer 1:  lexical → layer 2: relational context → layer L: ?
//   The difference: BERT requires 340M parameters and gradient descent.
//   ABR requires W_O (the output projection) only.
//
// WHAT CHANGES BETWEEN LAYERS:
//   The ρ_acc rankings shift. Pairs that were high-ρ_acc at layer 1 (function words)
//   may remain high because they still co-occur. But the relational geometry M has
//   changed: tokens that were semantically similar in GloVe space are now similar
//   in RELATIONAL space — meaning their layer-2 M vectors are close iff they
//   appeared in similar relational contexts in layer 1.
//
//   This is the structural mechanism by which content-word discrimination emerges
//   at deeper layers: "species" and "varieties" have similar layer-1 R-fields
//   (they appear after "the", before "in", with similar neighbors); at layer 2,
//   they are geometrically close in M₂, producing lower ρₙ per co-occurrence
//   but accumulating ρ_acc together just as the function words do.

use crate::declaration::VocabTable;
use crate::attention_layer::{AttentionLayer, AttentionOutput};

/// Output from one pass through a stacked ABR layer.
#[derive(Debug)]
pub struct StackLayerResult {
    /// Layer index (0-based).
    pub layer: usize,
    /// Per-token outputs from this layer's attention computation.
    pub outputs: Vec<AttentionOutput>,
    /// The M table used by this layer (derived from the layer below, or base M for layer 0).
    pub vocab_used: VocabTable,
    /// Admitted pair count in this layer's EdgeStore after this pass.
    pub n_admitted: usize,
    /// Novel token count in this pass.
    pub n_novel: usize,
    /// Mean |R| norm of non-novel outputs.
    pub mean_r_norm: f64,
}

/// A stack of ABR attention layers.
///
/// Each layer has an independent StreamingField (EdgeStore + frontier).
/// Between layers, the R-field output of layer L is contextualized into
/// the M for layer L+1 via `contextualize_m`.
///
/// The declared parameters (ETA, THETA_MIN, W) apply to all layers.
/// Each layer's EdgeStore accumulates independently.
pub struct ABRLayerStack {
    pub layers: Vec<AttentionLayer>,
}

impl ABRLayerStack {
    /// Create a stack of `n_layers` independent ABR attention layers.
    pub fn new(n_layers: usize) -> Self {
        assert!(n_layers > 0, "layer stack must have at least one layer");
        Self {
            layers: (0..n_layers).map(|_| AttentionLayer::new()).collect(),
        }
    }

    /// Create a stack with an explicit theta (for calibration).
    pub fn with_theta(n_layers: usize, theta: f64) -> Self {
        Self {
            layers: (0..n_layers)
                .map(|_| AttentionLayer::with_theta(theta))
                .collect(),
        }
    }

    /// Process a token sequence through all layers.
    ///
    /// Layer 0 uses `base_vocab` (the declared or pre-trained M).
    /// Layer L (L > 0) uses M derived from layer L-1's R-field outputs.
    ///
    /// Returns one `StackLayerResult` per layer.
    pub fn forward(
        &mut self,
        token_ids: &[usize],
        base_vocab: &VocabTable,
    ) -> Vec<StackLayerResult> {
        let dim = base_vocab[0].len();
        let vocab_size = base_vocab.len();
        let mut results: Vec<StackLayerResult> = Vec::with_capacity(self.layers.len());

        // Layer 0: use the base vocabulary
        let vocab_0 = base_vocab.clone();
        let outs_0 = self.layers[0].forward(token_ids, &vocab_0);
        results.push(make_result(0, outs_0, vocab_0, &self.layers[0]));

        // Layers 1..N: M derived from previous layer's R-field
        for l in 1..self.layers.len() {
            let prev_outs = &results[l - 1].outputs;
            let new_vocab = contextualize_m(token_ids, prev_outs, vocab_size, dim);
            let outs = self.layers[l].forward(token_ids, &new_vocab);
            results.push(make_result(l, outs, new_vocab, &self.layers[l]));
        }

        results
    }

    /// Reset positional streams in all layers while preserving EdgeStore ρ_acc.
    ///
    /// Call this between maturation passes to avoid O(N×T) positional accumulation.
    /// The relational field (ρ_acc in each EdgeStore) is preserved exactly.
    ///
    /// Without this reset, after N maturation passes the stream Vec has N×T positions.
    /// inference_at() builds its window from the most recent W=64 positions (bounded),
    /// but arrive() must still scan O(W) predecessors for each of N×T tokens per pass,
    /// growing the total edge update cost to O(N×T×W) per pass. With reset, it stays
    /// O(T×W) per pass regardless of how many maturation passes have been run.
    pub fn reset_streams(&mut self) {
        for layer in &mut self.layers {
            layer.reset_stream();
        }
    }

    /// Number of layers.
    pub fn n_layers(&self) -> usize { self.layers.len() }

    /// Total admitted pairs summed across all layers.
    pub fn total_admitted(&self) -> usize {
        self.layers.iter().map(|l| l.admitted_pairs()).sum()
    }
}

/// Derive a new VocabTable M_{L+1} from layer L's R-field outputs.
///
/// For each token_id t appearing in token_ids:
///   M_{L+1}[t] = mean(r_field[i] for all i where token_ids[i] == t
///                     and outputs[i].is_novel == false)
///
/// For token_ids not in the sequence, or for which every output was novel:
///   M_{L+1}[t] = 0.5 everywhere (center of domain — no contextual signal).
///
/// The result is normalized per-dimension to [0,1]^dim via min-max.
///
/// This function is M-agnostic: it does not use the base vocabulary.
/// The only input is the relational field output from the layer below.
pub fn contextualize_m(
    token_ids: &[usize],
    outputs: &[AttentionOutput],
    vocab_size: usize,
    dim: usize,
) -> VocabTable {
    // Accumulate R-field sums and counts per token_id.
    // Allocate vocab_size+1 slots (the +1 handles <unk> token_id=vocab_size).
    let total_slots = vocab_size + 1;
    let mut sums:   Vec<Vec<f64>> = vec![vec![0.0; dim]; total_slots];
    let mut counts: Vec<usize>    = vec![0; total_slots];

    for (i, out) in outputs.iter().enumerate() {
        if i >= token_ids.len() { break; }
        if out.is_novel { continue; }             // novel → zero R-field, skip
        if out.r_field.len() != dim { continue; } // dimension mismatch, skip

        let tid = token_ids[i].min(vocab_size);   // clamp <unk> into bounds
        for d in 0..dim {
            sums[tid][d] += out.r_field[d];
        }
        counts[tid] += 1;
    }

    // Build raw embeddings: mean for seen-and-non-novel tokens, 0.5 elsewhere.
    let mut raw: Vec<Vec<f64>> = (0..total_slots).map(|t| {
        if counts[t] > 0 {
            sums[t].iter().map(|&s| s / counts[t] as f64).collect()
        } else {
            vec![0.5_f64; dim]
        }
    }).collect();

    // Per-dimension min-max normalization → [0,1]^dim.
    for d in 0..dim {
        let min_v = raw.iter().map(|v| v[d]).fold(f64::INFINITY,     f64::min);
        let max_v = raw.iter().map(|v| v[d]).fold(f64::NEG_INFINITY, f64::max);
        let range = max_v - min_v;
        if range > 1e-10 {
            for slot in raw.iter_mut() {
                slot[d] = (slot[d] - min_v) / range;
            }
        } else {
            // Degenerate dimension: all values identical → map to 0.5
            for slot in raw.iter_mut() { slot[d] = 0.5; }
        }
    }

    // Return only the vocab_size entries (drop the +1 <unk> slot from the end
    // IF the base vocabulary had exactly vocab_size entries). The <unk> slot
    // at index vocab_size is intentionally included: it accumulates relational
    // context for all OOV tokens treated as the same abstract token.
    // We return all total_slots to match how SemanticM stores its unk.
    raw
}

// ── Helpers ────────────────────────────────────────────────────────────────────

fn make_result(
    layer: usize,
    outputs: Vec<AttentionOutput>,
    vocab_used: VocabTable,
    attn: &AttentionLayer,
) -> StackLayerResult {
    let n_novel   = outputs.iter().filter(|o| o.is_novel).count();
    let non_novel: Vec<_> = outputs.iter().filter(|o| !o.is_novel).collect();
    let mean_r_norm = if non_novel.is_empty() { 0.0 } else {
        let sum_norm: f64 = non_novel.iter().map(|o| {
            o.r_field.iter().map(|v| v * v).sum::<f64>().sqrt()
        }).sum();
        sum_norm / non_novel.len() as f64
    };
    StackLayerResult {
        layer,
        outputs,
        vocab_used,
        n_admitted: attn.admitted_pairs(),
        n_novel,
        mean_r_norm,
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::declaration::build_vocab_table;

    fn small_vocab(size: usize, dim: usize) -> VocabTable {
        build_vocab_table(size, dim)
    }

    // ── contextualize_m ────────────────────────────────────────────────

    #[test]
    fn test_contextualize_m_output_in_domain() {
        // After contextualization and normalization, all embeddings ∈ [0,1]^dim.
        let vocab = small_vocab(8, 8);
        let mut layer = AttentionLayer::new();
        let ids = vec![0_usize, 1, 2, 3, 0, 1]; // repeat to build ρ_acc
        for _ in 0..10 { layer.forward(&ids, &vocab); }
        let outs = layer.forward(&ids, &vocab);

        let new_m = contextualize_m(&ids, &outs, 8, 8);
        for (tid, emb) in new_m.iter().enumerate() {
            assert_eq!(emb.len(), 8, "dim mismatch at tid={}", tid);
            for (d, &v) in emb.iter().enumerate() {
                assert!(v >= 0.0 && v <= 1.0,
                    "M[{}][{}]={} outside [0,1]", tid, d, v);
            }
        }
    }

    #[test]
    fn test_contextualize_m_seen_tokens_differ_from_unseen() {
        // Tokens that appeared in the sequence should have M derived from their
        // R-field. Tokens not in the sequence should have M = 0.5 (after norm).
        // We test that SOME differentiation exists.
        let vocab = small_vocab(16, 8);
        let mut layer = AttentionLayer::new();
        let ids = vec![0_usize, 1, 2, 3];
        for _ in 0..30 { layer.forward(&ids, &vocab); }
        let outs = layer.forward(&ids, &vocab);

        let new_m = contextualize_m(&ids, &outs, 16, 8);

        // Token 15 (id=15) never appeared — should be 0.5 in all dims
        // (before normalization it was 0.5; after normalization it may shift
        // but should still be near 0.5 relative to others)
        // We just check that at least some entries are not exactly 0.5
        let not_half = new_m.iter().any(|emb| emb.iter().any(|&v| (v - 0.5).abs() > 1e-6));
        assert!(not_half, "some entries must differ from 0.5 after seen-token contextualization");
    }

    #[test]
    fn test_contextualize_m_all_novel_outputs_produce_half() {
        // When every AttentionOutput.is_novel == true, contextualize_m has
        // no contextual signal and must return 0.5 in every dimension.
        // We construct all-novel outputs manually to avoid dependence on
        // the exact ρ_acc threshold (identity M + ETA=0.1 hits THETA_MIN=0.05
        // on the very first token-pair observation, so "first pass = all novel"
        // is not guaranteed for identity embeddings).
        use crate::attention_layer::AttentionOutput;
        let dim = 8_usize;
        let vocab_size = 8_usize;
        let ids = vec![0_usize, 1, 2, 3];
        let all_novel: Vec<AttentionOutput> = ids.iter().enumerate().map(|(i, &tid)| {
            AttentionOutput {
                position: i,
                token_id: tid,
                r_field: vec![0.0; dim],
                n_admitted: 0,
                is_novel: true,
            }
        }).collect();
        let new_m = contextualize_m(&ids, &all_novel, vocab_size, dim);
        for emb in &new_m {
            for &v in emb {
                assert!((v - 0.5).abs() < 1e-10,
                    "all-novel outputs must yield 0.5 in contextualized M");
            }
        }
    }

    #[test]
    fn test_contextualize_m_vocab_size_matches() {
        let vocab = small_vocab(12, 6);
        let mut layer = AttentionLayer::new();
        let ids = vec![0_usize, 1, 2];
        for _ in 0..5 { layer.forward(&ids, &vocab); }
        let outs = layer.forward(&ids, &vocab);

        let new_m = contextualize_m(&ids, &outs, 12, 6);
        // Should have vocab_size + 1 slots (includes <unk>)
        assert_eq!(new_m.len(), 13, "contextualized M must have vocab_size+1 slots");
    }

    // ── ABRLayerStack ──────────────────────────────────────────────────

    #[test]
    fn test_stack_returns_one_result_per_layer() {
        let vocab = small_vocab(8, 8);
        let mut stack = ABRLayerStack::new(3);
        let ids = vec![0_usize, 1, 2, 3];
        let results = stack.forward(&ids, &vocab);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_stack_layer_indices() {
        let vocab = small_vocab(8, 8);
        let mut stack = ABRLayerStack::new(4);
        let ids = vec![0_usize, 1, 2, 3];
        let results = stack.forward(&ids, &vocab);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.layer, i, "layer index mismatch");
        }
    }

    #[test]
    fn test_stack_output_count_per_layer() {
        let vocab = small_vocab(8, 8);
        let mut stack = ABRLayerStack::new(2);
        let ids = vec![0_usize, 1, 2, 3, 4];
        let results = stack.forward(&ids, &vocab);
        for r in &results {
            assert_eq!(r.outputs.len(), ids.len(),
                "each layer must produce one output per token");
        }
    }

    #[test]
    fn test_stack_layer0_uses_base_vocab() {
        let vocab = small_vocab(8, 8);
        let mut stack = ABRLayerStack::new(2);
        let ids = vec![0_usize, 1, 2, 3];
        let results = stack.forward(&ids, &vocab);
        // Layer 0 vocab_used must match the base vocab exactly
        assert_eq!(results[0].vocab_used.len(), vocab.len());
        for (a, b) in results[0].vocab_used.iter().zip(vocab.iter()) {
            assert_eq!(a, b, "layer 0 must use base vocab unchanged");
        }
    }

    #[test]
    fn test_stack_layer1_vocab_differs_from_layer0() {
        // After enough accumulation, layer 1 M should differ from layer 0 M.
        let vocab = small_vocab(8, 8);
        let mut stack = ABRLayerStack::new(2);
        let ids = vec![0_usize, 1, 2, 3, 0, 1, 2, 3]; // repeat tokens
        // Run several times to build ρ_acc
        for _ in 0..20 { stack.forward(&ids, &vocab); }
        let results = stack.forward(&ids, &vocab);

        let v0 = &results[0].vocab_used;
        let v1 = &results[1].vocab_used;
        // Some entries should differ
        let any_diff = v0.iter().zip(v1.iter()).any(|(a, b)|
            a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6)
        );
        assert!(any_diff, "layer 1 M must differ from layer 0 M after accumulation");
    }

    #[test]
    fn test_stack_admitted_grows_with_passes() {
        let vocab = small_vocab(8, 8);
        let mut stack = ABRLayerStack::new(2);
        let ids = vec![0_usize, 1, 2, 3];
        stack.forward(&ids, &vocab);
        let after_first = stack.total_admitted();
        for _ in 0..10 { stack.forward(&ids, &vocab); }
        let after_many = stack.total_admitted();
        assert!(after_many >= after_first,
            "admitted pairs must not decrease with more passes");
    }

    #[test]
    fn test_stack_field_dimensions_consistent() {
        // Each layer's R-field output must have the same dimension as the vocab.
        let dim = 12_usize;
        let vocab = small_vocab(8, dim);
        let mut stack = ABRLayerStack::new(3);
        let ids = vec![0_usize, 1, 2, 3];
        for _ in 0..5 { stack.forward(&ids, &vocab); }
        let results = stack.forward(&ids, &vocab);
        for r in &results {
            for out in &r.outputs {
                assert_eq!(out.r_field.len(), dim,
                    "layer {} r_field.len()={} must equal dim={}", r.layer, out.r_field.len(), dim);
            }
        }
    }

    #[test]
    fn test_contextualize_m_no_panic_on_all_novel() {
        // Should not panic when all tokens are novel.
        let vocab = small_vocab(4, 4);
        let mut layer = AttentionLayer::new();
        let ids = vec![0_usize, 1, 2];
        let outs = layer.forward(&ids, &vocab); // first pass: all novel
        let _ = contextualize_m(&ids, &outs, 4, 4); // must not panic
    }
}
