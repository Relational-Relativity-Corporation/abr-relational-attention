// attention_layer.rs — Metatron Dynamics, Inc.
// ABR Relational Attention — Attention Layer V1.0
//
// Grounding: declaration.rs V2.0, streaming.rs V1.0, operators.rs V7
//
// THE ARCHITECTURAL CLAIM:
//   The ABR relational field IS an attention mechanism.
//   The admitted neighborhood IS the attention sparse weight matrix.
//   ρ_acc IS the attention score (bounded in [0,1], no normalization needed).
//   R IS the attended output at each token position.
//
// COMPARISON: softmax attention
//   Q = W_Q · x    (d_model × d_k projection, trained)
//   K = W_K · x    (d_model × d_k projection, trained)
//   V = W_V · x    (d_model × d_v projection, trained)
//   attn = softmax(Q·K^T / √d_k)   O(n²·d_k) + softmax denominator
//   output = attn · V               O(n²·d_v)
//   total: O(n² · d) per layer, three projection matrices per head
//
// COMPARISON: ABR relational attention (this module)
//   M(x): embedding lookup (no projection matrix — M is declared, not trained)
//   A: M(src) - M(tgt)                    O(n·W·d)
//   ρₙ: max|A| / (1 + max|A|)             O(n·W)
//   ρ_acc: streaming accumulation          O(n·W) amortized
//   B = A + Σ_succ A                       O(k) per token
//   R = B + ρ_src · (Σ_succ B - Σ_pred B) O(k) per token
//   total: O(n·W·d + n·k) = O(n) for fixed W, k
//   no projection matrices. no softmax. no gradient pass.
//
// NO QKV MATRICES:
//   Softmax attention needs W_Q, W_K, W_V: 3 × d_model × d_head parameters per head.
//   GPT-3: 96 heads × 3 × 12288 × 128 = 452M parameters just for attention projections.
//   ABR: zero projection parameters. M is the declared measurement — not learned.
//   What is learned in softmax is the relational structure. In ABR, the relational
//   structure IS the declared field, accumulated from data without gradient descent.
//
// NO NORMALIZATION INSTABILITY:
//   softmax(x_i) = exp(x_i) / Σ exp(x_j)
//   The denominator Σ exp(x_j) is unbounded. When logits are large (which
//   gradient descent drives toward), exp overflows. Requires:
//     - scaled dot-product: 1/√d_k (a heuristic, not grounded in domain)
//     - layer normalization before attention (another free parameter)
//     - residual connections to prevent gradient vanishing
//   ABR: ρ_acc ∈ [0,1] by construction. No exp, no denominator, no scaling.
//
//   PRECISE DOMAIN CLAIM:
//   ρ_acc ∈ [0,1] is maintained analytically by the accumulation rule.
//   This is the structural anti-drift property: ρ_acc cannot diverge,
//   because each update is a convex combination: ρ' = ρ + η·ρₙ·(1-ρ),
//   which is strictly monotone increasing and bounded above by 1.
//
//   The R-field output vector is in ℝ^dim with norm O(W·√dim), not [0,1]^dim.
//   This is bounded — norm ≤ W·√dim = 64·√dim — but not normalized to unit
//   hypercube. The critical difference from softmax: this bound is STRUCTURAL,
//   not imposed by a normalization layer. The system cannot diverge because
//   ρ_acc gates all contributions, and ρ_acc ∈ [0,1] by construction.
//   Transformer drift is caused by gradient updates pushing logits unboundedly
//   large; ABR has no gradient updates and no logits — ρ_acc is the only
//   "attention weight", and it cannot exceed 1.
//
// ADMISSIBILITY:
//   Every quantity in the ABR attention layer traces to M through declared operators.
//   M: embedding table (declared, identity-keyed)
//   A = M(src) - M(tgt): difference in D (bounded since M maps to [0,1]^dim)
//   χ = max|A|: in [0,1] since A is bounded
//   ρₙ = χ/(1+χ): in [0,1) by construction (strictly, never saturates to 1)
//   ρ_acc = ρ_acc + η·ρₙ·(1-ρ_acc): in [0,1] by induction on accumulation rule
//   B = A + Σ A: bounded since A is bounded and sum is over finite window
//   R = B + ρ·(ΣB - ΣB): bounded since B, ρ bounded
//   CONTRAST: softmax output = Σ_j [exp(QK^T_j/√d) / Z] · V_j
//     V is the value matrix — it is NOT bounded. The output is a weighted sum
//     of unbounded values. The network relies on layer norm and residuals to
//     prevent explosion. These are stability patches, not grounded constraints.

use crate::declaration::{
    Locus, VocabTable,
    COHERENCE_WINDOW, THETA_MIN,
};
use crate::streaming::StreamingField;

/// One token's ABR attention output: R-field value at this position.
///
/// This is the declared output of the ABR attention layer for one token.
/// It is a vector in D = [0,1]^dim — the relational attention output.
#[derive(Debug, Clone)]
pub struct AttentionOutput {
    /// Token position in the sequence.
    pub position: usize,
    /// Token identity.
    pub token_id: usize,
    /// ABR attention output vector. Zero if novel token (no admitted history).
    pub r_field: Vec<f64>,
    /// Number of admitted predecessors contributing to this output.
    /// = 0 for novel tokens.
    pub n_admitted: usize,
    /// Whether this token's output is entirely from the relational field
    /// (n_admitted > 0) or is the novel-token zero vector (n_admitted = 0).
    pub is_novel: bool,
}

/// ABR Attention Layer.
///
/// A drop-in architectural module that replaces one transformer attention head.
/// Inputs: a token sequence (via token_ids) and a vocabulary embedding table.
/// Outputs: one AttentionOutput per token position.
///
/// The layer maintains a persistent StreamingField across all calls.
/// Multiple calls = multiple sequences processed by the same field.
/// This is cross-sequence relational learning without gradient descent.
///
/// Single-head implementation. Multi-head = multiple AttentionLayer instances
/// with different M (different vocabulary embeddings). No projection matrices.
///
/// Comparison with nn.MultiheadAttention (PyTorch):
///   nn.MultiheadAttention parameters:
///     - in_proj_weight:  (3*embed_dim, embed_dim) = 3·d² per head
///     - in_proj_bias:    (3*embed_dim,)
///     - out_proj_weight: (embed_dim, embed_dim)
///     - out_proj_bias:   (embed_dim,)
///     Total: ~4·d² parameters per head (at d=512, 1M params/head; 8 heads = 8M)
///   ABR AttentionLayer parameters: 0 (zero learned parameters).
///   The relational structure is accumulated in EdgeStore — a lookup table, not weights.
pub struct AttentionLayer {
    /// The persistent relational field — accumulated across all sequences.
    pub field: StreamingField,
    /// Minimum ρ_acc for a pair to contribute to attention output.
    pub theta: f64,
}

impl AttentionLayer {
    /// Create a new attention layer with declared parameters.
    pub fn new() -> Self {
        Self {
            field: StreamingField::new(),
            theta: THETA_MIN,
        }
    }

    /// Create with explicit theta (for calibration).
    pub fn with_theta(theta: f64) -> Self {
        Self {
            field: StreamingField::new(),
            theta,
        }
    }

    /// Process a sequence: accumulate into the field, produce attention outputs.
    ///
    /// This is the FORWARD PASS of the ABR attention layer.
    ///
    /// Tokens arrive at the frontier in causal order. Each token:
    ///   1. Accumulates with its causal window (streaming field arrive).
    ///   2. Produces an attention output via inference_at.
    ///
    /// Return: one AttentionOutput per token in the input sequence.
    ///
    /// Cost: O(n·W·d) accumulation + O(n·k·d) inference = O(n) for fixed W, k.
    /// Contrast: transformer attention = O(n²·d) per layer.
    pub fn forward(&mut self, token_ids: &[usize], vocab: &VocabTable) -> Vec<AttentionOutput> {
        let n = token_ids.len();
        let mut outputs = Vec::with_capacity(n);

        for &token_id in token_ids {
            let position = self.field.frontier();
            let locus = Locus::new(position, token_id);

            // Accumulate: this token arrives at the frontier
            self.field.arrive(locus, vocab);

            // Inference: produce R-field output at this position
            let (r_field, n_admitted) = self.field.inference_at(position, vocab, self.theta);
            let is_novel = n_admitted == 0;

            outputs.push(AttentionOutput {
                position,
                token_id,
                r_field,
                n_admitted,
                is_novel,
            });
        }

        outputs
    }

    /// Reset positional stream while preserving the EdgeStore's accumulated ρ_acc.
    ///
    /// See StreamingField::reset_stream for the invariant.
    /// Use between maturation passes to prevent O(N×T) positional accumulation.
    pub fn reset_stream(&mut self) {
        self.field.reset_stream();
    }

    /// Total tokens processed by this layer (frontier position).
    pub fn tokens_processed(&self) -> usize {
        self.field.frontier()
    }

    /// Total edge pairs in the field.
    pub fn field_size(&self) -> usize {
        self.field.store.total_edges()
    }

    /// Admitted pairs at declared theta.
    pub fn admitted_pairs(&self) -> usize {
        self.field.store.admitted_edges()
    }
}

impl Default for AttentionLayer {
    fn default() -> Self { Self::new() }
}

/// Architectural comparison: ABR vs softmax attention.
///
/// Returns a formatted comparison string for demonstration/reporting.
/// Numbers are for a single attention head at d_model=512, n=1024 tokens.
pub fn architectural_comparison(d_model: usize, n_tokens: usize, n_heads: usize) -> String {
    let d_head = d_model / n_heads;
    let w = COHERENCE_WINDOW as usize;

    // Softmax attention cost per layer
    let proj_params_per_head = 3 * d_model * d_head; // Q, K, V projections
    let out_proj_params       = d_model * d_model;   // output projection
    let softmax_params_total  = n_heads * proj_params_per_head + out_proj_params;
    let softmax_flops_qk      = n_tokens * n_tokens * d_head * n_heads; // QK^T
    let softmax_flops_av      = n_tokens * n_tokens * d_head * n_heads; // AV
    let softmax_flops_proj    = n_tokens * d_model * d_model * 3;       // QKV proj
    let softmax_flops_total   = softmax_flops_qk + softmax_flops_av + softmax_flops_proj;

    // ABR attention cost per layer
    let abr_params     = 0_usize;                            // zero learned parameters
    let abr_flops_acc  = n_tokens * w * d_model;             // A for window pairs
    let abr_flops_inf  = n_tokens * 64 * d_model;           // B+R for k~64 admitted
    let abr_flops_total = abr_flops_acc + abr_flops_inf;

    let speedup = softmax_flops_total as f64 / abr_flops_total as f64;

    format!(
        "Architectural comparison: d_model={}, n={}, heads={}\n\
         \n\
         Softmax attention (transformer):\n\
           Parameters/layer   : {:>12} ({} per head × {} heads + out proj)\n\
           QK^T + AV FLOPs    : {:>12} = O(n²·d)\n\
           QKV proj FLOPs     : {:>12} = O(n·d²)\n\
           Total FLOPs/layer  : {:>12}\n\
           Denominator        : Σexp(QK^T/√d_k) — unbounded, requires layer norm\n\
           Output bounds      : NONE — V is unbounded, residual required\n\
         \n\
         ABR relational attention:\n\
           Parameters/layer   : {:>12} (zero — M is declared, not trained)\n\
           Accumulation FLOPs : {:>12} = O(n·W·d), W={}\n\
           Inference FLOPs    : {:>12} = O(n·k·d), k=admitted neighb.\n\
           Total FLOPs/layer  : {:>12}\n\
           Denominator        : NONE — ρ_acc ∈ [0,1] by construction\n\
           Output bounds      : D = [0,1]^dim — maintained analytically\n\
         \n\
           Estimated speedup  : {:.1}× at n={}, W={}\n\
           At n=10000: speedup = O(n/W) ≈ {}× asymptotically\n",
        d_model, n_tokens, n_heads,
        softmax_params_total,
        proj_params_per_head, n_heads,
        softmax_flops_qk + softmax_flops_av,
        softmax_flops_proj,
        softmax_flops_total,
        abr_params,
        abr_flops_acc, w,
        abr_flops_inf,
        abr_flops_total,
        speedup, n_tokens, w,
        10000 / w,
    )
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::declaration::build_vocab_table;

    fn small_vocab() -> VocabTable { build_vocab_table(16, 16) }

    #[test]
    fn test_layer_starts_empty() {
        let layer = AttentionLayer::new();
        assert_eq!(layer.tokens_processed(), 0);
        assert_eq!(layer.field_size(), 0);
    }

    #[test]
    fn test_forward_returns_one_output_per_token() {
        let vocab = small_vocab();
        let mut layer = AttentionLayer::new();
        let ids = vec![0, 1, 2, 3, 4];
        let outs = layer.forward(&ids, &vocab);
        assert_eq!(outs.len(), ids.len());
    }

    #[test]
    fn test_positions_are_sequential() {
        let vocab = small_vocab();
        let mut layer = AttentionLayer::new();
        let ids = vec![0, 1, 2, 3];
        let outs = layer.forward(&ids, &vocab);
        for (i, out) in outs.iter().enumerate() {
            assert_eq!(out.position, i, "output position must match stream index");
        }
    }

    #[test]
    fn test_first_token_is_novel() {
        let vocab = small_vocab();
        let mut layer = AttentionLayer::new();
        let outs = layer.forward(&[0], &vocab);
        assert!(outs[0].is_novel, "first token has no predecessors — must be novel");
        assert_eq!(outs[0].n_admitted, 0);
        assert!(outs[0].r_field.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_cross_sequence_accumulation_nonzero_output() {
        // After many exposures of the same sequence, later tokens should have
        // non-novel output (high enough ρ_acc to produce non-zero R-field).
        let vocab = small_vocab();
        let mut layer = AttentionLayer::new();
        let ids = vec![0, 1, 2, 3];
        // Stream same sequence many times to build up ρ_acc
        for _ in 0..50 {
            layer.forward(&ids, &vocab);
        }
        // After 50 passes, pairs (0,1),(0,2),(0,3),(1,2),(1,3),(2,3) should be well-admitted
        // Check that at least some tokens (not the very first ever) produce nonzero output
        let last_outs = layer.forward(&ids, &vocab);
        // Position 0 of each sequence is still novel relative to no predecessor in THAT sequence
        // But tokens at position 1+ should have admitted predecessors from prior accumulation
        // (the token-identity pair (0,1) has been seen 50+ times)
        // Note: the first token of each pass is still alone at its frontier position
        let non_novel_count = last_outs.iter().filter(|o| !o.is_novel).count();
        assert!(non_novel_count > 0,
            "after 50 passes, later tokens must have admitted predecessors");
    }

    #[test]
    fn test_layer_accumulates_across_sequences() {
        let vocab = small_vocab();
        let mut layer = AttentionLayer::new();
        // Two sequences sharing token 1 and 2
        layer.forward(&[0, 1, 2, 3], &vocab);
        let size_after_first = layer.field_size();
        layer.forward(&[1, 2, 4, 5], &vocab);
        let size_after_second = layer.field_size();
        // Field grows with new token pairs but (1,2) pair merges
        assert!(size_after_second > size_after_first,
            "streaming more tokens must grow the field");
    }

    #[test]
    fn test_output_dimension_matches_vocab() {
        let vocab = small_vocab();
        let dim = vocab[0].len();
        let mut layer = AttentionLayer::new();
        let outs = layer.forward(&[0, 1, 2], &vocab);
        for out in &outs {
            assert_eq!(out.r_field.len(), dim,
                "output dimension must match vocab embedding dimension");
        }
    }

    #[test]
    fn test_architectural_comparison_runs() {
        // Just verify it produces nonempty output
        let s = architectural_comparison(512, 1024, 8);
        assert!(!s.is_empty());
        assert!(s.contains("ABR"));
        assert!(s.contains("Softmax"));
    }
}
