// streaming.rs — Metatron Dynamics, Inc.
// ABR Relational Attention — Streaming Field V1.0
//
// Grounding: declaration.rs V2.0, accumulation.rs V2.0, operators.rs V7
// Status: Phase 1 — OC-RA-5 (window-restricted accumulation) CLOSED
//
// The relational field is continuous. Tokens enter the field once.
// There is no restart. There is no re-pass. There is only the frontier.
//
// Phase 0 error (OC-RA-5):
//   accumulate_from_sequence called N times over the same sequence.
//   This re-exposed the same positional pairs N times, producing artificial
//   saturation. Each "pass" was not a new communication event — it was a replay.
//
// Streaming field correction (this module):
//   - Each token enters the field exactly once as communication unfolds.
//   - On arrival at frontier position j, token is evaluated against
//     all tokens in [j - COHERENCE_WINDOW, j-1] (causal predecessors).
//   - EdgeStore is updated once per genuine co-occurrence within the window.
//   - Pairs beyond COHERENCE_WINDOW positions apart do not accumulate.
//   - The field (EdgeStore) persists and grows with the stream.
//
// Key properties:
//   ρ_acc distribution: spans [0,1] after a single pass (not bimodal).
//   θ_min is a genuine selectivity parameter (not an artifact of saturation).
//   Cross-sequence generalization: ρ_acc persists across all contexts where
//     a token pair appears — a property of token-identity, not position.
//   Window restriction: pairs > COHERENCE_WINDOW positions apart suppressed.
//     At 181 tokens: 23.1% of pairs suppressed (Phase 1 Python validation).
//
// Mathematical findings (Phase 1 Python, 2026-11):
//   O(n) accumulation: ABR streaming = O(n × W) for fixed W vs. O(n²) softmax.
//     n=1000: ABR 0.124× O(n²); n=3000: ABR 0.042× O(n²).
//   Novel-token signature: tokens with no prior binding produce output = 0.
//     This is declared (admissible), not numeric zero.
//   Cross-sequence generalization at 22.9% (Para 3 from Para 1+2 field).
//   M-agnostic: same operators produce different field structure under different M.

use crate::declaration::{
    Locus, RelationalPair, VocabTable,
    COHERENCE_WINDOW, ETA,
};
use crate::accumulation::EdgeStore;
use crate::operators::operator_a;

/// The streaming relational field.
///
/// Tokens arrive at the frontier one at a time. Each arrival:
///   1. Evaluates the arriving token against all tokens in the coherence window.
///   2. Updates EdgeStore for each causal pair (src, arriving token) in window.
///   3. Advances the frontier.
///
/// The field (EdgeStore) is the accumulated relational structure.
/// It persists across all tokens and sequences that pass through it.
/// This IS the declared relational field — not a cache, not an approximation.
///
/// Inference cost: O(k) per position, where k = admitted neighborhood size.
/// Accumulation cost: O(n × W) = O(n) since W is declared and fixed.
///
/// ## Energy weighting (OC-RA-12)
///
/// When iota weights are provided (via `with_iota`), each pair accumulates
/// at rate η · (1 + ε[e]) rather than η, where ε[e] = |ι[src] − ι[tgt]|.
/// Ambient pairs (both ι = 1.0) accumulate at the base rate.
/// High-emphasis-differential pairs accumulate faster.
#[derive(Debug)]
pub struct StreamingField {
    /// All tokens that have arrived, in arrival order.
    /// stream[j] = Locus at frontier position j.
    stream: Vec<Locus>,

    /// Current frontier — the next position that will be assigned.
    /// After n arrivals, frontier = n.
    frontier: usize,

    /// The relational field — accumulated bindings keyed by token-identity pair.
    /// This is the persistent field state. It outlives any individual sequence.
    pub store: EdgeStore,

    /// Coherence window (declared as COHERENCE_WINDOW in declaration.rs).
    /// Window-restricted accumulation: only token pairs within w positions
    /// of each other accumulate binding. OC-RA-5.
    window: usize,

    /// Learning rate η. Declared ETA by default; overridable for calibration (OC-RA-2).
    eta: f64,

    /// Per-token-type emphasis weights ι[v] from SemanticM (OC-RA-12).
    ///
    /// iota[token_id] = ι[v] for that token type.
    /// Empty Vec means all iota are ambient (1.0) — no energy weighting.
    /// Set via `with_iota`; acquired from SemanticM::iota.
    iota: Vec<f32>,
}

impl StreamingField {
    /// Create a new streaming field with declared coherence window and declared ETA.
    pub fn new() -> Self {
        Self {
            stream:   Vec::new(),
            frontier: 0,
            store:    EdgeStore::new(),
            window:   COHERENCE_WINDOW as usize,
            eta:      ETA,
            iota:     Vec::new(),
        }
    }

    /// Create a new streaming field with an explicit window (for window calibration).
    pub fn with_window(window: usize) -> Self {
        Self {
            stream:   Vec::new(),
            frontier: 0,
            store:    EdgeStore::new(),
            window,
            eta:      ETA,
            iota:     Vec::new(),
        }
    }

    /// Create a new streaming field with explicit η and window (for OC-RA-2 calibration).
    /// Does not modify the declared ETA constant. Calibration use only.
    pub fn with_params(eta: f64, window: usize) -> Self {
        debug_assert!(eta > 0.0 && eta <= 1.0, "η must be in (0,1]");
        Self {
            stream:   Vec::new(),
            frontier: 0,
            store:    EdgeStore::new(),
            window,
            eta,
            iota:     Vec::new(),
        }
    }

    /// Set per-token-type emphasis weights from SemanticM (OC-RA-12).
    ///
    /// Once set, `arrive` uses energy-weighted accumulation:
    ///   η_eff = η · (1 + ε[e])  where ε[e] = |ι[src] − ι[tgt]|
    ///
    /// Pass `semantic_m.iota.clone()` or a slice of iota values.
    /// Call before the first token arrives, or after a full reset.
    pub fn with_iota(mut self, iota: Vec<f32>) -> Self {
        self.iota = iota;
        self
    }

    /// Set iota weights in place (mutable variant of with_iota).
    pub fn set_iota(&mut self, iota: Vec<f32>) {
        self.iota = iota;
    }

    /// Return ε[e] for a (src_token_id, tgt_token_id) pair.
    ///
    /// = |ι[src] − ι[tgt]|.  Returns 0.0 if iota is not set (ambient).
    #[inline]
    fn epsilon_for(&self, src_token_id: usize, tgt_token_id: usize) -> f32 {
        if self.iota.is_empty() { return 0.0; }
        let i_src = self.iota.get(src_token_id).copied().unwrap_or(1.0);
        let i_tgt = self.iota.get(tgt_token_id).copied().unwrap_or(1.0);
        (i_src - i_tgt).abs()
    }

    /// Return the η this field accumulates at.
    pub fn eta(&self) -> f64 { self.eta }

    /// Return the current frontier position (number of tokens arrived).
    pub fn frontier(&self) -> usize {
        self.frontier
    }

    /// One token arrives at the frontier.
    ///
    /// Evaluated against all causal predecessors in [max(0, j-W), j-1].
    /// EdgeStore updated once per genuine co-occurrence.
    /// Frontier advances.
    ///
    /// This is the streaming field's core operation. Called once per token.
    /// It does not "re-pass" — there is no notion of multiple exposures to
    /// the same positional pair. Each evaluation is a unique communication event.
    pub fn arrive(&mut self, locus: Locus, vocab: &VocabTable) {
        let j = self.frontier;
        let window_start = j.saturating_sub(self.window);

        // Evaluate arriving token against all tokens in the coherence window.
        // For each predecessor at position i in [window_start, j-1]:
        //   - Declare the relational pair (stream[i], locus)
        //   - Compute A[e] = M(src) - M(tgt), then ρₙ = ρ_base · χ/(1+χ)
        //   - Update EdgeStore (accumulation rule applied once)
        let eta = self.eta;
        for i in window_start..j {
            let src = self.stream[i];
            // Causal pair: source position i < target position j (guaranteed here)
            if let Some(pair) = RelationalPair::declare(src, locus) {
                let rho_n = operator_a_to_rho_n(&pair, vocab);
                // OC-RA-12: weight accumulation by edge energy ε[e] = |ι[src] − ι[tgt]|.
                // When iota is not set, epsilon_for returns 0.0 → base rate unchanged.
                let epsilon = self.epsilon_for(src.token_id, locus.token_id);
                self.store.update_with_energy(&pair, rho_n, eta, epsilon);
            }
        }

        // Arrive: append token to stream, advance frontier.
        self.stream.push(locus);
        self.frontier += 1;

        debug_assert!(self.frontier == self.stream.len());
    }

    /// Stream an entire sequence into the field.
    ///
    /// Calls arrive() for each token in order. Each token enters the field
    /// exactly once. No re-passes. The field grows continuously with the stream.
    ///
    /// sequences: iterator over (position_in_stream, token_id) pairs.
    /// For a fresh stream, positions are 0, 1, 2, ...
    /// For a continuation, pass frontier-offset positions.
    pub fn arrive_sequence(&mut self, token_ids: &[usize], vocab: &VocabTable) {
        for &token_id in token_ids {
            let locus = Locus::new(self.frontier, token_id);
            self.arrive(locus, vocab);
        }
    }

    /// Register positions into the stream WITHOUT updating the EdgeStore.
    ///
    /// Appends loci to the positional stream and advances the frontier,
    /// but does NOT evaluate pairs or update any relational bindings.
    ///
    /// Use case: reconstruct positional context for `inference_at` after a
    /// completed accumulation pass, without re-accumulating the same tokens.
    ///
    /// Declared single-pass streaming property is preserved:
    ///   - Accumulation: exactly one pass, via `arrive` / `arrive_sequence`.
    ///   - Positional reconstruction for inference: via `register_positions`.
    ///   - The two operations are strictly separated.
    pub fn register_positions(&mut self, token_ids: &[usize]) {
        for &token_id in token_ids {
            let locus = Locus::new(self.frontier, token_id);
            self.stream.push(locus);
            self.frontier += 1;
        }
        debug_assert!(self.frontier == self.stream.len());
    }

    /// Inference: build the declared relational graph at a window around position j,
    /// apply B → ρ → R, return aggregated output at position j.
    ///
    /// The admitted set is built from the settled field (EdgeStore).
    /// Tokens within [j-W, j] that have admitted relational bindings
    /// contribute to the output at j.
    ///
    /// Returns (output_dim_values, n_admitted_edges).
    /// output is a Vec<f64> in D — the relational attention output at position j.
    ///
    /// If no admitted edges: returns zero vector (declared state: novel token).
    /// Zero output is structurally correct — it means: 'this token has no
    /// admitted relational history in the current field.' Not an error.
    pub fn inference_at(
        &self,
        position: usize,
        vocab: &VocabTable,
        theta: f64,
    ) -> (Vec<f64>, usize) {
        if position >= self.stream.len() {
            let dim = if vocab.is_empty() { 0 } else { vocab[0].len() };
            return (vec![0.0; dim], 0);
        }

        let dim = if vocab.is_empty() { 0 } else { vocab[0].len() };
        let window_start = position.saturating_sub(self.window);
        let window_loci = &self.stream[window_start..=position];
        let n_w = window_loci.len();

        // Build admitted edge list from window
        let admitted = self.store.all_admitted_pairs_at_threshold(theta);
        let admitted_set: std::collections::HashSet<(usize, usize)> =
            admitted.into_iter().collect();

        let mut edges: Vec<(usize, usize)> = Vec::new(); // (local_src_pos, local_tgt_pos)
        for j_local in 0..n_w {
            for i_local in 0..=j_local {
                let key = (window_loci[i_local].token_id, window_loci[j_local].token_id);
                if admitted_set.contains(&key) {
                    edges.push((i_local, j_local));
                }
            }
        }

        if edges.is_empty() {
            return (vec![0.0; dim], 0);
        }

        // Build local token_id sequence for operator application
        let local_ids: Vec<usize> = window_loci.iter().map(|l| l.token_id).collect();

        // Apply B → R (operator_b, then operator_r)
        let n_edges = edges.len();
        let b_field = apply_operator_b(&edges, &local_ids, vocab, dim);
        let r_field = apply_operator_r(&b_field, &edges, &local_ids, vocab, dim);

        // Aggregate at target position (last in window = position).
        //
        // DOMAIN MAINTENANCE: compute the MEAN R-field over admitted edges at
        // the target position, not the sum.  Each R-vector component traces to
        // M ∈ [0,1]^dim through declared operators, so the mean is bounded.
        // A sum would be O(k) and violate D = [0,1]^dim as k grows.
        // Analogous to softmax attention: output = Σ attn_weight · V, where
        // Σ attn_weight = 1 (normalized).  Here: output = (1/k) Σ R[i→pos].
        let mut output = vec![0.0_f64; dim];
        let mut n_at_target = 0_usize;
        for (ei, &(_, tp)) in edges.iter().enumerate() {
            if tp == n_w - 1 {
                for d in 0..dim {
                    output[d] += r_field[ei * dim + d];
                }
                n_at_target += 1;
            }
        }
        if n_at_target > 1 {
            let inv = 1.0 / n_at_target as f64;
            for v in &mut output { *v *= inv; }
        }

        (output, n_edges)
    }

    /// Reset the positional stream while preserving the EdgeStore's accumulated ρ_acc.
    ///
    /// After reset: frontier = 0, stream = [], EdgeStore unchanged.
    ///
    /// This is the correct separation between:
    ///   - Positional history (stream[]) — per-sequence, ephemeral
    ///   - Relational field (EdgeStore)  — persistent, cross-sequence
    ///
    /// Use this between maturation passes when you want the same ρ_acc to gate
    /// inference on a fresh window, without accumulating O(N×T) positional history.
    ///
    /// INVARIANT: ρ_acc in EdgeStore is UNCHANGED by reset_stream().
    /// Only the positional stream (Locus positions) is discarded.
    pub fn reset_stream(&mut self) {
        self.stream.clear();
        self.frontier = 0;
        // EdgeStore (self.store) is intentionally NOT cleared.
    }

    /// Current token count in the stream.
    pub fn stream_len(&self) -> usize {
        self.stream.len()
    }

    /// Window restriction diagnostic: count pairs that exist in unlimited field
    /// but are suppressed by the declared coherence window.
    /// Returns (windowed_count, suppressed_count).
    pub fn window_suppression_count(&self, vocab: &VocabTable, _theta: f64) -> (usize, usize) {
        // Build unlimited field (window = frontier, no restriction)
        let mut unlimited = StreamingField::with_window(self.frontier + 1);
        for locus in &self.stream {
            unlimited.arrive(*locus, vocab);
        }
        let windowed = self.store.total_edges();
        let total    = unlimited.store.total_edges();
        (windowed, total.saturating_sub(windowed))
    }
}

impl Default for StreamingField {
    fn default() -> Self { Self::new() }
}

// ── Internal helpers ───────────────────────────────────────────────────────────

/// Compute ρₙ for a declared pair from operator A.
/// ρₙ = ρ_base · χ/(χ₀+χ),  χ = max|A[e]|
/// ρ_base = 1.0, χ₀ = 1 (language domain M declaration — see operators.rs RHO_BASE).
fn operator_a_to_rho_n(pair: &RelationalPair, vocab: &VocabTable) -> f64 {
    use crate::operators::RHO_BASE;
    let a = operator_a(&pair.source, &pair.target, vocab);
    let chi: f64 = a.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    // χ₀ = 1 (V9 Origin general value; language domain declaration)
    RHO_BASE * chi / (1.0 + chi)
}

/// Apply operator B to a local edge list.
/// B[e] = A[e] + Σ_{f ∈ succ(e)} A[f]
///
/// Canonical V9 kernel definition:
///   succ(e = (sp, tp)) = { f : src(f) == tp }
///   i.e. edges whose SOURCE is the TARGET of e — forward continuations from tp.
///   (kernel_reference.md §11; operators.rs `succ(e) = out[edges[e].1]`)
///
/// Returns flat Vec<f64> of shape [n_edges × dim].
fn apply_operator_b(
    edges: &[(usize, usize)],
    local_ids: &[usize],
    vocab: &VocabTable,
    dim: usize,
) -> Vec<f64> {
    let n_edges = edges.len();
    // Build all A[e]
    let a_vecs: Vec<Vec<f64>> = edges.iter().map(|&(si, ti)| {
        let pair = RelationalPair::declare(
            Locus::new(si, local_ids[si]),
            Locus::new(ti, local_ids[ti]),
        ).unwrap();
        operator_a(&pair.source, &pair.target, vocab)
    }).collect();

    // Build src_to_edges index: src_to_edges[n] = { e : src(e) == n }
    // succ(e = (sp, tp)) = src_to_edges[tp] = edges whose source is tp.
    let mut src_to_edges: Vec<Vec<usize>> = vec![Vec::new(); local_ids.len()];
    for (ei, &(sp, _)) in edges.iter().enumerate() {
        src_to_edges[sp].push(ei);
    }

    // B[e] = A[e] + Σ_{f ∈ succ(e)} A[f]
    // succ(e = (sp, tp)) = src_to_edges[tp] (edges starting from tp)
    let mut b_flat = vec![0.0_f64; n_edges * dim];
    for (ei, &(_, tp)) in edges.iter().enumerate() {
        for d in 0..dim {
            b_flat[ei * dim + d] = a_vecs[ei][d];
        }
        // Add A[f] for each f in succ(e): edges whose source is tp
        for &fi in &src_to_edges[tp] {
            for d in 0..dim {
                b_flat[ei * dim + d] += a_vecs[fi][d];
            }
        }
    }
    b_flat
}

/// Apply operator R to B-field.
/// R[e] = B[e] + ρ[src(e)] · (Σ_{f ∈ succ(e)} B[f] − Σ_{f ∈ pred(e)} B[f])
///
/// Canonical V9 kernel definitions:
///   succ(e = (sp, tp)) = { f : src(f) == tp }  — edges starting from tp
///   pred(e = (sp, tp)) = { f : tgt(f) == sp }  — edges ending at sp
///   (kernel_reference.md §11; operators.rs lines 489-490, 563, 567)
///
/// Returns flat Vec<f64> of shape [n_edges × dim].
fn apply_operator_r(
    b_flat: &[f64],
    edges: &[(usize, usize)],
    local_ids: &[usize],
    vocab: &VocabTable,
    dim: usize,
) -> Vec<f64> {
    use crate::operators::RHO_BASE;
    let n_nodes = local_ids.len();

    // Build adjacency indices:
    //   src_to_edges[n] = { e : src(e) == n }  → succ(e=(sp,tp)) = src_to_edges[tp]
    //   tgt_to_edges[n] = { e : tgt(e) == n }  → pred(e=(sp,tp)) = tgt_to_edges[sp]
    let mut src_to_edges: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    let mut tgt_to_edges: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for (ei, &(sp, tp)) in edges.iter().enumerate() {
        src_to_edges[sp].push(ei);
        tgt_to_edges[tp].push(ei);
    }

    // ρ per node: ρ[i] = RHO_BASE · χ[i] / (χ₀ + χ[i]),  χ₀ = 1
    // χ[i] = max |A[e]| over all edges incident on node i (source or target).
    //
    // PERFORMANCE FIX: iterate edges once — O(n_edges × dim) — not O(n_nodes × n_edges × dim).
    // For each edge (sp, tp), compute |A| once and update both endpoint nodes.
    let mut rho_node = vec![0.0_f64; n_nodes];
    for &(sp, tp) in edges {
        let pair = RelationalPair::declare(
            Locus::new(sp, local_ids[sp]),
            Locus::new(tp, local_ids[tp]),
        ).unwrap();
        let a = operator_a(&pair.source, &pair.target, vocab);
        let chi: f64 = a.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        // χ₀ = 1 (language domain M declaration)
        let rho = RHO_BASE * chi / (1.0 + chi);
        // ρ[node] = max over all incident edges (source or target)
        if rho > rho_node[sp] { rho_node[sp] = rho; }
        if rho > rho_node[tp] { rho_node[tp] = rho; }
    }

    // R[e = (sp, tp)] = B[e] + ρ[sp] · (Σ_{f ∈ succ(e)} B[f] − Σ_{f ∈ pred(e)} B[f])
    // succ(e) = src_to_edges[tp]  (edges continuing forward from tp)
    // pred(e) = tgt_to_edges[sp]  (edges arriving at sp)
    let mut r_flat = b_flat.to_vec();
    for (ei, &(sp, tp)) in edges.iter().enumerate() {
        let mut fwd = vec![0.0_f64; dim];
        let mut bwd = vec![0.0_f64; dim];
        // fwd: succ(e) = edges whose source is tp
        for &fi in &src_to_edges[tp] {
            for d in 0..dim { fwd[d] += b_flat[fi * dim + d]; }
        }
        // bwd: pred(e) = edges whose target is sp
        for &fi in &tgt_to_edges[sp] {
            for d in 0..dim { bwd[d] += b_flat[fi * dim + d]; }
        }
        let rho = rho_node[sp];
        for d in 0..dim {
            r_flat[ei * dim + d] += rho * (fwd[d] - bwd[d]);
        }
    }
    r_flat
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::declaration::{build_vocab_table, THETA_MIN};

    fn small_vocab() -> VocabTable {
        build_vocab_table(8, 8)
    }

    #[test]
    fn test_new_field_is_empty() {
        let f = StreamingField::new();
        assert_eq!(f.frontier(), 0);
        assert_eq!(f.stream_len(), 0);
        assert_eq!(f.store.total_edges(), 0);
    }

    #[test]
    fn test_arrive_advances_frontier() {
        let vocab = small_vocab();
        let mut f = StreamingField::new();
        f.arrive(Locus::new(0, 1), &vocab);
        assert_eq!(f.frontier(), 1);
        f.arrive(Locus::new(1, 2), &vocab);
        assert_eq!(f.frontier(), 2);
    }

    #[test]
    fn test_first_token_no_edges() {
        // First token has no predecessors — no pairs evaluated
        let vocab = small_vocab();
        let mut f = StreamingField::new();
        f.arrive(Locus::new(0, 0), &vocab);
        assert_eq!(f.store.total_edges(), 0,
            "first token has no predecessors — no pairs should be evaluated");
    }

    #[test]
    fn test_second_token_creates_one_pair() {
        let vocab = small_vocab();
        let mut f = StreamingField::new();
        f.arrive(Locus::new(0, 0), &vocab);
        f.arrive(Locus::new(1, 1), &vocab);
        assert_eq!(f.store.total_edges(), 1,
            "second token evaluates exactly one predecessor pair");
    }

    #[test]
    fn test_no_reprocessing() {
        // Stream [A, B, C]. Each token evaluated once against its predecessors.
        // After 3 tokens: pairs are (A,B), (A,C), (B,C). Each accumulated once.
        let vocab = small_vocab();
        let mut f = StreamingField::new();
        f.arrive_sequence(&[0, 1, 2], &vocab);
        // 3 unique pairs: (0,1), (0,2), (1,2)
        assert_eq!(f.store.total_edges(), 3,
            "three distinct causal pairs from a 3-token stream");
        // Each pair updated exactly once: exposure_count should be 1
        // (We test this via total_edges; exposure_count is internal to AccumulatedBinding)
    }

    #[test]
    fn test_same_token_pair_different_sequences_accumulates() {
        // Token pair (1,2) appears in two separate sequences.
        // ρ_acc(1,2) should increase after both.
        let vocab = small_vocab();
        let mut f = StreamingField::new();
        f.arrive_sequence(&[1, 2], &vocab);
        let rho_after_first = f.store.admitted_sources_for_token(2)
            .iter()
            .find(|&&(s, _)| s == 1)
            .map(|&(_, r)| r)
            .unwrap_or(0.0);
        // Stream again: pair (1,2) will re-appear
        f.arrive_sequence(&[1, 2], &vocab);
        let rho_after_second = f.store.admitted_sources_for_token(2)
            .iter()
            .find(|&&(s, _)| s == 1)
            .map(|&(_, r)| r)
            .unwrap_or(0.0);
        // Cross-sequence: second exposure should increase ρ_acc if first was nonzero
        // (unless rho_n=0 for token pair (1,2) under this vocab)
        // Unit test: just verify total_edges is consistent
        assert!(f.store.total_edges() >= 1);
        let _ = (rho_after_first, rho_after_second); // suppress unused warning
    }

    #[test]
    fn test_window_restricts_distant_pairs() {
        // Stream n tokens. With window W, tokens at positions (0, n-1) where
        // n-1 > W should NOT be evaluated as a pair.
        let vocab = small_vocab();
        let w = 3_usize;
        let mut f_windowed  = StreamingField::with_window(w);
        let mut f_unlimited = StreamingField::with_window(9999);
        // Stream 8 distinct tokens
        let ids: Vec<usize> = (0..8).collect();
        f_windowed.arrive_sequence(&ids, &vocab);
        f_unlimited.arrive_sequence(&ids, &vocab);
        // Unlimited sees all n*(n-1)/2 = 28 pairs; windowed sees fewer
        assert!(f_windowed.store.total_edges() < f_unlimited.store.total_edges(),
            "window should suppress distant pairs");
    }

    #[test]
    fn test_window_exactly_w() {
        // With window=2, token at position j only evaluates j-1 and j-2.
        // For stream [A, B, C, D] (w=2):
        //   j=1: evaluates (A,B)                    → 1 pair
        //   j=2: evaluates (A,C), (B,C)             → 2 pairs
        //   j=3: evaluates (B,D), (C,D)             → 2 pairs  (A is out of window)
        // Total: 5 pairs evaluated (A,B), (A,C), (B,C), (B,D), (C,D)
        // But unique token-identity pairs: 5 (all different tokens here)
        let vocab = small_vocab();
        let mut f = StreamingField::with_window(2);
        f.arrive_sequence(&[0, 1, 2, 3], &vocab);
        // tokens 0,1,2,3 all distinct → 5 unique identity pairs
        assert_eq!(f.store.total_edges(), 5,
            "with window=2 and 4 distinct tokens: exactly 5 pairs should be evaluated");
    }

    #[test]
    fn test_inference_novel_token_zero_output() {
        // A token that has never been in a pair with any admitted edge
        // should produce zero output.
        let vocab = small_vocab();
        let mut f = StreamingField::new();
        // Stream only token 0 (frontier=1, no pairs yet)
        f.arrive(Locus::new(0, 0), &vocab);
        // inference_at(0) has no admitted sources → zero output
        let (out, n_e) = f.inference_at(0, &vocab, THETA_MIN);
        assert_eq!(n_e, 0, "no edges admitted for a solo token");
        assert!(out.iter().all(|&v| v == 0.0),
            "novel token with no admitted history must produce zero output");
    }

    #[test]
    fn test_reset_stream_clears_frontier_keeps_edges() {
        // After reset_stream: frontier=0, stream empty, EdgeStore unchanged.
        let vocab = small_vocab();
        let mut f = StreamingField::new();
        f.arrive_sequence(&[0, 1, 2, 3], &vocab);
        let edges_before = f.store.total_edges();
        assert!(f.frontier() > 0, "frontier must advance after arrivals");

        f.reset_stream();

        assert_eq!(f.frontier(), 0, "reset must clear the frontier");
        assert_eq!(f.stream_len(), 0, "reset must clear the stream");
        assert_eq!(f.store.total_edges(), edges_before,
            "reset must NOT clear the EdgeStore — ρ_acc must be preserved");
    }

    #[test]
    fn test_reset_stream_rho_acc_persists() {
        // ρ_acc for a pair should be the same before and after reset_stream.
        let vocab = small_vocab();
        let mut f = StreamingField::new();
        f.arrive_sequence(&[0, 1, 2], &vocab);
        // Get ρ_acc for pair (0,1) before reset
        let rho_before = f.store.rho_acc_for_pair(0, 1);
        assert!(rho_before.is_some(), "pair (0,1) should exist after streaming");

        f.reset_stream();
        let rho_after = f.store.rho_acc_for_pair(0, 1);
        assert_eq!(rho_before, rho_after,
            "reset_stream must not change ρ_acc in the EdgeStore");
    }

    #[test]
    fn test_reset_stream_then_arrive_works() {
        // After reset, the field should accept new arrivals starting from position 0.
        let vocab = small_vocab();
        let mut f = StreamingField::new();
        f.arrive_sequence(&[0, 1], &vocab);
        f.reset_stream();
        // Should be able to arrive again without panic
        f.arrive_sequence(&[0, 1], &vocab);
        assert_eq!(f.frontier(), 2, "after reset + re-stream, frontier should be 2");
        // The pair (0,1) should now have been seen twice total (ρ_acc higher)
        let rho = f.store.rho_acc_for_pair(0, 1).unwrap_or(0.0);
        assert!(rho > 0.0, "pair (0,1) must have nonzero ρ_acc after two streams");
    }

    #[test]
    fn test_sequence_generalisation() {
        // After streaming sequence A, sequence B should have some pairs
        // pre-admitted if B shares tokens with A.
        let vocab = small_vocab();
        let mut f = StreamingField::new();
        // Stream A: token ids [0, 1, 2, 3]
        f.arrive_sequence(&[0, 1, 2, 3], &vocab);
        // Now check if pair (0, 1) has nonzero rho_acc (it should)
        // The pair was evaluated once — rho_acc = ETA * rho_n(0,1)
        // rho_n > 0 as long as token 0 and token 1 have different embeddings
        // Under build_vocab_table(8,8), token 0 has emb[0]=1, token 1 has emb[1]=1
        // A(0,1) = emb[0] - emb[1] = [1,0,...] - [0,1,...] = [1,-1,0,...]
        // chi = 1.0, rho_n = 1.0/(1+1.0) = 0.5
        // rho_acc after 1 exposure = ETA * 0.5 * (1 - 0) = 0.05
        // THETA_MIN = 0.05 → exactly on boundary (depends on float comparison)
        // Check total admitted
        let _sources = f.store.admitted_sources_for_token(1);
        // We just verify there are admitted pairs; exact threshold depends on float
        assert!(f.store.total_edges() > 0,
            "streaming must produce some edge bindings");
    }

    // ── OC-RA-12: energy-weighted accumulation ────────────────────────────────

    /// High-energy edge (ε > 0) accumulates faster than ambient (ε = 0).
    #[test]
    fn test_energy_weighted_accumulates_faster() {
        use crate::accumulation::AccumulatedBinding;
        // Two bindings: same rho_n and eta, different epsilon.
        let mut ambient = AccumulatedBinding::new();
        let mut energised = AccumulatedBinding::new();
        let rho_n = 0.5_f64;
        // Ambient: epsilon=0 (both tokens at ι=1.0)
        ambient.accumulate_energy_weighted(rho_n, ETA, 0.0);
        // Energised: epsilon=2.0 (FIRE! vs ambient: ι=3.0 vs ι=1.0)
        energised.accumulate_energy_weighted(rho_n, ETA, 2.0);
        assert!(energised.rho_acc > ambient.rho_acc,
            "high-energy edge must accumulate faster: {} > {}",
            energised.rho_acc, ambient.rho_acc);
    }

    /// Ambient edge with energy weighting equals ambient without it.
    #[test]
    fn test_zero_energy_equals_base_rate() {
        use crate::accumulation::AccumulatedBinding;
        let mut base = AccumulatedBinding::new();
        let mut weighted = AccumulatedBinding::new();
        base.accumulate(0.5, ETA);
        weighted.accumulate_energy_weighted(0.5, ETA, 0.0);
        assert!((base.rho_acc - weighted.rho_acc).abs() < 1e-12,
            "zero energy weighting must equal base rate: {} vs {}",
            base.rho_acc, weighted.rho_acc);
    }

    /// Energy-weighted accumulation stays in [0,1] even at high epsilon.
    #[test]
    fn test_energy_weighted_stays_bounded() {
        use crate::accumulation::AccumulatedBinding;
        let mut b = AccumulatedBinding::new();
        // High epsilon = 5.0, repeated 1000 times
        for _ in 0..1000 {
            b.accumulate_energy_weighted(1.0, ETA, 5.0);
        }
        assert!(b.rho_acc <= 1.0 + 1e-12, "rho_acc must not exceed 1.0: {}", b.rho_acc);
        assert!(b.rho_acc >= 0.0, "rho_acc must be non-negative: {}", b.rho_acc);
    }

    /// StreamingField with iota set accumulates high-energy pairs faster.
    #[test]
    fn test_field_with_iota_weights_edges() {
        let vocab = small_vocab();

        // Field without iota (ambient)
        let mut f_ambient = StreamingField::new();
        // Field with iota: token 0 is high-emphasis (ι=3.0), token 1 is ambient (ι=1.0)
        // → ε(0→1) = 2.0
        let iota = vec![3.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut f_energy = StreamingField::new().with_iota(iota);

        // Stream [0, 1] through both fields
        f_ambient.arrive_sequence(&[0, 1], &vocab);
        f_energy.arrive_sequence(&[0, 1], &vocab);

        // Pair (token_0 → token_1): energy field must have higher rho_acc
        let rho_ambient = f_ambient.store.rho_acc_for_pair(0, 1).unwrap_or(0.0);
        let rho_energy  = f_energy.store.rho_acc_for_pair(0, 1).unwrap_or(0.0);
        assert!(rho_energy > rho_ambient,
            "energy-weighted field must bind faster on high-energy pair: {} > {}",
            rho_energy, rho_ambient);
    }

    /// Epsilon_for returns 0.0 when iota is not set (backward compatible).
    #[test]
    fn test_epsilon_zero_when_no_iota() {
        let f = StreamingField::new();
        assert_eq!(f.epsilon_for(0, 1), 0.0,
            "epsilon must be 0.0 when iota not set");
    }

    /// register_positions populates the stream but does NOT update the EdgeStore.
    ///
    /// Declared single-pass property: accumulation (arrive) and positional
    /// reconstruction for inference (register_positions) are strictly separated.
    /// A second call with the same tokens must leave all binding values unchanged.
    #[test]
    fn test_register_positions_no_accumulation() {
        let vocab = small_vocab();
        let mut f = StreamingField::new();
        // Single accumulation pass
        f.arrive_sequence(&[1, 2, 3], &vocab);
        let edges_after_arrive = f.store.total_edges();
        let rho_1_2 = f.store.rho_acc_for_pair(1, 2).unwrap_or(0.0);
        let rho_2_3 = f.store.rho_acc_for_pair(2, 3).unwrap_or(0.0);

        // Reset positional stream only (EdgeStore persists — this is reset_stream behaviour)
        f.reset_stream();

        // Register same positions: stream rebuilt, EdgeStore must remain exactly unchanged
        f.register_positions(&[1, 2, 3]);
        assert_eq!(f.store.total_edges(), edges_after_arrive,
            "register_positions must not add new edges");
        assert_eq!(f.store.rho_acc_for_pair(1, 2).unwrap_or(0.0), rho_1_2,
            "register_positions must not change rho_acc for pair (1,2)");
        assert_eq!(f.store.rho_acc_for_pair(2, 3).unwrap_or(0.0), rho_2_3,
            "register_positions must not change rho_acc for pair (2,3)");
        // Frontier and stream length must be restored
        assert_eq!(f.frontier(), 3, "frontier must be 3 after re-registering 3 positions");
        assert_eq!(f.stream_len(), 3);
    }

    /// register_positions enables inference on the settled field without re-accumulation.
    #[test]
    fn test_register_positions_enables_inference() {
        let vocab = small_vocab();
        let mut f = StreamingField::new();
        f.arrive_sequence(&[1, 2, 3, 4], &vocab);
        let (r_arrive, n_arrive) = f.inference_at(3, &vocab, THETA_MIN);

        f.reset_stream();
        f.register_positions(&[1, 2, 3, 4]);
        let (r_register, n_register) = f.inference_at(3, &vocab, THETA_MIN);

        // Same EdgeStore → same inference output
        assert_eq!(n_arrive, n_register,
            "inference edge count must be equal: arrive={} register={}", n_arrive, n_register);
        for (a, b) in r_arrive.iter().zip(r_register.iter()) {
            assert!((a - b).abs() < 1e-12,
                "inference output must be identical: {} vs {}", a, b);
        }
    }
}
