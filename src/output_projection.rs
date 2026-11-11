// output_projection.rs — Metatron Dynamics, Inc.
// ABR Relational Attention — Output Projection W_O V1.0
//
// Grounding: declaration.rs V2.0, layer_stack.rs V1.0
//
// W_O IS THE ONLY LEARNED PARAMETER IN ABR RELATIONAL ATTENTION.
//
// Everything else in ABR is accumulated or declared:
//   - M (embedding table): declared (GloVe/identity) or derived from R-fields
//   - ρ_acc (EdgeStore): accumulated from data via the streaming rule
//   - A, B, R (operators): declared closed-form expressions
//   - ETA, THETA_MIN, W (hyperparameters): declared constants
//
// W_O alone is trained by gradient descent.
//
// ARCHITECTURAL ROLE:
//   The ABR relational field produces R ∈ ℝ^dim at each token position.
//   R encodes the relational context — which prior tokens have admitted
//   bindings and what their combined relational geometry is.
//   W_O maps R → logits ∈ ℝ^vocab_size, enabling:
//     - Language modeling: predict next token P(t+1 | t, t-1, ...)
//     - Classification: predict label P(y | sequence)
//     - Generation: sample from P(t+1 | R_t)
//
// WHY W_O IS SUFFICIENT:
//   In transformer-based models, there are three sets of learned parameters:
//     1. W_Q, W_K, W_V: query/key/value projection matrices (the attention parameters)
//     2. FFN weights: feed-forward network (the capacity)
//     3. W_O: output projection (the prediction head)
//   ABR replaces (1) entirely with the declared relational field.
//   The claim is that (1) can be replaced without (2) for many tasks,
//   because the relational field accumulates the necessary structure from data.
//   W_O then learns to decode that structure into predictions.
//
// PARAMETER COUNT:
//   W_O: dim × vocab_size parameters.
//   For dim=50, vocab=10000: 500,000 parameters.
//   GPT-2 small W_Q + W_K + W_V: 12 heads × 3 × 768 × 64 = 1,769,472 per layer.
//   ABR with W_O only: 500,000 total (across all layers, shared or not).
//
// TRAINING:
//   Mini-batch SGD (or full-batch gradient descent).
//   Loss: cross-entropy on the target token distribution.
//   Gradient of cross-entropy w.r.t. W_O logits:
//     ∂L/∂logits_i = softmax(logits)_i - 1(i == target)
//     ∂L/∂W_i = (∂L/∂logits_i) · R^T
//     ∂L/∂b_i = ∂L/∂logits_i
//   The gradient does NOT flow back through the relational field.
//   ABR is a fixed representation; W_O is the learned readout.

/// Output projection W_O: maps R-field output → logits over vocabulary.
///
/// W: (vocab_out × dim)  — projection matrix
/// b: (vocab_out,)       — bias vector
///
/// forward(r): logits = W · r + b  (unnormalized log-probabilities)
/// update(): SGD gradient step (the only place gradient descent is used in ABR)
#[derive(Debug)]
pub struct OutputProjection {
    /// Projection matrix W: shape [vocab_out][dim].
    /// Row i is the weight vector for output class i.
    pub w: Vec<Vec<f64>>,
    /// Bias vector b: shape [vocab_out].
    pub b: Vec<f64>,
    /// Input dimension (R-field dim = 50 for GloVe-50d).
    pub dim: usize,
    /// Output vocabulary size.
    pub vocab_out: usize,
}

impl OutputProjection {
    /// Create a new W_O with Xavier/Glorot uniform initialization.
    ///
    /// Xavier initialization: W[i][j] ~ Uniform(-scale, +scale)
    ///   scale = sqrt(6 / (dim + vocab_out))
    /// Bias initialized to zero.
    ///
    /// Uses a deterministic LCG for reproducibility.
    pub fn new_random(dim: usize, vocab_out: usize) -> Self {
        let scale = (6.0 / (dim + vocab_out) as f64).sqrt();
        let mut rng = 42u64; // deterministic seed

        let w: Vec<Vec<f64>> = (0..vocab_out).map(|_| {
            (0..dim).map(|_| {
                scale * (2.0 * lcg(&mut rng) - 1.0)
            }).collect()
        }).collect();

        Self {
            b: vec![0.0; vocab_out],
            w,
            dim,
            vocab_out,
        }
    }

    /// Create a W_O initialized to zeros (for testing).
    pub fn new_zero(dim: usize, vocab_out: usize) -> Self {
        Self {
            w: vec![vec![0.0; dim]; vocab_out],
            b: vec![0.0; vocab_out],
            dim,
            vocab_out,
        }
    }

    /// Forward pass: compute logits = W · r + b.
    ///
    /// r: R-field output at one token position, shape [dim].
    /// Returns logits (unnormalized), shape [vocab_out].
    ///
    /// These logits feed directly into softmax or log-softmax.
    /// Do not apply softmax here — training uses log-softmax + NLL
    /// for numerical stability.
    pub fn forward(&self, r: &[f64]) -> Vec<f64> {
        debug_assert_eq!(r.len(), self.dim);
        (0..self.vocab_out).map(|i| {
            let dot: f64 = self.w[i].iter().zip(r.iter()).map(|(w, x)| w * x).sum();
            dot + self.b[i]
        }).collect()
    }

    /// Compute cross-entropy loss from raw logits and a target class.
    ///
    /// Uses numerically stable log-softmax:
    ///   log_softmax[i] = logits[i] - max - log(Σ exp(logits[j] - max))
    ///   loss = -log_softmax[target]
    ///
    /// Returns (loss, log_softmax) so the caller can reuse log_softmax for
    /// gradient computation without recomputing.
    pub fn cross_entropy_and_log_softmax(
        logits: &[f64],
        target: usize,
    ) -> (f64, Vec<f64>) {
        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_shifted: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exp: f64 = exp_shifted.iter().sum();
        let log_sum_exp = sum_exp.ln() + max_logit;
        let log_softmax: Vec<f64> = logits.iter().map(|&x| x - log_sum_exp).collect();
        let loss = -log_softmax[target];
        (loss, log_softmax)
    }

    /// Compute perplexity from mean cross-entropy loss.
    /// perplexity = exp(mean_cross_entropy)
    pub fn perplexity(mean_loss: f64) -> f64 {
        mean_loss.exp()
    }

    /// Backward pass: compute gradients given input r, log-softmax output, and target.
    ///
    /// Returns (dW, db) — gradients for W and b respectively.
    ///
    /// Gradient derivation:
    ///   softmax[i] = exp(log_softmax[i])
    ///   ∂L/∂logits_i = softmax[i] - 1(i == target)   (softmax + cross-entropy gradient)
    ///   ∂L/∂W[i][d] = ∂L/∂logits_i · r[d]
    ///   ∂L/∂b[i] = ∂L/∂logits_i
    ///
    /// The gradient does NOT flow through the relational field.
    /// ABR R-fields are treated as fixed (non-differentiable) representations.
    pub fn backward(
        r: &[f64],
        log_softmax: &[f64],
        target: usize,
        vocab_out: usize,
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        // ∂L/∂logits: softmax - one_hot(target)
        let d_logits: Vec<f64> = (0..vocab_out).map(|i| {
            log_softmax[i].exp() - if i == target { 1.0 } else { 0.0 }
        }).collect();

        // ∂L/∂W[i][d] = d_logits[i] · r[d]
        let dw: Vec<Vec<f64>> = d_logits.iter().map(|&dl| {
            r.iter().map(|&x| dl * x).collect()
        }).collect();

        // ∂L/∂b[i] = d_logits[i]
        let db = d_logits;

        (dw, db)
    }

    /// SGD update: W -= lr · dW, b -= lr · db.
    ///
    /// For gradient descent with learning rate lr.
    /// Pass dW / batch_size to get the mean gradient across a batch.
    pub fn sgd_update(&mut self, dw: &[Vec<f64>], db: &[f64], lr: f64) {
        debug_assert_eq!(dw.len(), self.vocab_out);
        debug_assert_eq!(db.len(), self.vocab_out);
        for i in 0..self.vocab_out {
            for d in 0..self.dim {
                self.w[i][d] -= lr * dw[i][d];
            }
            self.b[i] -= lr * db[i];
        }
    }

    /// Adam optimizer update.
    ///
    /// Adam: adaptive learning rate with moment estimates.
    ///   m_t = β₁ · m_{t-1} + (1-β₁) · g_t       (first moment)
    ///   v_t = β₂ · v_{t-1} + (1-β₂) · g_t²      (second moment)
    ///   m̂_t = m_t / (1 - β₁^t)                   (bias correction)
    ///   v̂_t = v_t / (1 - β₂^t)                   (bias correction)
    ///   θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
    ///
    /// Pass the AdamState alongside the projection for stateful updates.
    pub fn adam_update(
        &mut self,
        dw: &[Vec<f64>],
        db: &[f64],
        state: &mut AdamState,
        lr: f64,
    ) {
        state.t += 1;
        let t = state.t as f64;
        let beta1 = state.beta1;
        let beta2 = state.beta2;
        let eps   = state.epsilon;

        // Bias correction factors
        let bc1 = 1.0 - beta1.powf(t);
        let bc2 = 1.0 - beta2.powf(t);

        for i in 0..self.vocab_out {
            for d in 0..self.dim {
                let g = dw[i][d];
                state.m_w[i][d] = beta1 * state.m_w[i][d] + (1.0 - beta1) * g;
                state.v_w[i][d] = beta2 * state.v_w[i][d] + (1.0 - beta2) * g * g;
                let m_hat = state.m_w[i][d] / bc1;
                let v_hat = state.v_w[i][d] / bc2;
                self.w[i][d] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
            let g = db[i];
            state.m_b[i] = beta1 * state.m_b[i] + (1.0 - beta1) * g;
            state.v_b[i] = beta2 * state.v_b[i] + (1.0 - beta2) * g * g;
            let m_hat = state.m_b[i] / bc1;
            let v_hat = state.v_b[i] / bc2;
            self.b[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
}

/// Adam optimizer state for one OutputProjection.
#[derive(Debug)]
pub struct AdamState {
    pub t: usize,
    pub beta1:   f64,
    pub beta2:   f64,
    pub epsilon: f64,
    pub m_w: Vec<Vec<f64>>,  // first moment for W
    pub v_w: Vec<Vec<f64>>,  // second moment for W
    pub m_b: Vec<f64>,       // first moment for b
    pub v_b: Vec<f64>,       // second moment for b
}

impl AdamState {
    pub fn new(dim: usize, vocab_out: usize) -> Self {
        Self {
            t:       0,
            beta1:   0.9,
            beta2:   0.999,
            epsilon: 1e-8,
            m_w: vec![vec![0.0; dim]; vocab_out],
            v_w: vec![vec![0.0; dim]; vocab_out],
            m_b: vec![0.0; vocab_out],
            v_b: vec![0.0; vocab_out],
        }
    }

    /// Adam with custom hyperparameters.
    pub fn with_params(dim: usize, vocab_out: usize, beta1: f64, beta2: f64, eps: f64) -> Self {
        Self { beta1, beta2, epsilon: eps, ..Self::new(dim, vocab_out) }
    }
}

/// Evaluate: return (mean_loss, accuracy, top5_accuracy).
///
/// accuracy: fraction of positions where argmax(logits) == target
/// top5_accuracy: fraction where target is in top-5 logits
pub fn evaluate(
    wo: &OutputProjection,
    representations: &[Vec<f64>],
    targets: &[usize],
) -> (f64, f64, f64) {
    let n = representations.len().min(targets.len());
    if n == 0 { return (0.0, 0.0, 0.0); }

    let mut total_loss = 0.0;
    let mut n_correct  = 0_usize;
    let mut n_top5     = 0_usize;

    for i in 0..n {
        let logits = wo.forward(&representations[i]);
        let (loss, _) = OutputProjection::cross_entropy_and_log_softmax(&logits, targets[i]);
        total_loss += loss;

        // Accuracy
        let pred = logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        if pred == targets[i] { n_correct += 1; }

        // Top-5 accuracy
        let mut indexed: Vec<(usize, f64)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        if indexed.iter().take(5).any(|(j, _)| *j == targets[i]) { n_top5 += 1; }
    }

    (
        total_loss / n as f64,
        n_correct as f64 / n as f64,
        n_top5 as f64 / n as f64,
    )
}

/// Full-batch gradient descent training step.
///
/// Returns the mean cross-entropy loss over the batch.
pub fn train_epoch(
    wo: &mut OutputProjection,
    representations: &[Vec<f64>],
    targets: &[usize],
    lr: f64,
) -> f64 {
    let n = representations.len().min(targets.len());
    if n == 0 { return 0.0; }

    let dim = wo.dim;
    let vocab_out = wo.vocab_out;
    let mut total_loss = 0.0;
    let mut acc_dw: Vec<Vec<f64>> = vec![vec![0.0; dim]; vocab_out];
    let mut acc_db: Vec<f64> = vec![0.0; vocab_out];

    for i in 0..n {
        let logits = wo.forward(&representations[i]);
        let (loss, log_softmax) = OutputProjection::cross_entropy_and_log_softmax(
            &logits, targets[i]);
        total_loss += loss;

        let (dw, db) = OutputProjection::backward(
            &representations[i], &log_softmax, targets[i], vocab_out);

        for j in 0..vocab_out {
            for d in 0..dim {
                acc_dw[j][d] += dw[j][d];
            }
            acc_db[j] += db[j];
        }
    }

    // Mean gradient
    let inv_n = 1.0 / n as f64;
    for j in 0..vocab_out {
        for d in 0..dim { acc_dw[j][d] *= inv_n; }
        acc_db[j] *= inv_n;
    }

    wo.sgd_update(&acc_dw, &acc_db, lr);
    total_loss / n as f64
}

/// Adam training step (full-batch).
pub fn train_epoch_adam(
    wo: &mut OutputProjection,
    state: &mut AdamState,
    representations: &[Vec<f64>],
    targets: &[usize],
    lr: f64,
) -> f64 {
    let n = representations.len().min(targets.len());
    if n == 0 { return 0.0; }

    let dim = wo.dim;
    let vocab_out = wo.vocab_out;
    let mut total_loss = 0.0;
    let mut acc_dw: Vec<Vec<f64>> = vec![vec![0.0; dim]; vocab_out];
    let mut acc_db: Vec<f64> = vec![0.0; vocab_out];

    for i in 0..n {
        let logits = wo.forward(&representations[i]);
        let (loss, log_softmax) = OutputProjection::cross_entropy_and_log_softmax(
            &logits, targets[i]);
        total_loss += loss;

        let (dw, db) = OutputProjection::backward(
            &representations[i], &log_softmax, targets[i], vocab_out);

        for j in 0..vocab_out {
            for d in 0..dim { acc_dw[j][d] += dw[j][d]; }
            acc_db[j] += db[j];
        }
    }

    // Mean gradient
    let inv_n = 1.0 / n as f64;
    for j in 0..vocab_out { for d in 0..dim { acc_dw[j][d] *= inv_n; } acc_db[j] *= inv_n; }

    wo.adam_update(&acc_dw, &acc_db, state, lr);
    total_loss / n as f64
}

// ── Internal ───────────────────────────────────────────────────────────────────

/// Deterministic LCG (Linear Congruential Generator).
/// Produces f64 in [0,1). Used for weight initialization.
/// NOT for cryptographic or statistical purposes.
fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005)
                  .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_wo() -> OutputProjection {
        OutputProjection::new_random(8, 16)
    }

    #[test]
    fn test_forward_output_dimension() {
        let wo = small_wo();
        let r = vec![0.5_f64; 8];
        let logits = wo.forward(&r);
        assert_eq!(logits.len(), 16, "forward output must equal vocab_out");
    }

    #[test]
    fn test_cross_entropy_loss_in_range() {
        let logits = vec![1.0_f64, 2.0, 0.5, -1.0];
        let (loss, _) = OutputProjection::cross_entropy_and_log_softmax(&logits, 1);
        assert!(loss >= 0.0, "cross-entropy loss must be non-negative");
        assert!(loss.is_finite(), "cross-entropy loss must be finite");
    }

    #[test]
    fn test_cross_entropy_perfect_prediction() {
        // If logits[target] >> all others, loss ≈ 0
        let mut logits = vec![-1000.0_f64; 8];
        logits[3] = 1000.0;
        let (loss, _) = OutputProjection::cross_entropy_and_log_softmax(&logits, 3);
        assert!(loss < 1e-6, "perfect prediction must produce near-zero loss, got {}", loss);
    }

    #[test]
    fn test_cross_entropy_uniform() {
        // Uniform logits for n classes → loss = log(n)
        let n = 10_usize;
        let logits = vec![0.0_f64; n];
        let (loss, _) = OutputProjection::cross_entropy_and_log_softmax(&logits, 0);
        let expected = (n as f64).ln();
        assert!((loss - expected).abs() < 1e-10,
            "uniform logits must produce loss=log(n)={:.6}, got {:.6}", expected, loss);
    }

    #[test]
    fn test_log_softmax_sums_to_zero() {
        // Σ exp(log_softmax[i]) should sum to 1 → Σ log_softmax is NOT 0,
        // but Σ exp(log_softmax) MUST sum to 1.
        let logits = vec![1.0_f64, 2.0, 0.5, -1.0, 3.0];
        let (_, log_softmax) = OutputProjection::cross_entropy_and_log_softmax(&logits, 0);
        let sum: f64 = log_softmax.iter().map(|&x| x.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-10,
            "softmax probabilities must sum to 1, sum={:.10}", sum);
    }

    #[test]
    fn test_gradient_shape() {
        let wo = small_wo();
        let r = vec![0.5_f64; 8];
        let logits = wo.forward(&r);
        let (_, log_softmax) = OutputProjection::cross_entropy_and_log_softmax(&logits, 2);
        let (dw, db) = OutputProjection::backward(&r, &log_softmax, 2, 16);
        assert_eq!(dw.len(), 16, "dW must have vocab_out rows");
        assert_eq!(dw[0].len(), 8, "dW rows must have dim columns");
        assert_eq!(db.len(), 16, "db must have vocab_out entries");
    }

    #[test]
    fn test_sgd_update_changes_weights() {
        let mut wo = small_wo();
        let w_before: f64 = wo.w[0][0];
        let dw = vec![vec![1.0_f64; wo.dim]; wo.vocab_out];
        let db = vec![1.0_f64; wo.vocab_out];
        wo.sgd_update(&dw, &db, 0.1);
        // w[0][0] should decrease by lr × dw[0][0] = 0.1
        assert!((wo.w[0][0] - (w_before - 0.1)).abs() < 1e-10,
            "SGD must subtract lr × gradient from weight");
    }

    #[test]
    fn test_loss_decreases_after_training() {
        // Sanity check: after training on a simple fixed dataset,
        // the loss should decrease.
        let dim = 8_usize;
        let vocab_out = 4_usize;
        let mut wo = OutputProjection::new_random(dim, vocab_out);
        // Simple dataset: one fixed representation, one fixed target
        let reps = vec![vec![1.0_f64; dim]];
        let targets = vec![2_usize];
        let loss_before = {
            let logits = wo.forward(&reps[0]);
            let (l, _) = OutputProjection::cross_entropy_and_log_softmax(&logits, targets[0]);
            l
        };
        // Train for 100 epochs
        for _ in 0..100 {
            train_epoch(&mut wo, &reps, &targets, 0.1);
        }
        let loss_after = {
            let logits = wo.forward(&reps[0]);
            let (l, _) = OutputProjection::cross_entropy_and_log_softmax(&logits, targets[0]);
            l
        };
        assert!(loss_after < loss_before,
            "loss must decrease after training: before={:.4} after={:.4}",
            loss_before, loss_after);
    }

    #[test]
    fn test_adam_state_dimensions() {
        let state = AdamState::new(8, 16);
        assert_eq!(state.m_w.len(), 16);
        assert_eq!(state.m_w[0].len(), 8);
        assert_eq!(state.v_w.len(), 16);
        assert_eq!(state.m_b.len(), 16);
        assert_eq!(state.v_b.len(), 16);
    }

    #[test]
    fn test_adam_loss_decreases() {
        let dim = 8_usize;
        let vocab_out = 4_usize;
        let mut wo = OutputProjection::new_random(dim, vocab_out);
        let mut adam = AdamState::new(dim, vocab_out);
        let reps = vec![vec![1.0_f64; dim]];
        let targets = vec![2_usize];
        let loss_before = {
            let logits = wo.forward(&reps[0]);
            let (l, _) = OutputProjection::cross_entropy_and_log_softmax(&logits, targets[0]);
            l
        };
        for _ in 0..50 {
            train_epoch_adam(&mut wo, &mut adam, &reps, &targets, 0.01);
        }
        let loss_after = {
            let logits = wo.forward(&reps[0]);
            let (l, _) = OutputProjection::cross_entropy_and_log_softmax(&logits, targets[0]);
            l
        };
        assert!(loss_after < loss_before,
            "Adam: loss must decrease after training: before={:.4} after={:.4}",
            loss_before, loss_after);
    }

    #[test]
    fn test_evaluate_returns_correct_shape() {
        let wo = small_wo();
        let reps = vec![vec![0.5_f64; 8]; 10];
        let targets: Vec<usize> = (0..10).map(|i| i % 16).collect();
        let (loss, acc, top5) = evaluate(&wo, &reps, &targets);
        assert!(loss.is_finite());
        assert!(acc >= 0.0 && acc <= 1.0);
        assert!(top5 >= acc, "top-5 accuracy must be >= top-1");
    }

    #[test]
    fn test_zero_wo_uniform_logits() {
        // Zero-initialized W+b → all logits = 0 → uniform distribution
        let wo = OutputProjection::new_zero(8, 10);
        let r = vec![0.5_f64; 8];
        let logits = wo.forward(&r);
        assert!(logits.iter().all(|&v| v == 0.0),
            "zero W+b must produce zero logits");
        let (loss, _) = OutputProjection::cross_entropy_and_log_softmax(&logits, 3);
        let expected = (10_f64).ln();
        assert!((loss - expected).abs() < 1e-10,
            "zero W+b must produce uniform cross-entropy = log(vocab_out)");
    }

    #[test]
    fn test_new_random_weights_nonzero() {
        let wo = small_wo();
        let all_zero = wo.w.iter().all(|row| row.iter().all(|&v| v == 0.0));
        assert!(!all_zero, "Xavier initialization must produce nonzero weights");
    }

    #[test]
    fn test_new_random_weights_bounded_by_xavier_scale() {
        let dim = 50_usize;
        let vocab_out = 135_usize;
        let wo = OutputProjection::new_random(dim, vocab_out);
        let scale = (6.0 / (dim + vocab_out) as f64).sqrt();
        let all_bounded = wo.w.iter().all(|row| row.iter().all(|&v| v.abs() <= scale));
        assert!(all_bounded,
            "Xavier init: all weights must be in [-scale, scale] where scale={:.4}", scale);
    }
}
