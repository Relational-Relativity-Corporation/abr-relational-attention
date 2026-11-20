# abr-relational-attention

**Metatron Dynamics, Inc.**
**Declaration V4.6 (2026-11-12) · README updated 2026-11-20**

ABR relational attention replaces softmax attention with a bounded relational binding. The binding is accumulated directly, pair by pair, as tokens stream in. No gradient flows through the attention weights.

---

## What this is

Standard transformer attention computes:

```
Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V        O(n²) per sequence
```

Its weights are learned by backpropagation over many training passes.

ABR relational attention accumulates a binding magnitude between token pairs instead:

```
ρ_acc(i,j) ← ρ_acc(i,j) + η · ρₙ(i,j) · (1 − ρ_acc(i,j))
```

- **ρₙ** comes from the ABR operators (A → B → R) applied to declared observables (the token representations through M).
- **Every value stays in [0, 1].**
- **Each update is local to one token pair**, made once, as the token arrives in a 64-token causal window.
- **Admission:** a pair takes part in inference when ρ_acc ≥ θ_min (0.05).
- **Inference** applies B and R over the admitted pairs in the window and returns the mean R-field at the current token.

The output projection W_O used in the language benchmarks is trained with Adam. The attention weights themselves are not.

---

## Kernel

Kernel V9 (`kernel_reference.md`, 2026-10-24), as declared in `docs/DECLARATION.md`.

- Primary observable: change.
- Relation: its invariant structure.
- Admissibility: traceable to a declared observable through M.

All quantities are bounded members of D := { x ∈ ℝⁿ | n < ∞, |x[i]| < ∞ ∀ i }.

---

## Repository structure

```
src/
  declaration.rs        M declaration, domain bounds, Locus, RelationalPair
  operators.rs          A, B, R operators and ρ
  accumulation.rs       accumulation rule and EdgeStore (bindings keyed by token pair)
  streaming.rs          streaming field: windowed accumulation and inference
  attention.rs          relational attention pass
  attention_layer.rs    attention layer (accumulate + infer per token)
  multi_head.rs         multi-head layer
  layer_stack.rs        stacked layers
  semantic.rs           semantic M: corpus PPMI vectors, GloVe loading, emphasis
  output_projection.rs  W_O output projection (Adam)
  bin/                  demonstrations and benchmarks (below)
docs/
  DECLARATION.md        full M declaration, open conditions, benchmark records
```

---

## Running

```bash
cargo test --release --lib --bins --tests
```

```bash
cargo run --release --bin scaling_benchmark
```

```bash
cargo run --release --bin held_out_benchmark -- --train /path/to/wikitext2/train.txt --test /path/to/wikitext2/test.txt
```

The other binaries are demonstrations: `architecture_demo`, `multi_head_demo`, `semantic_demo`, `glove_demo` (needs GloVe vectors), `stack_demo`, `wo_demo`, `streaming_demo`, `eta_calibration` and `demonstrate`.

141 unit and integration tests pass. One documentation test (`operators::RHO_BASE`) fails, because a doc comment's math is compiled as example code. It is a known issue, unrelated to the attention code.

---

## Measured results

### CPU scaling (`scaling_benchmark`, 2026-11-20)

The setup is synthetic: 2,000 token ids drawn from a Zipf(1.0) distribution, 50-dimensional seeded embeddings, θ_min = 0.05 and a 64-token window. Timings are wall-clock on one CPU core of a cloud container. Compare the rows with each other, not with other machines.

Per token, by sequence length (fresh field):

| n | ABR (µs/token) | Causal softmax, same embeddings (µs/token) |
| --- | --- | --- |
| 256 | 182 | 4 |
| 1,024 | 210 | 25 |
| 4,000 | 243 | 71 |

Long stream (the field persists across chunks of 4,000 tokens):

| Tokens streamed | Accumulation (µs/token) | Inference (µs/token) | Pairs stored |
| --- | --- | --- | --- |
| 4,000 | 7 | 253 | 92k |
| 16,000 | 11 | 281 | 272k |
| 32,000 | 7 | 296 | 452k |

**What this shows:**
- **Accumulation and inference cost per token do not grow with stream history.**
- **Softmax cost per token grows with sequence length.** On these numbers the two would meet at roughly 14,000 tokens.
- **Below that length, ABR is slower than plain softmax on CPU,** by 3× at 4,000 tokens and 45× at 256.
- **At θ_min = 0.05, about 60% of the pairs in a window are admitted,** so inference is not yet sparse.
- **The remaining cost is mostly the ~2,100 hash lookups per token,** one per pair in the 65-position window.

Two changes on 2026-11-20 brought inference from ~2,900–9,100 µs per token (growing with the field) to the figures above:
1. Window pairs are looked up directly, instead of scanning the whole store at each token.
2. R is evaluated only on the edges ending at the current token.

Outputs are unchanged: bit-for-bit for the first change, and within 10⁻¹⁰ (floating-point rounding) for the second, checked by `output_at_target_matches_full_evaluation`.

### Held-out language modelling (`held_out_benchmark`, 2026-11-12)

WikiText-2, 10,000 training tokens, 3,000 held-out test tokens. The baseline uses the same PPMI embeddings, the same W_O and the same training, with no ABR operators.

| Condition | Test perplexity | Top-1 accuracy |
| --- | --- | --- |
| Baseline (no ABR operators) | 100.4 | 38.4% |
| ABR | 237.4 | 15.6% |

The baseline is better in this configuration. 40.4% of the test tokens were outside the vocabulary, and the corpus is small. `docs/DECLARATION.md` records the protocol and the declared causes in full.

---

## Open conditions

As recorded in `docs/DECLARATION.md`:

| Condition | Status |
| --- | --- |
| OC-RA-1 θ_min calibration | Partially closed |
| OC-RA-2 η calibration | Partially closed |
| OC-RA-3 held-out validation | Closed (benchmark above) |
| OC-RA-4 bidirectional direction | Not declared (causal only) |
| OC-RA-5 window-restricted accumulation | Implemented in `streaming.rs` |
| OC-RA-6 χ max-selection rule | Closed |
| OC-RA-7 saturating form χ/(χ₀+χ) | Closed |
| OC-RA-8 higher-level connection scopes | Open |
| OC-RA-9 ι[v] / ε[e] through emphasis | Closed |

---

## License

Apache 2.0 — Metatron Dynamics, Inc.
