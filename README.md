# abr-relational-attention

**Metatron Dynamics, Inc.**
**V7 — 2026-08-21**

ABR relational attention: replacing quadratic softmax attention with a bounded, directly-accumulated relational binding magnitude — no backpropagation required.

---

## What This Is

Standard transformer attention computes:

```
Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V     [O(n²) cost]
```

The weights are gradient-derived approximations of relational significance, discovered through millions of forward-backward training cycles.

This repository implements the ABR alternative:

```
ρ_acc(i,j) ← ρ_acc(i,j) + η · ρₙ(i,j) · (1 - ρ_acc(i,j))
```

`ρ_acc(i,j)` is the accumulated binding magnitude between source locus i and target locus j, computed directly from declared observables through the ABR operators (A → B → R). It replaces the softmax weight. It requires no backpropagation and no global loss function.

---

## Kernel

V7 — `operators.rs` (Metatron Dynamics, Inc.)

Primary observable: change.
Relation: its invariant structure.
Admissibility: traceable to a declared observable through M.

All quantities in this repository are bounded members of D := { x ∈ ℝⁿ | n < ∞, |x[i]| < ∞ ∀ i }.

---

## Repository Structure

```
src/
  declaration.rs    — M declaration, domain bounds, Locus, RelationalPair
  operators.rs      — A, B, R operators and ρ computation
  accumulation.rs   — Accumulation rule and EdgeStore
  attention.rs      — Relational attention pass and training accumulation
  lib.rs            — Crate root
  main.rs           — Phase 0 demonstration

docs/
  DECLARATION.md    — Full M declaration V1.0
```

---

## Running

```bash
cargo test          # all tests must pass before Phase 1
cargo run           # Phase 0 demonstration
```

---

## Phase 0 Scope

- Accumulation rule implemented and bounded
- EdgeStore: O(n · k) admitted neighbourhood
- Relational attention pass replacing softmax
- Causal direction only (OC-RA-4)
- Synthetic sequence demonstration

## Open Conditions

- **OC-RA-1** — θ_min calibration (Phase 1)
- **OC-RA-2** — η calibration (Phase 1)
- **OC-RA-3** — Generalisation — held-out validation pass (Phase 1)
- **OC-RA-4** — Bidirectional — requires independent declaration

---

## License

Apache 2.0 — Metatron Dynamics, Inc.
