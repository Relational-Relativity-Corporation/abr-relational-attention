# ABR Relational Attention -- M Declaration V3.0
**Metatron Dynamics, Inc. -- 2026-08-21**
**Kernel: operators.rs V7**
**Revision: V2 closed -- canonical A, B, R, rho implemented**

---

## Domain

D := { x in R^n | n < inf, |x[i]| < inf for all i }

All quantities are bounded members of D.

---

## The Language Relational Field

Language is a physical relational field. It evolved as a compression of
physical phonemic structure -- acoustic production events condensed into
symbolic units that carry relational structure because the underlying
physical relationships were physically real before the symbols existed.

Tokens are the declared loci of the language relational field. The
observable at each locus is the phonemic-semantic state of that token.
M declares the embedding vector as the measurement of that state in D.

---

## Observable Mapping

M : O -> D

O = token positions in a declared input sequence.
Each M(ok) is the phonemic-semantic state of the token at position k,
represented as a finite-dimensional embedding vector (bounded by D).

---

## Relational Evolution Direction (V1 closed)

The admissible direction is relational coherence progression -- an
intrinsic physical property of language in which each token oj is
relationally dependent on its predecessors for coherence.

Sequence position is M's declared measurement of relational coherence
progression. It is not a numerical primitive -- it is the observable
trace of the physical relational dependency structure.

Reverse direction (j -> i, j > i) is inadmissible: oi does not depend
on oj for coherence.

---

## Canonical V7 ABR Operators in the Language Domain (V2 closed)

### A -- Directed Difference

    A(oi, oj) = M(oi) - M(oj)

The directed contrast between two phonemic-semantic states. Encodes what
the source carries that is absent at the target, in the declared direction
of relational coherence progression.

Admissibility: difference of two declared M-measurements. Difference of
bounded vectors is bounded -- stays in D.

### B -- Successor Accumulation over Admitted Successors

    B(g)[e] = g[e] + sum_{f in succ_admitted(e)} g[f]

Accumulates the contrast vector over the declared admitted successor set.

succ_admitted(source) = { j' | rho_acc(token_source, token_j') >= theta_min }

Incoherent successors (rho_acc < theta_min) are EXCLUDED.

Origin declaration: departure from coherence is loss of relevance, not
loss of volume. A token that is syntactically present but semantically
incoherent with the source carries no declared relational information
with respect to that source. Including it would accumulate noise, not
signal. The canonical B over the language field therefore accumulates
only over admitted successors.

In cold start (no admitted edges), B returns A unchanged -- the directed
contrast with no accumulated successor context.

Admissibility: sum of bounded contrast vectors is bounded.

### rho -- Binding Magnitude

    rho(i,j) = ||B(A(oi,oj))|| / (||B(A(oi,oj))|| + eps)

The norm of the accumulated directed contrast, normalised to [0,1].
Measures the magnitude of relational structure the source carries toward
the target, accumulated over its admitted successor set.

eps: declared stability constant (domain boundedness requirement).
Result is in [0,1]: norm/(norm+eps) < 1 always; >= 0 always (norm >= 0).

### R -- Antisymmetric Resolution

    R(B(A))[j] = B(A(oi,oj)) - B(A(oj,oi))

The antisymmetric component of the accumulated directed contrast.
Extracts the irreducibly directional part of the relational structure
at the target locus.

When the relation is symmetric, R = 0.
When directional (as causal language relations must be), R encodes
the net relational flow from source to target.

---

## Declared Observables

| Observable           | Symbol            | Traceable to M                       |
|----------------------|-------------------|--------------------------------------|
| Phonemic-semantic    | M(ok)             | embedding vector at token k          |
| Token identity       | token_id          | vocabulary identifier                |
| Directed contrast    | A(oi,oj)          | M(oi) - M(oj)                       |
| Successor set        | succ_admitted(e)  | tokens with rho_acc >= theta_min     |
| Accumulated contrast | B(A)              | A + sum over admitted successors     |
| Binding magnitude    | rho(i,j)          | ||B(A)|| normalised by eps           |
| Antisymm. resolution | R                 | B_forward - B_reverse                |
| Accum. binding       | rho_acc(ti, tj)   | rho integrated over exposures        |

---

## Declared Parameters (Origin)

| Parameter   | Symbol      | Value | Status                       |
|-------------|-------------|-------|------------------------------|
| Learning    | eta         | 0.1   | OC-RA-2 partially closed     |
| Threshold   | theta_min   | 0.05  | OC-RA-1 partially closed     |
| Stability   | eps         | 1e-8  | derived from domain bounds   |

---

## Accumulation Rule

    rho_acc(ti,tj) <- rho_acc(ti,tj) + eta * rho_n(i,j) * (1 - rho_acc(ti,tj))

rho_n is now computed from canonical B(A): rho_n = ||B(A(oi,oj))|| / (||...|| + eps)

Keyed by token-identity pair (ti, tj). Persists across sequences.
Stability: (1 - rho_acc) term derived from V7 domain boundedness.

---

## Complexity

Accumulation: O(n^2) Phase 0 -- all causal pairs evaluated (OC-RA-5).
Inference:    O(n*k) -- k = mean admitted neighbourhood size.

---

## Open Conditions

- OC-RA-1: theta_min calibration (Phase 1)
- OC-RA-2: eta calibration (Phase 1)
- OC-RA-3: generalisation -- held-out validation (Phase 1)
- OC-RA-4: bidirectional direction -- not declared
- OC-RA-5: window-restricted O(n*k) accumulation (Phase 1)

---

## Verifier Criterion

Can every mathematical operation be traced continuously back to declared
observables through declared operators? Every quantity in this repository
satisfies this criterion under Declaration V3.0.
