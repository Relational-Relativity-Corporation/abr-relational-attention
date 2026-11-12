# ABR Relational Attention -- M Declaration V4.0
**Metatron Dynamics, Inc. -- 2026-11-11**
**Kernel: V9 (kernel_reference.md 2026-10-24)**
**Revision: V3.0 ρ and R replaced with canonical V9 forms; ρ_base and χ₀ M-declared**

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

## Relational Evolution Direction

The admissible direction is relational coherence progression -- an
intrinsic physical property of language in which each token oj is
relationally dependent on its predecessors for coherence.

Sequence position is M's declared measurement of relational coherence
progression. It is not a numerical primitive -- it is the observable
trace of the physical relational dependency structure.

Reverse direction (j -> i, j > i) is inadmissible: oi does not depend
on oj for coherence.

---

## Declared Topology

The edge set over which B and R act is established before the operators
are applied. An edge (si, ti) is admitted when:

    rho_acc(token_si, token_ti) >= theta_min

rho_acc accumulates across exposures to (si, ti) pairs via the
Accumulation Rule below. theta_min is the declared admission threshold.

This topology declaration corresponds to DeclaredGraph in operators.rs.
B and R act on the declared graph without performing additional selection.

Incoherent pairs (rho_acc < theta_min) are excluded from the topology.
Cold start: no admitted edges; B returns A unchanged; R returns B unchanged.

---

## Canonical V9 ABR Operators (kernel_reference.md §11)

### A -- Directed Difference

    A[e] = M(src(e)) - M(tgt(e))

The directed contrast between two phonemic-semantic states. Encodes what
the source carries that is absent at the target.

Admissibility: difference of two declared M-measurements. Bounded vectors
in D; their difference stays in D.

### ρ -- Per-Node Binding Magnitude (V9 canonical)

    ρ[i] = ρ_base · χ[i] / (χ₀ + χ[i])
    χ[i] = max |A[e]| over all edges incident to node i (source or target)

ρ is a per-node scalar, not per-edge. It is computed from A, not from B.
The edge form is ρ[e] = ρ[src(e)].

χ max-selection rule (Origin declaration, 2026-11-11; OC-RA-6 closed):
χ[i] is a bracketing declaration. The max establishes the numerical ceiling
of χ[i] at each node so that ρ[i] reflects the peak declared local asymmetry
at i. Mean alternatives compress dynamic range and are degree-sensitive;
sum alternatives scale with degree. Max brackets the range without degree
dependence. Means are avoided whenever possible throughout this framework:
a mean discards the structure of the distribution; a selection preserves it.

ρ_base and χ₀ are M-declared parameters; see Declared Parameters.

Result: ρ[i] ∈ [0, ρ_base). Zero if no edges incident at i. Saturates
below ρ_base because χ/(χ₀+χ) < 1 for all finite χ.

V3.0 form rho(i,j) = ||B(A)||/(||B||+eps) is superseded. The canonical
V9 ρ is computed from A before B; it does not depend on B.

### B -- Successor Accumulation

    B[e] = A[e] + Σ_{f ∈ succ(e)} A[f]

    succ(e = (sp, tp)) = { f : src(f) == tp }
    (edges starting from the target of e; immediate successors only)

Accumulates the contrast vector of e with the contrast vectors of edges
that depart from e's target. Terminal edges (no successors) have B[e] = A[e].

B does not perform topology selection. The admitted edge set (declared
topology, see above) determines which f are present in succ(e). B then
acts on that structure.

### R -- Directed Imbalance Resolution (V9 canonical)

    R[e] = B[e] + ρ[src(e)] · (Σ_{succ(e)} B[f] − Σ_{pred(e)} B[p])

    succ(e = (sp, tp)) = { f : src(f) == tp }   (edges from tp)
    pred(e = (sp, tp)) = { p : tgt(p) == sp }   (edges to sp)

R adds to B[e] the rho-modulated difference between the forward (successor)
and backward (predecessor) B-sums at e. This extracts the net directed
relational imbalance at e's position in the declared graph.

ρ[src(e)] is the per-node scalar at the source of e, computed from A.
When succ and pred are both empty, the imbalance term is zero; R[e] = B[e].
Antisymmetric expression (non-zero imbalance) is admissible and preserved.

V3.0 form R = B(A(oi,oj)) - B(A(oj,oi)) (antisymmetric pair evaluation)
is superseded. V9 R computes directed imbalance from succ/pred adjacency
of the declared graph; it does not require fabricating a reverse edge.

---

## W_O -- Output Projection (Learned Parameter)

    W_O ∈ R^{vocab × dim}

The output projection maps the R-field at each token position to vocabulary
logits for next-token prediction. It is the only gradient-descent-updated
parameter in the system.

Declared training protocol:
- Optimizer: Adam (β₁=0.9, β₂=0.999, lr=1e-3)
- Loss: cross-entropy over vocabulary
- Input: R-field vector at each position (last layer, dim=50)
- Target: next-token identity

The relational field (ρ_acc in EdgeStore) is not updated by backprop.
ABR decouples representation accumulation (topology formation) from
prediction learning (W_O gradient descent).

---

## Declared Observables

| Observable           | Symbol            | Traceable to M                            |
|----------------------|-------------------|-------------------------------------------|
| Phonemic-semantic    | M(ok)             | embedding vector at token k               |
| Token identity       | token_id          | vocabulary identifier                     |
| Directed contrast    | A[e]              | M(src(e)) - M(tgt(e))                    |
| Per-node asymmetry   | χ[i]              | max |A[e]| over edges incident at i        |
| Per-node binding     | ρ[i]              | ρ_base · χ[i] / (χ₀ + χ[i])             |
| Admitted edge set    | DeclaredGraph     | pairs with rho_acc >= theta_min           |
| Successor set        | succ(e)           | { f : src(f) == tgt(e) } in DeclaredGraph |
| Predecessor set      | pred(e)           | { p : tgt(p) == src(e) } in DeclaredGraph |
| Accumulated contrast | B[e]              | A[e] + Σ_{succ(e)} A[f]                  |
| Directed imbalance   | R[e]              | B[e] + ρ[src(e)] · (Σ_succ B − Σ_pred B) |
| Accum. binding       | rho_acc(ti, tj)   | rho_n integrated over exposures           |
| Vocabulary logits    | W_O R[e]          | linear projection of R-field              |

---

## Declared Parameters (M Declaration)

| Parameter   | Symbol      | Value   | M Declaration                                                                                              |
|-------------|-------------|---------|-------------------------------------------------------------------------------------------------------------|
| Learning    | eta         | 0.1     | OC-RA-2 partially closed                                                                                    |
| Threshold   | theta_min   | 0.05    | OC-RA-1 partially closed                                                                                    |
| Ceiling     | ρ_base      | 1.0     | Language domain. V9 general value: 0.5. Declared 1.0: GloVe-50d chi values span [0, ~2.0]; ρ_base=0.5 saturates at 0.5 × chi/(1+chi) → max 0.5, insufficient dynamic range. ρ_base=1.0 preserves chi-proportional scaling (ρ ∈ [0,1)). Reason recorded per V9 §9. |
| Reference   | χ₀          | 1       | Language domain. V9 general value: 1 unit of χ. Declared as 1 unit of GloVe-50d embedding distance. χ is dimensionless in normalized GloVe space; χ₀=1 is one unit of that space. |
| Stability   | eps         | 1e-8    | Implementation tolerance; not a system property. Retained in rho_acc update only. |
| W_O lr      | Adam lr     | 1e-3    | Standard Adam learning rate for vocabulary projection.                                                       |
| W_O β₁      | 0.9         | 0.9     | Standard Adam first moment decay.                                                                            |
| W_O β₂      | 0.999       | 0.999   | Standard Adam second moment decay.                                                                           |

---

## Accumulation Rule

    rho_acc(ti, tj) <- rho_acc(ti, tj) + eta * rho_n(i,j) * (1 - rho_acc(ti, tj))

    rho_n(i, j) = ρ_base · χ_n / (χ₀ + χ_n)
    χ_n = max |A[e]| over edges incident at node i in the current sequence context

Keyed by token-identity pair (ti, tj). Persists across sequences.
Stability: (1 - rho_acc) bounds rho_acc in [0, 1). eta in (0, 1).

V3.0 used rho_n = ||B(A)||/(||B||+eps). Replaced with canonical V9 ρ
form (chi-based, computed from A before B).

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
- OC-RA-6: χ max-selection rule -- CLOSED (Origin declaration, 2026-11-11)
- OC-RA-7: saturating form χ/(χ₀+χ) -- OPEN per V9 §11.2 (Origin declaration required)

---

## V3.0 → V4.0 Change Record

| Item         | V3.0 form                                   | V4.0 canonical V9 form                                              |
|--------------|---------------------------------------------|----------------------------------------------------------------------|
| ρ formula    | \|\|B(A)\|\| / (\|\|B(A)\|\| + eps)        | ρ_base · χ[i] / (χ₀ + χ[i]), χ[i] = max\|A[e]\| at node i         |
| R formula    | B(A(oi,oj)) − B(A(oj,oi))                  | B[e] + ρ[src(e)] · (Σ_{succ(e)} B − Σ_{pred(e)} B)                |
| succ/pred    | succ_admitted filtered by rho_acc           | topology declared separately; succ(e) = {f : src(f)==tgt(e)} in DeclaredGraph |
| ρ_base       | implicit (1.0 in code, undeclared)          | M-declared with reason (V9 §9)                                      |
| χ₀           | implicit (1.0 in code, undeclared)          | M-declared (V9 §9)                                                  |
| rho_n (accum)| \|\|B(A)\|\| / (\|\|B\|\| + eps)           | ρ_base · χ_n / (χ₀ + χ_n) from A                                   |
| W_O          | not declared                                | declared as only gradient-updated parameter                          |

---

## Verifier Criterion

Can every mathematical operation be traced continuously back to declared
observables through declared operators? Every quantity in this repository
satisfies this criterion under Declaration V4.0 and Kernel V9.

Kernel V9 scope: this repository is application construction scope
(kernel_reference.md §4). V9 provenance machinery (IRRS, RPPI,
ObservableRelationDeclaration, ProvenancedOutput) governs abr-kernel and
abr-irrs; it does not apply here.
