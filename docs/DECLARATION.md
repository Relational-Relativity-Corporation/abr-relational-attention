# ABR Relational Attention -- M Declaration V4.4
**Metatron Dynamics, Inc. -- 2026-11-12**
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

## Language Region — Observable Variable Set (Pre-Topological Declaration)

**What M can admissibly map into D at the language region, before any topology is declared.**

### Human Reference Frame

The language region is bounded by the human perceptual and cognitive frame — to what
human perception and cognition can declare through M into D. Change and relation are
invariant within that boundary. No claim is made beyond D.

Language is the primary bounded domain of human symbolic communication. Its
relational structure is admissible as an observable because it evolved as a compression
of physical phonemic structure — acoustic production events that were physically real
before the symbols existed. The symbolic compression preserves relational structure
as a downstream projection of that physical origin.

### Degrees of Freedom — Nested Confinement Hierarchy

Origin declaration, 2026-11-12.

The degrees of freedom in the language region are nested and confining, not additive.
Each level removes degrees of freedom from the level below:

    character count
      --> word structure         (not all character sequences form words)
        --> sentence structure   (not all word sequences form sentences)
          --> semantic structure (not all sentence sequences carry coherent meaning)
              over the course of: words --> phrases --> sentences --> paragraphs --> volumes

Every locus declaration above the character level collapses interior relational structure
from the level below. This is consistent with the kernel's declared locus boundary rule:
at each scale transition, sub-locus relational structure is collapsed into the declared
locus variable set for that level.

### M at the Word/Token Level (SemanticM — PPMI)

The declared M in this application operates at the word/token level.
SemanticM::from_corpus constructs PPMI co-occurrence embeddings from the declared
corpus text.

**What M preserves:**
- Directional co-occurrence relationships within the declared window
- Token-identity distinction (each token type has a distinct embedding)
- Statistical trace of sentence-level semantic proximity at window scale
- Relative frequency information above min_count

**What M discards:**
- Character-level relational structure (morphological decomposition)
- Word order within the co-occurrence window
- Sentence boundary signals below window scale
- Cross-sentence structure beyond the declared window
- Tokens below min_count (collapsed to unk_id)

The 50 declared dimensions (PPMI_DIM = 50) are a projection of the word-level degrees
of freedom through PPMI dimensionality reduction — a downstream projection of
semantic-level containment and sentence-level proximity.

Embeddings are min-max normalized to [0, 1]^dim before operator application. This is
part of M. The normalization discards absolute scale; it preserves relative token-type
distinctions within the declared vocabulary.

### Edge Provenance Declaration

Every edge (ti, tj) in EdgeStore carries **Continuation** provenance: within-sequence
causal continuation established by sequence position (token at position i precedes
token at position j in the declared sequence). Open boundary — no wraparound.
Direction is admissible: relational coherence progression (source precedes target).

### Open Conditions at This Level

**ι[v] at the language region: OPEN.**
The inertial response of a declared token to an applied relational contrast is not yet
declared through M at the language region. What resists change at the word, sentence,
or semantic level has not been declared. Under the declared-information rule,
ε[e] = |Δ(ι)[e]| · 𝟙[e] is not evaluated at language edges — not
evaluated-and-equal-to-zero; not declared through M.

**Connection / entanglement at the language region:**

Origin declaration, 2026-11-12.

Connection in the language region is nested and scope-bounded, parallel to the
degrees of freedom hierarchy:

    characters are connected within words
      words are connected within sentences
        sentences are connected within paragraphs
          paragraphs are connected within volumes

At each level, the connection scope is bounded by the containing unit at the level
above. A character's connections are scoped to the word that contains it. A word's
connections are scoped to the sentence that contains it. A sentence's connections
are scoped to the paragraph that contains it. A paragraph's connections are scoped
to the volume that contains it.

This is dual to the degrees of freedom hierarchy: the DOF hierarchy states what
relational structure is possible at each level; the connection hierarchy states
the scope within which those relations are admitted at each level.

**M at word/token level — connection scope declared:**

SemanticM (PPMI, window=5) captures word-level connections within a local
co-occurrence window. The window approximates sentence-proximity scope — words
within five positions carry co-occurrence signal. This is a declared approximation
of word-within-sentence connection scope.

What the current M captures from the declared connection hierarchy:
- Word-level connections within sentence-proximity scope (via window=5): **captured (approximation)**

What the declared connection hierarchy admits but M does not yet declare:
- Character-level connections within words: not encoded (M operates at word level)
- Sentence-level connections within paragraphs: not encoded
- Paragraph-level connections within volumes: not encoded

The current EdgeStore (token-identity pairs, Continuation provenance) operates
at the word/token level. The declared connection hierarchy states that higher-level
scopes are admissible but not yet declared through M at this level.

**OC-RA-8: Higher-level connection scopes — OPEN.**
Sentence, paragraph, and volume connection structures are declared admissible
per the connection hierarchy. Not yet declared through M.

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

Domain context: GloVe-50d and PPMI-derived (SemanticM::from_corpus) embeddings
are min-max normalized to [0,1]^dim, so chi = max|A[e]| ∈ [0, 1]. The
saturating form achieves its midpoint at chi = χ₀ = 1 (the domain ceiling),
making χ₀=1 commensurate with the declared chi range.

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

### wo_demo Scope Declaration (exploratory; 2026-11-11)

wo_demo (src/bin/wo_demo.rs) is an exploratory training-fit measurement.
The following conditions are declared:

**Training fit, not generalization.** wo_demo trains and evaluates on the
same 233-token Darwin paragraph. Reported perplexity and accuracy are
training-fit metrics. They do not establish held-out predictive improvement.
No generalization claim is made from these results.

**Replay-based accumulation (declared exploratory deviation).** wo_demo
runs N_MATURE=3 passes over the same paragraph during R-field construction.
The streaming declaration prohibits replay-based accumulation: rho_acc is
intended to accumulate over distinct exposures, not repeated passes through
the same sequence. The repeated passes inflate rho_acc beyond single-pass
levels. This is a declared exploratory deviation. Results are valid as
training-fit measurements under these declared conditions; they do not
represent the streaming accumulation protocol.

**Random baseline is a position-memorization baseline.** The random baseline
([A] in wo_demo) assigns a distinct LCG-seeded 50-dim vector to each token
*position*, not each token *type*. W_O therefore learns a position-to-next-token
mapping over the fixed 233-position sequence — equivalent to memorizing the
sequence by position index. It is not a frequency-only floor. The apparent
dominance of the random baseline at 500 epochs reflects this memorization
property. The meaningful comparison is ABR vs GloVe: both use token-type
representations and neither enables position memorization.

**Later layers receive whole-paragraph context (declared exploratory condition).**
Each layer's M is derived from the layer below via contextualize_m
(layer_stack.rs): M_{L+1}[t] = mean R-field over all positions of token type t
in the full paragraph. This aggregation is not causal — it uses every position
of token t regardless of sequence order. Later layers therefore receive
whole-paragraph context. This is a declared property of the layer stack in its
current exploratory form; a streaming causal M-derivation is not implemented.

**ABR vs GloVe comparison (training fit).** ABR R-field achieves lower
training loss than GloVe (pplx 10.7 vs 57.6, 5.37×) on the Darwin paragraph
under the declared exploratory conditions. This measures the R-field's
training-fit advantage over static token-type embeddings. It does not
establish held-out generalization.

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
| Ceiling     | ρ_base      | 1.0     | Language domain. V9 general value: 0.5. Declared 1.0: GloVe-50d and PPMI-derived (SemanticM) embeddings are min-max normalized to [0,1]^dim; chi = max\|A[e]\| ∈ [0,1]. With χ₀=1: ρ_base=0.5 → ρ_max=0.25; ρ_base=1.0 → ρ_max=0.50. Declaring 1.0 uses the available chi range proportionally (doubles the ceiling). No claim that 0.25 is insufficient. Reason recorded per V9 §9. |
| Reference   | χ₀          | 1       | Language domain. V9 general value: 1 unit of χ. Declared as 1 unit of normalized embedding distance ([0,1]^dim space; applies to GloVe-50d and PPMI-derived SemanticM). χ is dimensionless in this normalized space; χ₀=1 is one unit of that space. |
| Stability   | eps         | 1e-8    | Implementation tolerance; not a system property. Retained in rho_acc update only. |
| W_O lr      | Adam lr     | 1e-3    | Standard Adam learning rate for vocabulary projection.                                                       |
| W_O β₁      | 0.9         | 0.9     | Standard Adam first moment decay.                                                                            |
| W_O β₂      | 0.999       | 0.999   | Standard Adam second moment decay.                                                                           |

---

## Accumulation Rule

    rho_acc(ti, tj) <- rho_acc(ti, tj) + eta * rho_n(i,j) * (1 - rho_acc(ti, tj))

    rho_n(i, j) = ρ_base · χ_n / (χ₀ + χ_n)
    χ_n = max component |A(ti, tj)| — the directed contrast of the specific pair only

rho_n is computed from the A vector of the pair being accumulated, not from
the per-node max over all incident edges. Per-pair chi captures the
contrast of that specific token relationship; it differs from the per-node
chi used in the R operator (which takes max over all edges incident at a node
to scale the antisymmetric term). Both are chi-based; they are evaluated at
different granularities for different purposes.

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
- OC-RA-3: generalisation -- held-out validation (Phase 1) -- **CLOSED** (OC-RA-3 benchmark, 2026-11-12; see below)
- OC-RA-4: bidirectional direction -- not declared
- OC-RA-5: window-restricted O(n*k) accumulation (Phase 1)
- OC-RA-6: χ max-selection rule -- CLOSED (Origin declaration, 2026-11-11)
- OC-RA-7: saturating form χ/(χ₀+χ) -- **CLOSED** (Origin declaration, 2026-11-12)
- OC-RA-8: higher-level connection scopes (sentence, paragraph, volume) -- OPEN (declared admissible, not yet declared through M; Origin declaration 2026-11-12)

---

## OC-RA-3 Benchmark Declaration (Held-Out Validation)

**Status: CLOSED — result obtained 2026-11-12**
**Binary: src/bin/held_out_benchmark.rs**

### Protocol

| Parameter       | Value                                                           |
|-----------------|-----------------------------------------------------------------|
| Corpus          | WikiText-2 (pytorch/examples)                                   |
| Train tokens    | first 10,000 tokens of train.txt                               |
| Test tokens     | first 3,000 tokens of test.txt (held out; W_O frozen)          |
| M               | PPMI corpus-derived (SemanticM::from_corpus)                   |
| M dim / window  | dim=50, window=5, min_count=5                                   |
| M vocabulary    | 407 words (built from first ~80K chars of train only)           |
| OOV rate (test) | 40.4% (test tokens not in train vocab → unk_id)                 |
| Accumulation    | single-pass streaming, no replay (OC-RA-5 compliant)           |
| Chunk size      | 4000 tokens (MAX_SEQUENCE_LENGTH=4096 hard limit; ~0.025% pairs missed at boundaries) |
| ETA / THETA_MIN | 0.1 / 0.05                                                      |
| Coherence W     | 64                                                              |
| W_O training    | Adam(lr=0.001), 150 epochs, cross-entropy                       |
| Baseline        | Direct M — W_O on raw PPMI M(token), no ABR operators (same M, same W_O init, same training protocol) |

### Results

| Condition    | Loss   | Perplexity | Top-1 Acc | Top-5 Acc |
|--------------|--------|------------|-----------|-----------|
| ABR train    | 5.6973 | 298.1      | 10.6%     | 25.2%     |
| ABR test     | 5.4696 | 237.4      | 15.6%     | 34.3%     |
| Direct train | 4.9452 | 140.5      | 27.9%     | 50.0%     |
| Direct test  | 4.6088 | 100.4      | 38.4%     | 56.4%     |

EdgeStore after 10,000 train tokens: 49,218 total edges; 29,074 admitted at theta=0.05.

Generalisation gap (test − train loss): ABR −0.2277, Direct −0.3364 (underfitting, not overfitting).

**ABR advantage over Direct M (positive = ABR better):**
  Test  loss delta: −0.8608 (Direct M is better on held-out).
  Train loss delta: −0.7521 (Direct M is better on train).

### Declared Finding

**Direct M test perplexity (100.4) < ABR test perplexity (237.4) on this configuration.**

The result is accurate for the stated conditions. It does not refute ABR — it records the
performance boundary of this parameter regime.

**Declared causes of ABR underperformance in this configuration:**

1. **High OOV rate (40.4%).** M vocabulary was built from first ~80K characters of train
   only (a subset to control W_O matrix size). Test tokens outside that vocabulary receive
   the unk embedding (0.5 in all dims). With 40.4% of test tokens collapsed to a single
   vector, the R-field loses most relational structure on test. Direct M is equally affected
   by OOV, but its representations are not modified by ABR operators, so the uniform unk
   embedding passes through cleanly. ABR operators amplify the uniform signal through
   the succ/pred structure and may distort it.

2. **Small corpus (10,000 train tokens).** Insufficient for PPMI co-occurrence statistics
   to reflect meaningful relational geometry. R-field accumulation depends on observed
   pair exposures; with 10K tokens and a 407-word vocabulary the EdgeStore is sparse
   relative to what PPMI-grounded ABR requires.

3. **R-field scaling at initialization.** At epoch 0, ABR perplexity was 5.1×10^12 vs
   Direct M 412 — a ratio of ~12 billion. This indicates the R-field vectors at initialization
   have a numerical range W_O cannot adapt to quickly. The ABR R-field is a sum of directed
   contrasts and their imbalance residuals; without warm-up scheduling its magnitude is
   poorly matched to W_O's Xavier-initialized weight scale. Direct M passes normalized [0,1]
   vectors directly to W_O, which Xavier init handles well.

**Generalisation claim bounded to declared conditions.** Larger corpus, lower OOV rate,
and R-field magnitude scheduling are expected to shift the comparison. No claim that
ABR cannot generalise — only that it did not outperform Direct M under these stated conditions.

**Closed-world M declaration:** PPMI vocabulary built from train text only. Test tokens
outside train vocab receive unk treatment. This is correct protocol (no test leakage into M).
The OOV rate is a property of the small M corpus, not a protocol violation.

---

## V4.3 → V4.4 Change Record

| Item | V4.3 | V4.4 |
|------|-------|-------|
| Connection / entanglement at language region | OPEN — placeholder | CLOSED declaration (Origin, 2026-11-12): nested scope-bounded connection hierarchy — characters within words, words within sentences, sentences within paragraphs, paragraphs within volumes. Dual to DOF hierarchy. |
| M connection scope | Not stated | Stated: window=5 captures word-level connections at sentence-proximity scope (approximation). Character, sentence, paragraph, volume scopes not yet declared through M. |
| OC-RA-8 | Not listed | OPEN: higher-level connection scopes (sentence, paragraph, volume) declared admissible, not yet declared through M. |

---

## V4.2 → V4.3 Change Record

| Item | V4.2 | V4.3 |
|------|-------|-------|
| Language Region Observable Variable Set | Not declared | Added (Origin declaration, 2026-11-12): pre-topological declaration of what M can admissibly map into D at the language region. |
| Human reference frame bounding | Implicit in Domain section | Explicitly stated: bounded by the human perceptual and cognitive frame; language admissible as observable via physical phonemic origin. |
| Degrees of freedom | Not declared | Nested confinement hierarchy declared: character → word → sentence → semantic structure over words → phrases → sentences → paragraphs → volumes. Each level removes degrees of freedom from the level below. |
| M preserves / discards | Not stated | Stated for SemanticM (PPMI): preserves directional co-occurrence, token-identity distinction, semantic proximity at window scale, relative frequency above min_count. Discards character-level structure, word order within window, sub-window sentence signals, cross-window structure, tokens below min_count. Normalization discards absolute scale. |
| Edge provenance | Not declared | Declared: Continuation (within-sequence causal continuation; open boundary; no wraparound). |
| ι[v] at language region | Not addressed | OPEN — declared not yet declared through M; ε[e] not evaluated (not equal to zero — not declared). |
| Connection / entanglement | Not addressed | OPEN — declared as next variable to be declared. |

---

## V4.1 → V4.2 Change Record

| Item | V4.1 | V4.2 |
|------|-------|-------|
| OC-RA-3 | OPEN — held-out validation required | CLOSED — OC-RA-3 benchmark 2026-11-12: Direct M (pplx 100.4) outperforms ABR (pplx 237.4) on WikiText-2 10K-token / 407-word PPMI configuration. Result declared accurate. Causes: 40.4% OOV, small corpus, R-field scaling at init. See OC-RA-3 Benchmark Declaration section above. |
| Domain context note | GloVe-50d embeddings only | PPMI-derived (SemanticM::from_corpus) embeddings are also min-max normalized to [0,1]^dim; ρ_base=1.0 and χ₀=1 declarations apply to both. |

---

## V4.0 → V4.1 Change Record

| Item | V4.0 | V4.1 |
|------|-------|-------|
| OC-RA-7 | OPEN — saturating form Origin declaration required | CLOSED — Origin declaration 2026-11-12: bounded (chi/(chi_0+chi) < 1 for all finite chi, stays in D); monotone (increases with chi); midpoint at domain ceiling (chi=chi_0=1 → ρ=ρ_base/2, maximum dynamic range across declared [0,1] domain); dimensionally consistent (chi_0 carries dimension of chi); smooth and continuous (threshold alternatives are discontinuous; linear alternatives do not saturate). These properties are already stated in the ρ section above. No code change. |

---

## V3.0 → V4.0 Change Record

| Item         | V3.0 form                                   | V4.0 canonical V9 form                                              |
|--------------|---------------------------------------------|----------------------------------------------------------------------|
| ρ formula    | \|\|B(A)\|\| / (\|\|B(A)\|\| + eps)        | ρ_base · χ[i] / (χ₀ + χ[i]), χ[i] = max\|A[e]\| at node i         |
| R formula    | B(A(oi,oj)) − B(A(oj,oi))                  | B[e] + ρ[src(e)] · (Σ_{succ(e)} B − Σ_{pred(e)} B)                |
| succ/pred    | succ_admitted filtered by rho_acc           | topology declared separately; succ(e) = {f : src(f)==tgt(e)} in DeclaredGraph |
| ρ_base       | implicit (1.0 in code, undeclared)          | M-declared: doubles ρ ceiling for [0,1]^dim chi domain (V9 §9)     |
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
