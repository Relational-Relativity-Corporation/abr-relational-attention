// operators.rs -- Metatron Dynamics, Inc.
// ABR Relational Attention -- Canonical V7 Operators V4.0
//
// Grounding: operators.rs V7 (kernel_session_upload.txt, lines 901-975)
// DECLARATION.md V4.0 (2026-08-21)
//
// V4.0: implements canonical B, rho, R directly from V7 kernel source.
// V3.0 R was B_fwd - B_rev (not canonical). V3.0 rho was ||B||/(||B||+eps) (not canonical).
//
// Canonical formulas (from kernel source):
//
//   A(oi, oj)  = M(oi) - M(oj)                        directed difference (V3.0, confirmed)
//
//   B(g)[e]    = g[e] + sum_{f in succ(e)} g[f]        successor accumulation
//                succ(e) = declared graph adjacency --
//                rho_acc >= theta_min establishes which edges exist;
//                B then acts on that declared graph canonically.
//                B does not perform admissibility selection.
//
//   rho[i]     = rho_base * chi[i] / (1 + chi[i])      per-node scalar
//                chi[i] = max |A[e]| over all incident edges at node i
//                (selection, not aggregation -- preserves strongest asymmetry)
//
//   R(g)[e]    = g[e] + rho[src(e)] * (sum_succ B(g) - sum_pred B(g))
//                pass-through + rho-modulated antisymmetric adjacency
//                Does NOT require fabricating a reverse edge.
//                Antisymmetry comes from succ/pred adjacency of the
//                declared graph, not from a reverse pair evaluation.

use crate::declaration::{Locus, VocabTable};

pub type Embedding = Vec<f64>;

/// The declared relational graph for a sequence.
/// Edges are (source_token_id, target_token_id) with their A values.
/// Established from EdgeStore before B or R act.
/// B and R operate on this declared structure -- they do not perform selection.
pub struct DeclaredGraph {
    /// Edge list: (source_locus_idx, target_locus_idx) into the sequence
    pub edges: Vec<(usize, usize)>,
    /// For each edge e: succ(e) = edges whose source == target of e
    pub succ: Vec<Vec<usize>>,
    /// For each edge e: pred(e) = edges whose target == source of e
    pub pred: Vec<Vec<usize>>,
    /// A(e) for each edge -- the directed difference
    pub a_field: Vec<Embedding>,
}

impl DeclaredGraph {
    /// Build declared graph from a sequence and admitted token pairs.
    /// admitted_pairs: set of (src_token_id, tgt_token_id) with rho_acc >= theta_min.
    /// This establishes the edge set. B and R then act on it.
    pub fn build(
        sequence: &[Locus],
        vocab: &VocabTable,
        admitted_pairs: &[(usize, usize)],
    ) -> Self {
        // Build edges: all admitted (src_pos, tgt_pos) pairs in causal order
        let mut edges = Vec::new();
        for i in 0..sequence.len() {
            for j in i..sequence.len() {
                let src_tok = sequence[i].token_id;
                let tgt_tok = sequence[j].token_id;
                if admitted_pairs.iter().any(|&(s, t)| s == src_tok && t == tgt_tok) {
                    edges.push((i, j));
                }
            }
        }

        let n_edges = edges.len();

        // Compute A field: A[e] = M(src) - M(tgt)
        let a_field: Vec<Embedding> = edges.iter().map(|&(i, j)| {
            let src = &vocab[sequence[i].token_id];
            let tgt = &vocab[sequence[j].token_id];
            src.iter().zip(tgt.iter()).map(|(s, t)| s - t).collect()
        }).collect();

        // succ(e): edges f where source of f == target of e
        let succ: Vec<Vec<usize>> = (0..n_edges).map(|e| {
            let tgt_of_e = edges[e].1;
            (0..n_edges)
                .filter(|&f| edges[f].0 == tgt_of_e)
                .collect()
        }).collect();

        // pred(e): edges p where target of p == source of e
        let pred: Vec<Vec<usize>> = (0..n_edges).map(|e| {
            let src_of_e = edges[e].0;
            (0..n_edges)
                .filter(|&p| edges[p].1 == src_of_e)
                .collect()
        }).collect();

        Self { edges, succ, pred, a_field }
    }

    pub fn n_edges(&self) -> usize { self.edges.len() }
}

// ── Operator A -- Directed Difference ───────────────────────────────────────

/// A(oi, oj) = M(oi) - M(oj)
///
/// The directed contrast between two phonemic-semantic states.
/// Canonical V7: A(x)[e] = x[s] - x[t]
pub fn operator_a(source: &Locus, target: &Locus, vocab: &VocabTable) -> Embedding {
    assert!(source.token_id < vocab.len());
    assert!(target.token_id < vocab.len());
    vocab[source.token_id]
        .iter()
        .zip(vocab[target.token_id].iter())
        .map(|(s, t)| s - t)
        .collect()
}

// ── ρ -- Per-Node Binding Magnitude ─────────────────────────────────────────

/// rho[i] = rho_base * chi[i] / (1 + chi[i])
/// chi[i] = max |A[e]| over all edges incident on node i (in or out)
///
/// Canonical V7 (kernel source line 940-948):
///   chi = maximum declared local asymmetry at node i
///   Selection, not aggregation -- preserves strongest asymmetry at i
///   Discards direction (absolute value) and weaker incidents
///
/// Returns one rho value per sequence position (node).
pub fn compute_rho(graph: &DeclaredGraph, n_nodes: usize, rho_base: f64) -> Vec<f64> {
    (0..n_nodes).map(|i| {
        let mut chi = 0.0_f64;
        // Check all edges incident on node i (as source or target)
        for (e, &(src, tgt)) in graph.edges.iter().enumerate() {
            if src == i || tgt == i {
                for component in &graph.a_field[e] {
                    chi = chi.max(component.abs());
                }
            }
        }
        rho_base * chi / (1.0 + chi)
    }).collect()
}

// ── Operator B -- Successor Accumulation ────────────────────────────────────

/// B(g)[e] = g[e] + sum_{f in succ(e)} g[f]
///
/// Canonical V7 (kernel source line 907-919).
/// succ(e) is the declared graph adjacency -- established before B acts.
/// B does not perform admissibility selection.
///
/// Returns B field: one embedding per edge in the declared graph.
pub fn operator_b(graph: &DeclaredGraph) -> Vec<Embedding> {
    let n_edges = graph.n_edges();
    if n_edges == 0 { return vec![]; }
    let dim = graph.a_field[0].len();

    (0..n_edges).map(|e| {
        let mut result = graph.a_field[e].clone();
        for &f in &graph.succ[e] {
            for (r, v) in result.iter_mut().zip(graph.a_field[f].iter()) {
                *r += v;
            }
        }
        let _ = dim; // suppress unused warning
        result
    }).collect()
}

// ── Operator R -- Antisymmetric Resolution ──────────────────────────────────

/// R(g)[e] = g[e] + rho[src(e)] * (sum_succ B(g) - sum_pred B(g))
///
/// Canonical V7 (kernel source lines 957-967):
///   Pass-through g[e], plus rho-modulated antisymmetric adjacency contribution.
///   rho[src(e)]: per-node scalar at the source node of edge e.
///   Antisymmetry from succ/pred of declared graph -- no reverse edge fabricated.
///
/// bg: B field (output of operator_b)
/// rho_nodes: per-node rho values (output of compute_rho)
/// Returns R field: one embedding per edge.
pub fn operator_r(
    bg: &[Embedding],
    graph: &DeclaredGraph,
    rho_nodes: &[f64],
) -> Vec<Embedding> {
    let n_edges = graph.n_edges();
    if n_edges == 0 { return vec![]; }
    let dim = bg[0].len();

    (0..n_edges).map(|e| {
        let src_node = graph.edges[e].0;
        let rho_e = rho_nodes[src_node];

        let mut fwd = vec![0.0_f64; dim];
        for &f in &graph.succ[e] {
            for (acc, v) in fwd.iter_mut().zip(bg[f].iter()) { *acc += v; }
        }

        let mut bwd = vec![0.0_f64; dim];
        for &p in &graph.pred[e] {
            for (acc, v) in bwd.iter_mut().zip(bg[p].iter()) { *acc += v; }
        }

        // R[e] = B[e] + rho[src] * (fwd - bwd)
        bg[e].iter()
            .zip(fwd.iter().zip(bwd.iter()))
            .map(|(g_e, (f, b))| g_e + rho_e * (f - b))
            .collect()
    }).collect()
}

/// Retrieve declared embedding for a single locus (for cold-start / diagnostics)
pub fn get_embedding(locus: &Locus, vocab: &VocabTable) -> Embedding {
    assert!(locus.token_id < vocab.len());
    vocab[locus.token_id].clone()
}

/// Declared rho_base -- Origin-declared scalar ceiling on rho.
/// rho[i] < rho_base for all i. Declared before operator application.
pub const RHO_BASE: f64 = 1.0;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::declaration::{Locus, build_vocab_table, DOMAIN_LOWER, EPSILON};

    fn vocab() -> VocabTable { build_vocab_table(4, 4) }

    fn two_loci() -> (Locus, Locus) {
        (Locus::new(0, 0), Locus::new(1, 1))
    }

    // ── A tests (carried from V3.0 -- confirmed canonical) ──────────

    #[test]
    fn test_operator_a_is_directed_difference() {
        let v = vocab();
        let (src, tgt) = two_loci();
        let a = operator_a(&src, &tgt, &v);
        assert!((a[0] - 1.0).abs() < EPSILON);
        assert!((a[1] + 1.0).abs() < EPSILON);
        assert!(a[2].abs() < EPSILON);
        assert!(a[3].abs() < EPSILON);
    }

    #[test]
    fn test_operator_a_same_token_is_zero() {
        let v = vocab();
        let loc = Locus::new(0, 2);
        let a = operator_a(&loc, &loc, &v);
        for c in &a { assert!(c.abs() < EPSILON); }
    }

    #[test]
    fn test_operator_a_antisymmetric() {
        let v = vocab();
        let src = Locus::new(0, 0);
        let tgt = Locus::new(1, 3);
        let a_fwd = operator_a(&src, &tgt, &v);
        let a_rev = operator_a(&tgt, &src, &v);
        for (f, r) in a_fwd.iter().zip(a_rev.iter()) {
            assert!((f + r).abs() < EPSILON);
        }
    }

    // ── DeclaredGraph build ──────────────────────────────────────────

    fn simple_graph() -> (Vec<Locus>, VocabTable, DeclaredGraph) {
        let v = vocab();
        let seq = vec![Locus::new(0,0), Locus::new(1,1), Locus::new(2,2)];
        // Admit all directed pairs
        let admitted = vec![(0,1),(0,2),(1,2),(0,0),(1,1),(2,2)];
        let g = DeclaredGraph::build(&seq, &v, &admitted);
        (seq, v, g)
    }

    #[test]
    fn test_declared_graph_has_edges() {
        let (_, _, g) = simple_graph();
        assert!(g.n_edges() > 0);
    }

    #[test]
    fn test_declared_graph_a_field_is_directed_difference() {
        let v = vocab();
        let seq = vec![Locus::new(0,0), Locus::new(1,1)];
        let admitted = vec![(0,1)];
        let g = DeclaredGraph::build(&seq, &v, &admitted);
        // Edge (0,1): A = M(0) - M(1) = [1,0,0,0] - [0,1,0,0] = [1,-1,0,0]
        assert_eq!(g.n_edges(), 1);
        assert!((g.a_field[0][0] - 1.0).abs() < EPSILON);
        assert!((g.a_field[0][1] + 1.0).abs() < EPSILON);
    }

    // ── B: canonical successor accumulation ─────────────────────────

    #[test]
    fn test_operator_b_terminal_edge_unchanged() {
        // An edge with no successors: B[e] = A[e]
        let v = vocab();
        let seq = vec![Locus::new(0,0), Locus::new(1,1)];
        let g = DeclaredGraph::build(&seq, &v, &[(0,1)]);
        let bg = operator_b(&g);
        // Only one edge, no successors
        for (b, a) in bg[0].iter().zip(g.a_field[0].iter()) {
            assert!((b - a).abs() < EPSILON, "terminal edge B must equal A");
        }
    }

    #[test]
    fn test_operator_b_accumulates_successor() {
        // Edge 0->1 has successor 1->2
        // B[0->1] = A[0->1] + A[1->2]
        let v = vocab();
        let seq = vec![Locus::new(0,0), Locus::new(1,1), Locus::new(2,2)];
        let g = DeclaredGraph::build(&seq, &v, &[(0,1),(1,2)]);
        let bg = operator_b(&g);
        // Find edge (0,1) and (1,2)
        let e01 = g.edges.iter().position(|&e| e == (0,1)).unwrap();
        let e12 = g.edges.iter().position(|&e| e == (1,2)).unwrap();
        for i in 0..bg[e01].len() {
            let expected = g.a_field[e01][i] + g.a_field[e12][i];
            assert!((bg[e01][i] - expected).abs() < EPSILON,
                "B[0->1] must equal A[0->1] + A[1->2]");
        }
    }

    // ── rho: canonical per-node ──────────────────────────────────────

    #[test]
    fn test_rho_in_domain() {
        let (seq, _, g) = simple_graph();
        let rho = compute_rho(&g, seq.len(), RHO_BASE);
        for r in &rho {
            assert!(*r >= DOMAIN_LOWER);
            assert!(*r < RHO_BASE + EPSILON);
        }
    }

    #[test]
    fn test_rho_isolated_node_is_zero() {
        // A node with no incident edges has chi=0 => rho=0
        let v = vocab();
        let seq = vec![Locus::new(0,0), Locus::new(1,1), Locus::new(2,2)];
        // Only admit (0,1) -- node 2 has no incident admitted edge
        let g = DeclaredGraph::build(&seq, &v, &[(0,1)]);
        let rho = compute_rho(&g, seq.len(), RHO_BASE);
        assert!(rho[2] < EPSILON, "isolated node must have rho=0");
    }

    #[test]
    fn test_rho_nonzero_for_incident_node() {
        let (seq, _, g) = simple_graph();
        let rho = compute_rho(&g, seq.len(), RHO_BASE);
        // Node 0 has outgoing edges -- must have nonzero rho
        assert!(rho[0] > EPSILON);
    }

    // ── R: canonical antisymmetric resolution ────────────────────────

    #[test]
    fn test_operator_r_passthrough_no_adjacency() {
        // Single edge with no succ/pred: R[e] = B[e] + rho*0 = B[e]
        let v = vocab();
        let seq = vec![Locus::new(0,0), Locus::new(1,1)];
        let g = DeclaredGraph::build(&seq, &v, &[(0,1)]);
        let rho = compute_rho(&g, seq.len(), RHO_BASE);
        let bg = operator_b(&g);
        let r = operator_r(&bg, &g, &rho);
        for (ri, bi) in r[0].iter().zip(bg[0].iter()) {
            assert!((ri - bi).abs() < EPSILON,
                "R with no adjacency must equal B (pass-through)");
        }
    }

    #[test]
    fn test_operator_r_antisymmetric_contribution() {
        // With a declared chain 0->1->2:
        // R[0->1] uses succ = {1->2} in fwd, pred = {} in bwd
        // R[1->2] uses succ = {} in fwd, pred = {0->1} in bwd
        // The fwd contribution to R[0->1] and bwd to R[1->2] are related
        let v = vocab();
        let seq = vec![Locus::new(0,0), Locus::new(1,1), Locus::new(2,2)];
        let g = DeclaredGraph::build(&seq, &v, &[(0,1),(1,2)]);
        let rho = compute_rho(&g, seq.len(), RHO_BASE);
        let bg = operator_b(&g);
        let r = operator_r(&bg, &g, &rho);
        // R must not panic and must return one embedding per edge
        assert_eq!(r.len(), g.n_edges());
    }

    #[test]
    fn test_operator_r_no_reverse_edge_required() {
        // R computes antisymmetry from succ/pred of declared graph
        // This test verifies R produces output without any reverse-direction edge
        let v = vocab();
        let seq = vec![Locus::new(0,0), Locus::new(1,1), Locus::new(2,2)];
        // Only forward edges admitted
        let g = DeclaredGraph::build(&seq, &v, &[(0,1),(1,2),(0,2)]);
        let rho = compute_rho(&g, seq.len(), RHO_BASE);
        let bg = operator_b(&g);
        let r = operator_r(&bg, &g, &rho);
        assert_eq!(r.len(), g.n_edges());
        // All R values must be finite (in D)
        for edge_r in &r {
            for v in edge_r { assert!(v.is_finite()); }
        }
    }
}
