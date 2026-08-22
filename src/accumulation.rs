// accumulation.rs — Metatron Dynamics, Inc.
// ABR Relational Attention — Accumulation Rule V2.0
//
// Grounding: operators.rs V7, DECLARATION.md V2.0 (2026-08-21)
//
// V2.0 changes (Verifier findings V4, V5):
//   V5 closed: EdgeStore keyed by token-identity pair (token_id_src, token_id_tgt)
//              not by (position_src, position_tgt). ρ_acc is a property of
//              phonemic-semantic state pairs, persistent across sequences.
//   V4 corrected: accumulation is O(n²) in Phase 0 — window restricts decay
//              weight only, not pair enumeration. O(n·k) is inference cost.
//              Window-restricted accumulation is OC-RA-5 (Phase 1).
//
// Accumulation rule (declared, V7 kernel):
//   ρ_acc(token_i, token_j) ← ρ_acc(token_i, token_j)
//                              + η · ρₙ(i,j) · (1 - ρ_acc(token_i, token_j))
//
// Stability: derived from V7 domain constraint |x| < ∞.
// The factor (1 - ρ_acc) guarantees ρ_acc ∈ [0,1] for all admissible η, ρₙ.

use std::collections::HashMap;
use crate::declaration::{RelationalPair, ETA, THETA_MIN, DOMAIN_LOWER, DOMAIN_UPPER};

/// Accumulated binding for one declared token-identity pair.
/// Persists across sequences — ρ_acc is a property of the token pair,
/// not of the sequence positions where it was observed.
#[derive(Debug, Clone, Copy)]
pub struct AccumulatedBinding {
    pub rho_acc: f64,
    pub exposure_count: usize,
}

impl AccumulatedBinding {
    pub fn new() -> Self {
        Self { rho_acc: DOMAIN_LOWER, exposure_count: 0 }
    }

    /// Apply accumulation rule for one new ρₙ observation.
    ///
    /// ρ_acc ← ρ_acc + η · ρₙ · (1 - ρ_acc)
    ///
    /// Invariant: ρ_acc ∈ [0,1] before and after every call.
    pub fn accumulate(&mut self, rho_n: f64, eta: f64) {
        debug_assert!(rho_n >= DOMAIN_LOWER && rho_n <= DOMAIN_UPPER);
        debug_assert!(eta > 0.0 && eta <= 1.0);
        let delta = eta * rho_n * (1.0 - self.rho_acc);
        self.rho_acc = (self.rho_acc + delta).min(DOMAIN_UPPER);
        self.exposure_count += 1;
        debug_assert!(self.rho_acc >= DOMAIN_LOWER && self.rho_acc <= DOMAIN_UPPER);
    }

    /// Whether this binding is admitted to the active neighbourhood.
    pub fn is_admitted(&self) -> bool {
        self.rho_acc >= THETA_MIN
    }
}

impl Default for AccumulatedBinding {
    fn default() -> Self { Self::new() }
}

/// The edge store — accumulated bindings keyed by token-identity pair.
///
/// Key: (source.token_id, target.token_id) — the phonemic-semantic
/// identity of the relation. Position is not part of the key; it is
/// M's measurement of when the relation was observed, not what it is.
///
/// This means ρ_acc(token_i, token_j) accumulates across all sequences
/// in which token_i precedes token_j in any causal position pair —
/// which is the correct declared behaviour for generalisation (OC-RA-3).
///
/// Inference cost: O(n · k) where k = mean admitted neighbourhood size.
/// Accumulation cost: O(n²) in Phase 0 (OC-RA-5, see DECLARATION.md V4).
#[derive(Debug, Default)]
pub struct EdgeStore {
    /// Keyed by (source_token_id, target_token_id).
    edges: HashMap<(usize, usize), AccumulatedBinding>,
}

impl EdgeStore {
    pub fn new() -> Self { Self::default() }

    /// Update binding for a declared pair given a new ρₙ observation.
    /// Creates entry if not yet seen (cold start admissible on first step).
    pub fn update(&mut self, pair: &RelationalPair, rho_n: f64) {
        let key = pair.token_key(); // (source.token_id, target.token_id)
        let binding = self.edges.entry(key).or_insert_with(AccumulatedBinding::new);
        binding.accumulate(rho_n, ETA);
    }

    /// Return admitted source token_ids and their ρ_acc for a given target token_id.
    /// Used by operator R to build the weighted sum at inference time.
    /// Cost: O(|edges|) scan — O(vocab²) worst case, O(k) for sparse admitted set.
    pub fn admitted_sources_for_token(&self, target_token_id: usize) -> Vec<(usize, f64)> {
        self.edges
            .iter()
            .filter(|((_, t), b)| *t == target_token_id && b.is_admitted())
            .map(|((s, _), b)| (*s, b.rho_acc))
            .collect()
    }

    pub fn total_edges(&self) -> usize { self.edges.len() }

    pub fn admitted_edges(&self) -> usize {
        self.edges.values().filter(|b| b.is_admitted()).count()
    }

    pub fn mean_rho_acc(&self) -> f64 {
        if self.edges.is_empty() { return 0.0; }
        let sum: f64 = self.edges.values().map(|b| b.rho_acc).sum();
        sum / self.edges.len() as f64
    }

    pub fn get(&self, pair: &RelationalPair) -> f64 {
        self.edges.get(&pair.token_key()).map(|b| b.rho_acc).unwrap_or(DOMAIN_LOWER)
    }

    /// Return all admitted token-identity pairs.
    /// Used to establish the declared graph topology before B and R act.
    pub fn all_admitted_pairs(&self) -> Vec<(usize, usize)> {
        self.edges
            .iter()
            .filter(|(_, b)| b.is_admitted())
            .map(|(&(s, t), _)| (s, t))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::declaration::{Locus, RelationalPair, EPSILON};

    fn make_pair(src_token: usize, tgt_token: usize) -> RelationalPair {
        // Use distinct positions to satisfy causal constraint
        RelationalPair::declare(
            Locus::new(0, src_token),
            Locus::new(1, tgt_token),
        ).expect("test pair must be admissible")
    }

    // ── Accumulation rule invariants ────────────────────────────────

    #[test]
    fn test_cold_start_is_zero() {
        let b = AccumulatedBinding::new();
        assert_eq!(b.rho_acc, 0.0);
        assert_eq!(b.exposure_count, 0);
    }

    #[test]
    fn test_accumulation_increases_rho_acc() {
        let mut b = AccumulatedBinding::new();
        b.accumulate(0.5, ETA);
        assert!(b.rho_acc > 0.0);
    }

    #[test]
    fn test_accumulation_never_exceeds_one() {
        let mut b = AccumulatedBinding::new();
        for _ in 0..1000 { b.accumulate(1.0, ETA); }
        assert!(b.rho_acc <= DOMAIN_UPPER);
    }

    #[test]
    fn test_accumulation_bounded_below() {
        let mut b = AccumulatedBinding::new();
        b.accumulate(0.0, ETA);
        assert!(b.rho_acc >= DOMAIN_LOWER);
    }

    #[test]
    fn test_accumulation_monotone_increasing() {
        let mut b = AccumulatedBinding::new();
        let mut prev = b.rho_acc;
        for _ in 0..20 {
            b.accumulate(0.7, ETA);
            assert!(b.rho_acc >= prev);
            prev = b.rho_acc;
        }
    }

    #[test]
    fn test_saturation_slows_accumulation() {
        let mut b = AccumulatedBinding::new();
        let mut increments = Vec::new();
        let mut prev = 0.0_f64;
        for _ in 0..10 {
            b.accumulate(1.0, ETA);
            increments.push(b.rho_acc - prev);
            prev = b.rho_acc;
        }
        for i in 1..increments.len() {
            assert!(increments[i] <= increments[i-1] + EPSILON);
        }
    }

    // ── EdgeStore — token-identity keying (V5) ──────────────────────

    #[test]
    fn test_edge_store_keyed_by_token_not_position() {
        let mut store = EdgeStore::new();
        // Same token pair, different positions — must update same entry
        let pair_a = RelationalPair::declare(Locus::new(0, 2), Locus::new(1, 5)).unwrap();
        let pair_b = RelationalPair::declare(Locus::new(3, 2), Locus::new(7, 5)).unwrap();
        store.update(&pair_a, 0.8);
        store.update(&pair_b, 0.8);
        // Should be one entry, not two
        assert_eq!(store.total_edges(), 1,
            "same token pair at different positions must map to same edge");
    }

    #[test]
    fn test_different_token_pairs_create_separate_entries() {
        let mut store = EdgeStore::new();
        let pair_a = make_pair(1, 2);
        let pair_b = make_pair(3, 4);
        store.update(&pair_a, 0.5);
        store.update(&pair_b, 0.5);
        assert_eq!(store.total_edges(), 2);
    }

    #[test]
    fn test_accumulation_persists_across_sequence_positions() {
        let mut store = EdgeStore::new();
        // token pair (1,2) observed at positions (0,1) in sequence A
        let pair_seq_a = RelationalPair::declare(Locus::new(0, 1), Locus::new(1, 2)).unwrap();
        // token pair (1,2) observed at positions (5,8) in sequence B
        let pair_seq_b = RelationalPair::declare(Locus::new(5, 1), Locus::new(8, 2)).unwrap();
        store.update(&pair_seq_a, 0.6);
        let after_first = store.get(&pair_seq_a);
        store.update(&pair_seq_b, 0.6);
        let after_second = store.get(&pair_seq_b);
        assert!(after_second > after_first,
            "second exposure of same token pair must increase ρ_acc");
    }

    #[test]
    fn test_admitted_sources_for_token() {
        let mut store = EdgeStore::new();
        let pair = make_pair(0, 3);
        for _ in 0..50 { store.update(&pair, 1.0); }
        let sources = store.admitted_sources_for_token(3);
        assert!(!sources.is_empty());
        assert_eq!(sources[0].0, 0); // source token_id
    }

    #[test]
    fn test_unadmitted_excluded_from_sources() {
        let mut store = EdgeStore::new();
        let pair = make_pair(0, 3);
        store.update(&pair, 0.001); // stays below θ_min
        let sources = store.admitted_sources_for_token(3);
        assert!(sources.is_empty());
    }
}

// V4.0 addition -- placed after closing brace of impl EdgeStore above in logic,
// but appended here as patch. The method is on EdgeStore.
