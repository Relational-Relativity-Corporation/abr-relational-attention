// declaration.rs — Metatron Dynamics, Inc.
// ABR Relational Attention — V7 Domain and Measurement Declaration V2.0
//
// Grounding: operators.rs V7, DECLARATION.md V2.0 (2026-08-21)
//
// V2.0 changes (Verifier pass 1 findings V1, V5, V6):
//   V1 closed: direction declared as relational coherence progression;
//              sequence position is M's measurement of that dependency.
//   V5 closed: EdgeStore keyed by token-identity pair, not position pair.
//   V6 closed: Vocabulary table declared; operator_a indexes by token_id
//              into a declared vocabulary embedding table, not positional vec.
//
// Language is a physical relational field. Tokens are declared loci.
// The observable at each locus is the phonemic-semantic state, measured
// by M as an embedding vector in D.
//
// M : O → D
//   O = token positions in a declared input sequence
//   D = { x ∈ ℝⁿ | n < ∞, |x[i]| < ∞ ∀ i }

/// Domain bounds — every declared quantity must satisfy these.
pub const DOMAIN_UPPER: f64 = 1.0;
pub const DOMAIN_LOWER: f64 = 0.0;

/// Stability epsilon — keeps expressions in D (no division by zero).
/// Derived from domain boundedness. Not a free parameter.
pub const EPSILON: f64 = 1e-8;

/// Learning rate η — declared by Origin (OC-RA-2 partially closed).
pub const ETA: f64 = 0.1;

/// Edge admission threshold θ_min (OC-RA-1 partially closed).
pub const THETA_MIN: f64 = 0.05;

/// Coherence window w — declared by Origin.
/// M's measurement of the relational distance over which dependency
/// attenuation is declared significant.
pub const COHERENCE_WINDOW: f64 = 64.0;

/// Maximum sequence length — n < ∞ in D.
pub const MAX_SEQUENCE_LENGTH: usize = 4096;

/// Maximum vocabulary size — bounds the declared locus space.
pub const MAX_VOCAB_SIZE: usize = 65536;

/// Declared direction of relational evolution.
/// Causal: oᵢ is a relational predecessor of oⱼ (i < j) because oⱼ
/// depends on oᵢ for coherence. This is a physical property of the
/// language relational field, not a numerical convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProcessDirection {
    /// Admissible: source position i ≤ target position j.
    /// Grounded in relational coherence progression (DECLARATION.md V2.0 V1).
    Causal,
    /// Requires independent observable provenance — not declared in Phase 0.
    Bidirectional,
}

pub const DECLARED_DIRECTION: ProcessDirection = ProcessDirection::Causal;

/// A declared observable locus — one token at one position in M(oₖ).
/// position: the relational step k in {M(o₁), ..., M(oₙ)} —
///           M's measurement of where in coherence progression this token appears.
/// token_id: the identity of the phonemic-semantic state at this locus,
///           indexing into the declared vocabulary embedding table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Locus {
    pub position: usize,
    pub token_id: usize,
}

impl Locus {
    pub fn new(position: usize, token_id: usize) -> Self {
        assert!(position < MAX_SEQUENCE_LENGTH, "position exceeds declared domain bound");
        assert!(token_id < MAX_VOCAB_SIZE, "token_id exceeds declared domain bound");
        Self { position, token_id }
    }
}

/// The declared vocabulary — a table of phonemic-semantic state embeddings,
/// one per declared token identity. Indexed by token_id.
/// This is the declared M mapping: token_id → embedding ∈ D.
/// (V6 fix: operator_a indexes this table by token_id, not by sequence position.)
pub type VocabTable = Vec<Vec<f64>>;

/// Build a declared vocabulary table for Phase 0 synthetic demonstration.
/// vocab_size: number of declared token identities.
/// dim: embedding dimension.
/// Each token gets a unit vector — traceable to its declared identity.
pub fn build_vocab_table(vocab_size: usize, dim: usize) -> VocabTable {
    (0..vocab_size).map(|id| {
        let mut emb = vec![0.0_f64; dim];
        emb[id % dim] = 1.0;
        emb
    }).collect()
}

/// A declared relational pair (source, target).
/// Admissible iff source.position ≤ target.position under causal direction.
/// The relational identity of this pair is (source.token_id, target.token_id) —
/// what persists across sequences. Position records when it was observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RelationalPair {
    pub source: Locus,
    pub target: Locus,
}

impl RelationalPair {
    /// Returns Some(pair) if admissible under declared direction, None otherwise.
    pub fn declare(source: Locus, target: Locus) -> Option<Self> {
        match DECLARED_DIRECTION {
            ProcessDirection::Causal => {
                if source.position <= target.position {
                    Some(Self { source, target })
                } else {
                    None
                }
            }
            ProcessDirection::Bidirectional => Some(Self { source, target }),
        }
    }

    /// The token-identity key for this pair.
    /// This is what ρ_acc persists — the binding between two phonemic-semantic
    /// states, not between two sequence positions (V5 fix).
    pub fn token_key(&self) -> (usize, usize) {
        (self.source.token_id, self.target.token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_bounds_ordered() {
        assert!(DOMAIN_LOWER < DOMAIN_UPPER);
    }

    #[test]
    fn test_epsilon_in_domain() {
        assert!(EPSILON > 0.0);
        assert!(EPSILON < 1.0);
    }

    #[test]
    fn test_eta_in_range() {
        assert!(ETA > 0.0);
        assert!(ETA <= 1.0);
    }

    #[test]
    fn test_theta_min_in_domain() {
        assert!(THETA_MIN >= DOMAIN_LOWER);
        assert!(THETA_MIN < DOMAIN_UPPER);
    }

    #[test]
    fn test_causal_pair_admissible() {
        let src = Locus::new(0, 10);
        let tgt = Locus::new(3, 20);
        assert!(RelationalPair::declare(src, tgt).is_some());
    }

    #[test]
    fn test_self_relation_admissible() {
        let loc = Locus::new(2, 5);
        assert!(RelationalPair::declare(loc, loc).is_some());
    }

    #[test]
    fn test_reverse_direction_inadmissible() {
        let src = Locus::new(5, 10);
        let tgt = Locus::new(2, 20);
        assert!(RelationalPair::declare(src, tgt).is_none());
    }

    #[test]
    fn test_token_key_is_identity_not_position() {
        // Same token_ids at different positions must yield same token_key
        let pair_a = RelationalPair::declare(Locus::new(0, 3), Locus::new(1, 7)).unwrap();
        let pair_b = RelationalPair::declare(Locus::new(5, 3), Locus::new(9, 7)).unwrap();
        assert_eq!(pair_a.token_key(), pair_b.token_key(),
            "token_key must depend on identity, not position");
    }

    #[test]
    fn test_different_token_ids_yield_different_keys() {
        let pair_a = RelationalPair::declare(Locus::new(0, 1), Locus::new(1, 2)).unwrap();
        let pair_b = RelationalPair::declare(Locus::new(0, 3), Locus::new(1, 4)).unwrap();
        assert_ne!(pair_a.token_key(), pair_b.token_key());
    }

    #[test]
    fn test_vocab_table_indexed_by_token_id() {
        let table = build_vocab_table(4, 4);
        assert_eq!(table.len(), 4);
        // token_id 0 should have embedding[0] = 1.0
        assert!((table[0][0] - 1.0).abs() < EPSILON);
        // token_id 1 should have embedding[1] = 1.0
        assert!((table[1][1] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_vocab_table_nontrivial_token_order() {
        // Token at position 0 may have token_id != 0
        // Verify table lookup by token_id is independent of position
        let table = build_vocab_table(4, 4);
        let locus = Locus::new(3, 2); // position 3, token_id 2
        let emb = &table[locus.token_id];
        assert!((emb[2] - 1.0).abs() < EPSILON,
            "embedding must be indexed by token_id, not position");
    }
}
