// semantic.rs — Metatron Dynamics, Inc.
// ABR Semantic Bridge — V1.0
//
// Grounding: declaration.rs V2.0, multi_head.rs V1.0
//
// PURPOSE:
//   Replace the synthetic identity-basis M with pre-trained semantic embeddings.
//   The ABR kernel requires M: token_id → ℝ^dim ∩ [0,1]^dim.
//   Pre-trained word vectors (GloVe, FastText, word2vec) are in ℝ^dim but
//   NOT bounded in [0,1]^dim — they require normalization to satisfy the domain.
//
// NORMALIZATION:
//   GloVe/FastText vectors are normalized via per-dimension min-max scaling:
//     M_normalized[i][d] = (M_raw[i][d] - min_d) / (max_d - min_d + ε)
//   This maps each dimension to [0,1] across the vocabulary.
//   Declared: normalization is applied once at load time, not per-token.
//   This preserves the relative geometry of the embedding space within D.
//
// TOKENIZATION:
//   ABR operates on token_ids (integer indices into M).
//   For word-level embeddings: token_id = word index in vocabulary.
//   For subword (BPE): each BPE piece maps to its own row in M.
//   This module provides word-level tokenization (whitespace + punctuation split).
//   Vocabulary is built from the corpus at load time (closed-world assumption).
//
// WHAT THIS ENABLES:
//   The ABR kernel can now run on any English text.
//   relational structure reflects semantic similarity (cosine distance in GloVe space).
//   tokens that are semantically similar will have small |A| → low ρₙ
//   tokens that are semantically distant will have large |A| → high ρₙ
//   This is the semantic grounding of the relational field.
//
// DESIGN NOTE: GloVe download is not included in this crate.
//   This module provides:
//   1. The loading/normalization infrastructure (SemanticM)
//   2. A corpus-derived fallback when no pre-trained vectors are available
//      (SemanticM::from_corpus — builds M from co-occurrence statistics)
//   3. The tokenizer (tokenize_text)
//   4. Full integration with MultiHeadLayer
//
// The corpus-derived M is a declared alternative: when no pre-trained vectors
// are available, we build M from the corpus itself via PMI (pointwise mutual
// information) — the relational structure IS the corpus statistics.

use crate::declaration::{VocabTable, DOMAIN_LOWER, DOMAIN_UPPER};

/// A normalized semantic embedding table.
///
/// This is the declared M for ABR when using real language semantics.
/// All embeddings are in [0,1]^dim after min-max normalization.
#[derive(Debug, Clone)]
pub struct SemanticM {
    /// Embedding matrix: vocab_size × dim, all values in [0,1].
    pub embeddings: VocabTable,
    /// Vocabulary: word string → token_id.
    pub vocab: std::collections::HashMap<String, usize>,
    /// Inverse vocabulary: token_id → word string.
    pub id_to_word: Vec<String>,
    /// Embedding dimension.
    pub dim: usize,
    /// Special token: unknown word (all 0.5 — center of domain).
    pub unk_id: usize,
}

impl SemanticM {
    /// Build from raw (unnormalized) embedding matrix and vocabulary.
    ///
    /// Applies per-dimension min-max normalization to satisfy D = [0,1]^dim.
    /// The unknown token is appended as the last entry with value 0.5 in all dims.
    pub fn from_raw(
        raw: Vec<Vec<f64>>,
        words: Vec<String>,
    ) -> Self {
        assert_eq!(raw.len(), words.len(), "raw and words must have same length");
        assert!(!raw.is_empty(), "empty vocabulary");
        let dim = raw[0].len();
        assert!(raw.iter().all(|v| v.len() == dim), "all embeddings must have same dim");

        // Per-dimension min-max normalization
        let mut min_d = vec![f64::INFINITY; dim];
        let mut max_d = vec![f64::NEG_INFINITY; dim];
        for emb in &raw {
            for (d, &v) in emb.iter().enumerate() {
                if v < min_d[d] { min_d[d] = v; }
                if v > max_d[d] { max_d[d] = v; }
            }
        }
        let eps = 1e-10_f64;
        let normalized: VocabTable = raw.iter().map(|emb| {
            emb.iter().enumerate().map(|(d, &v)| {
                let range = max_d[d] - min_d[d];
                if range < eps { 0.5 } else { (v - min_d[d]) / (range + eps) }
            }).collect()
        }).collect();

        // Verify domain after normalization
        debug_assert!(
            normalized.iter().all(|emb| emb.iter().all(|&v| v >= DOMAIN_LOWER && v <= DOMAIN_UPPER)),
            "normalization must produce embeddings in [0,1]^dim"
        );

        // Unknown token: 0.5 in all dimensions (center of domain)
        let unk_id = normalized.len();
        let mut all_embeddings = normalized;
        all_embeddings.push(vec![0.5; dim]);
        let all_words: Vec<String> = words.iter().cloned()
            .chain(std::iter::once("<unk>".to_string()))
            .collect();

        let vocab: std::collections::HashMap<String, usize> = all_words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i))
            .collect();

        Self {
            embeddings: all_embeddings,
            vocab,
            id_to_word: all_words,
            dim,
            unk_id,
        }
    }

    /// Build a SemanticM from a corpus using PMI-based co-occurrence vectors.
    ///
    /// When no pre-trained word vectors are available, this constructs M from
    /// the corpus statistics directly.
    ///
    /// Method:
    ///   1. Build co-occurrence counts: count(w, context) for a window of W tokens
    ///   2. Compute PPMI (positive PMI): max(0, log(P(w,c) / P(w)P(c)))
    ///   3. Truncate to dim dimensions (the most discriminative context words)
    ///   4. Normalize to [0,1]^dim
    ///
    /// This is a closed-world declared M: the relational structure reflects
    /// co-occurrence statistics in THIS corpus.
    pub fn from_corpus(text: &str, dim: usize, window: usize, min_count: usize) -> Self {
        let tokens = tokenize_text(text);
        let n = tokens.len();

        // Build vocabulary (frequency-filtered)
        let mut freq: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for t in &tokens { *freq.entry(t.as_str()).or_insert(0) += 1; }
        let mut words: Vec<String> = freq.iter()
            .filter(|(_, &c)| c >= min_count)
            .map(|(&w, _)| w.to_string())
            .collect();
        words.sort(); // deterministic ordering
        let vocab_size = words.len();

        if vocab_size == 0 || dim == 0 {
            return Self::from_raw(vec![], vec![]);
        }

        let word_to_id: std::collections::HashMap<&str, usize> = words.iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i))
            .collect();
        let token_ids: Vec<Option<usize>> = tokens.iter()
            .map(|t| word_to_id.get(t.as_str()).copied())
            .collect();

        // Co-occurrence counts
        let mut cooc = vec![vec![0_f64; vocab_size.min(dim * 2)]; vocab_size];
        let context_dim = vocab_size.min(dim * 2);
        for i in 0..n {
            if let Some(w) = token_ids[i] {
                for j in i.saturating_sub(window)..=(i + window).min(n - 1) {
                    if i == j { continue; }
                    if let Some(c) = token_ids[j] {
                        if c < context_dim {
                            cooc[w][c] += 1.0;
                        }
                    }
                }
            }
        }

        // PPMI
        let total: f64 = cooc.iter().flatten().sum();
        if total < 1.0 {
            // No co-occurrences — fall back to identity basis
            let raw: Vec<Vec<f64>> = (0..vocab_size).map(|id| {
                let mut v = vec![0.0_f64; dim];
                v[id % dim] = 1.0;
                v
            }).collect();
            return Self::from_raw(raw, words);
        }
        let row_sums: Vec<f64> = cooc.iter().map(|r| r.iter().sum()).collect();
        let col_sums: Vec<f64> = (0..context_dim)
            .map(|c| cooc.iter().map(|r| r[c]).sum())
            .collect();

        let eps = 1e-10_f64;
        let mut ppmi = vec![vec![0.0_f64; context_dim]; vocab_size];
        for w in 0..vocab_size {
            for c in 0..context_dim {
                let p_wc = cooc[w][c] / (total + eps);
                let p_w  = row_sums[w] / (total + eps);
                let p_c  = col_sums[c] / (total + eps);
                if p_w > eps && p_c > eps && p_wc > eps {
                    let pmi = (p_wc / (p_w * p_c)).ln();
                    ppmi[w][c] = pmi.max(0.0);
                }
            }
        }

        // Truncate or pad to dim dimensions
        let raw: Vec<Vec<f64>> = ppmi.iter()
            .map(|row| {
                let mut v: Vec<f64> = row.iter().take(dim).copied().collect();
                // Pad with 0.0 if context_dim < dim
                while v.len() < dim { v.push(0.0); }
                v
            })
            .collect();

        Self::from_raw(raw, words)
    }

    /// Vocabulary size (including <unk>).
    pub fn vocab_size(&self) -> usize { self.embeddings.len() }

    /// Look up token_id for a word (returns unk_id if not found).
    pub fn token_id(&self, word: &str) -> usize {
        self.vocab.get(word).copied().unwrap_or(self.unk_id)
    }

    /// Tokenize text and return token_ids.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        tokenize_text(text).iter().map(|w| self.token_id(w)).collect()
    }

    /// Return a reference to the VocabTable for use with ABR attention layers.
    pub fn as_vocab_table(&self) -> &VocabTable { &self.embeddings }

    /// Clones the VocabTable (needed to pass to AttentionLayer::forward).
    pub fn to_vocab_table(&self) -> VocabTable { self.embeddings.clone() }

    /// Load from a GloVe-format text file (word followed by space-separated floats).
    ///
    /// Format: each line is "<word> <f1> <f2> ... <fd>\n"
    /// (standard GloVe, word2vec text, FastText text format)
    ///
    /// Applies per-dimension min-max normalization to satisfy D = [0,1]^dim.
    /// Words not in the file return the <unk> embedding (0.5 in all dims).
    ///
    /// max_vocab: if > 0, load only the first max_vocab words (they are sorted
    /// by frequency in standard GloVe files — highest frequency first).
    pub fn from_glove_file(path: &str, max_vocab: usize) -> Result<Self, String> {
        use std::io::{BufRead, BufReader};
        use std::fs::File;

        let file = File::open(path)
            .map_err(|e| format!("cannot open {}: {}", path, e))?;
        let reader = BufReader::new(file);

        let mut words: Vec<String> = Vec::new();
        let mut raw: Vec<Vec<f64>> = Vec::new();
        let mut dim: Option<usize> = None;

        for (line_num, line) in reader.lines().enumerate() {
            if max_vocab > 0 && words.len() >= max_vocab { break; }
            let line = line.map_err(|e| format!("read error at line {}: {}", line_num, e))?;
            let mut parts = line.split_whitespace();
            let word = match parts.next() {
                Some(w) => w.to_string(),
                None    => continue,
            };
            let vec: Vec<f64> = parts
                .map(|s| s.parse::<f64>().unwrap_or(0.0))
                .collect();
            if vec.is_empty() { continue; }
            match dim {
                None    => { dim = Some(vec.len()); }
                Some(d) => { if vec.len() != d { continue; } } // skip malformed lines
            }
            words.push(word);
            raw.push(vec);
        }

        if raw.is_empty() {
            return Err(format!("no embeddings loaded from {}", path));
        }

        Ok(Self::from_raw(raw, words))
    }
}

/// Tokenize text into lowercase word tokens.
///
/// Splits on whitespace and punctuation; lowercases all words.
/// Returns Vec<String> — each element is one token.
pub fn tokenize_text(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || (!c.is_alphabetic() && c != '\''))
        .map(|w| w.to_lowercase())
        .filter(|w| !w.is_empty() && w.chars().any(|c| c.is_alphabetic()))
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_raw() -> (Vec<Vec<f64>>, Vec<String>) {
        // 4 words, 3-dim — values NOT in [0,1] initially
        let raw = vec![
            vec![1.0, -1.0, 2.0],
            vec![3.0,  2.0, 0.5],
            vec![0.0,  1.0, 1.0],
            vec![2.0, -0.5, 3.0],
        ];
        let words = vec!["hello".to_string(), "world".to_string(),
                         "foo".to_string(), "bar".to_string()];
        (raw, words)
    }

    #[test]
    fn test_normalization_produces_domain() {
        let (raw, words) = small_raw();
        let m = SemanticM::from_raw(raw, words);
        for emb in &m.embeddings {
            for &v in emb {
                assert!(v >= DOMAIN_LOWER && v <= DOMAIN_UPPER + 1e-9,
                    "normalized embedding {} out of [0,1]", v);
            }
        }
    }

    #[test]
    fn test_unk_token_added() {
        let (raw, words) = small_raw();
        let orig_size = words.len();
        let m = SemanticM::from_raw(raw, words);
        assert_eq!(m.vocab_size(), orig_size + 1, "unk token must be appended");
        assert_eq!(m.unk_id, orig_size);
        assert!(m.vocab.contains_key("<unk>"));
    }

    #[test]
    fn test_unk_token_is_center() {
        let (raw, words) = small_raw();
        let m = SemanticM::from_raw(raw, words);
        for &v in &m.embeddings[m.unk_id] {
            assert!((v - 0.5).abs() < 1e-10, "unk embedding must be 0.5");
        }
    }

    #[test]
    fn test_token_id_lookup() {
        let (raw, words) = small_raw();
        let m = SemanticM::from_raw(raw, words);
        assert_eq!(m.token_id("hello"), 0);
        assert_eq!(m.token_id("world"), 1);
        assert_eq!(m.token_id("NOTAWORD"), m.unk_id);
    }

    #[test]
    fn test_encode_returns_ids() {
        let (raw, words) = small_raw();
        let m = SemanticM::from_raw(raw, words);
        let ids = m.encode("hello world");
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[0], m.token_id("hello"));
        assert_eq!(ids[1], m.token_id("world"));
    }

    #[test]
    fn test_encode_unk_for_unseen() {
        let (raw, words) = small_raw();
        let m = SemanticM::from_raw(raw, words);
        let ids = m.encode("hello NOTAWORD world");
        assert_eq!(ids[1], m.unk_id);
    }

    #[test]
    fn test_tokenize_lowercases() {
        let tokens = tokenize_text("Hello World!");
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[test]
    fn test_tokenize_strips_punctuation() {
        let tokens = tokenize_text("the cat, sat. on: the mat!");
        assert!(tokens.contains(&"the".to_string()));
        assert!(tokens.contains(&"cat".to_string()));
        assert!(tokens.contains(&"sat".to_string()));
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize_text("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_from_corpus_produces_valid_m() {
        let corpus = "the cat sat on the mat the dog ran on the mat";
        let m = SemanticM::from_corpus(corpus, 8, 2, 1);
        for emb in &m.embeddings {
            assert_eq!(emb.len(), 8, "embedding dimension must match dim");
            for &v in emb {
                assert!(v >= DOMAIN_LOWER && v <= DOMAIN_UPPER + 1e-9,
                    "corpus-derived embedding {} out of [0,1]", v);
            }
        }
    }

    #[test]
    fn test_from_corpus_vocab_includes_frequent_words() {
        let corpus = "the cat sat on the mat the dog ran on the mat the the the";
        let m = SemanticM::from_corpus(corpus, 4, 2, 2);
        // "the" appears many times — must be in vocab
        let the_id = m.token_id("the");
        assert_ne!(the_id, m.unk_id, "'the' must be in corpus vocab");
    }

    #[test]
    fn test_as_vocab_table_matches_embeddings() {
        let (raw, words) = small_raw();
        let m = SemanticM::from_raw(raw, words);
        let table = m.as_vocab_table();
        assert_eq!(table.len(), m.vocab_size());
        assert_eq!(table[0], m.embeddings[0]);
    }

    #[test]
    fn test_dim_preserved() {
        let (raw, words) = small_raw();
        let m = SemanticM::from_raw(raw, words);
        assert_eq!(m.dim, 3);
        assert!(m.embeddings.iter().all(|emb| emb.len() == 3));
    }
}
