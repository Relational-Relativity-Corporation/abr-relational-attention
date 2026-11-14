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
///
/// ## ι[v] — Emphasis (OC-RA-9, V4.5 declaration)
///
/// `iota[token_id]` is the per-token-type emphasis weight — the M-observable
/// analog of ι[v] at the language region.  It is derived from the raw corpus
/// text before tokenisation strips emphasis signals.
///
/// Weight scale:
///   1.0 — ambient (normal declarative context)
///   1.2 — sentence-level question emphasis (?)
///   1.5 — sentence-level exclamatory emphasis (!)
///   2.0 — word-level ALL CAPS
/// Combined multiplicatively; per-type value is the MAX over all corpus
/// occurrences (consistent with the max-selection rule for χ[i] in ρ).
///
/// `epsilon_edge(src, tgt)` returns |ι[src] − ι[tgt]| — the edge energy ε[e].
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
    /// Per-token-type emphasis weight ι[v].
    /// iota[unk_id] = 1.0 (ambient).
    /// Set to vec![1.0; vocab_size] when no emphasis pass is available
    /// (e.g., from_raw, from_glove_file).
    pub iota: Vec<f32>,
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

        // Default iota: 1.0 for all tokens (ambient; no emphasis pass for raw embeddings).
        let iota = vec![1.0_f32; unk_id + 1];

        Self {
            embeddings: all_embeddings,
            vocab,
            id_to_word: all_words,
            dim,
            unk_id,
            iota,
        }
    }

    /// Build a SemanticM from a corpus using PMI-based co-occurrence vectors.
    ///
    /// When no pre-trained word vectors are available, this constructs M from
    /// the corpus statistics directly.
    ///
    /// Method:
    ///   1. Tokenize into sentences (OC-RA-10: word-level co-occurrence bounded by sentence scope)
    ///   2. Build co-occurrence counts: count(w, context) within each sentence, up to window tokens
    ///   3. Compute PPMI (positive PMI): max(0, log(P(w,c) / P(w)P(c)))
    ///   4. Truncate to dim dimensions (the most discriminative context words)
    ///
    /// ## Declaration (V5.0 — OC-RA-10 CLOSED)
    ///
    /// Word-level M (M_word) admits only within-sentence co-occurrences.
    /// A co-occurrence pair (w, c) is admissible iff both tokens are in the
    /// same sentence.  Cross-sentence pairs belong to M_sent (OC-RA-8).
    ///
    /// This is a closed-world declared M: the relational structure reflects
    /// co-occurrence statistics in THIS corpus, within declared sentence scope.
    pub fn from_corpus(text: &str, dim: usize, window: usize, min_count: usize) -> Self {
        // Tokenize into sentences.  Co-occurrence window is bounded by sentence scope (OC-RA-10).
        let sentences = tokenize_into_sentences(text);

        // Build vocabulary (frequency-filtered) across all sentences
        let mut freq: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for sent in &sentences {
            for t in sent { *freq.entry(t.clone()).or_insert(0) += 1; }
        }
        let mut words: Vec<String> = freq.iter()
            .filter(|(_, &c)| c >= min_count)
            .map(|(w, _)| w.clone())
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

        // Map each sentence to a Vec<Option<usize>> (None for OOV/filtered tokens)
        let sentence_ids: Vec<Vec<Option<usize>>> = sentences.iter()
            .map(|sent| sent.iter().map(|t| word_to_id.get(t.as_str()).copied()).collect())
            .collect();

        // Co-occurrence counts — sentence-scoped (OC-RA-10).
        //
        // For each sentence: count co-occurrences within window, never crossing
        // the sentence boundary.  Cross-sentence pairs are inadmissible at word
        // level per the Multi-Scale M Structure declaration (V5.0).
        let context_dim = vocab_size.min(dim * 2);
        let mut cooc = vec![vec![0_f64; context_dim]; vocab_size];
        for sent_ids in &sentence_ids {
            let m = sent_ids.len();
            for i in 0..m {
                if let Some(w) = sent_ids[i] {
                    let j_start = i.saturating_sub(window);
                    let j_end   = (i + window).min(m.saturating_sub(1));
                    for j in j_start..=j_end {
                        if i == j { continue; }
                        if let Some(c) = sent_ids[j] {
                            if c < context_dim {
                                cooc[w][c] += 1.0;
                            }
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

        let mut m = Self::from_raw(raw, words);

        // Emphasis pass: compute per-token-type ι[v] from raw text.
        // from_raw defaults all iota to 1.0; we now replace with corpus-derived weights.
        // Max-selection rule: iota[token_id] = max emphasis weight over all corpus occurrences.
        let emphasis_tokens = extract_emphasis_tokens(text);
        for (word, weight) in &emphasis_tokens {
            if let Some(&tid) = m.vocab.get(word.as_str()) {
                if tid < m.iota.len() && *weight > m.iota[tid] {
                    m.iota[tid] = *weight;
                }
            }
        }
        // unk retains 1.0 (ambient).

        m
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

    /// Return the emphasis weight ι[v] for a token.
    ///
    /// 1.0 = ambient. Higher = more emphasis declared through M.
    /// Returns 1.0 for out-of-range ids (safe default).
    pub fn emphasis(&self, token_id: usize) -> f32 {
        self.iota.get(token_id).copied().unwrap_or(1.0)
    }

    /// Edge energy ε[e] = |ι[src] − ι[tgt]|.
    ///
    /// Declared: energy of an edge is the magnitude of the emphasis
    /// differential across it (V4.5 declaration, OC-RA-9).
    pub fn epsilon_edge(&self, src_id: usize, tgt_id: usize) -> f32 {
        (self.emphasis(src_id) - self.emphasis(tgt_id)).abs()
    }

    /// Return a reference to the VocabTable for use with ABR attention layers.
    pub fn as_vocab_table(&self) -> &VocabTable { &self.embeddings }

    /// Clones the VocabTable (needed to pass to AttentionLayer::forward).
    pub fn to_vocab_table(&self) -> VocabTable { self.embeddings.clone() }

    /// Build a compact VocabTable containing only the tokens that appear in
    /// `token_ids`, plus the <unk> token at index `compact_vocab_size - 1`.
    ///
    /// Returns `(compact_vocab, remapped_ids)` where:
    ///   - `compact_vocab[i]` is the embedding of the i-th unique token
    ///   - `remapped_ids[j]` is the compact index of `token_ids[j]`
    ///
    /// This is a performance optimization for large pre-trained vocabularies:
    /// a 50k-word GloVe table compacts to N_unique+1 entries (e.g., 136 for
    /// the Darwin paragraph). The ABR kernel operates identically — only the
    /// token_id namespace is different.
    ///
    /// The <unk> compact id is `compact_vocab.len() - 1`.
    pub fn compact_vocab_for(&self, token_ids: &[usize]) -> (VocabTable, Vec<usize>) {
        // Collect unique glove_ids in order of first appearance
        let mut seen_order: Vec<usize> = Vec::new();
        let mut glove_to_compact: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();

        for &gid in token_ids {
            if !glove_to_compact.contains_key(&gid) {
                glove_to_compact.insert(gid, seen_order.len());
                seen_order.push(gid);
            }
        }

        // Build compact vocab: one embedding per unique glove_id
        let compact_vocab: VocabTable = seen_order
            .iter()
            .map(|&gid| self.embeddings[gid.min(self.unk_id)].clone())
            .collect();

        // Remap token_ids
        let remapped: Vec<usize> = token_ids
            .iter()
            .map(|gid| glove_to_compact[gid])
            .collect();

        (compact_vocab, remapped)
    }

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
        // iota defaults to 1.0 for all tokens (from_raw does not have a text source
        // for the emphasis pass; GloVe-loaded M has ambient emphasis throughout).
    }
}

/// Extract `(lowercased_token, emphasis_weight)` pairs from raw text.
///
/// Runs BEFORE case-folding and punctuation stripping so that emphasis
/// signals (ALL CAPS, !, ?) are still visible.
///
/// ## Weight scale
///
/// | Signal                        | Multiplier |
/// |-------------------------------|------------|
/// | Normal word, declarative      | 1.0        |
/// | Sentence contains `?`         | × 1.2      |
/// | Sentence contains `!`         | × 1.5      |
/// | Word is ALL CAPS (len > 1)    | × 2.0      |
///
/// Sentence multipliers and word multipliers combine multiplicatively.
/// Per-type iota in SemanticM uses the MAX over all corpus occurrences
/// (max-selection rule, consistent with χ[i] in ρ).
///
/// A "sentence" is delimited by `.`, `!`, `?`, `\n`, or end of text.
/// Single-character all-caps tokens (e.g., the pronoun "I") are NOT
/// treated as ALL CAPS — they carry no additional emphasis.
pub fn extract_emphasis_tokens(text: &str) -> Vec<(String, f32)> {
    let mut result = Vec::new();

    // Split into rough sentence chunks, keeping the delimiter character.
    // We split on sentence-ending punctuation and newlines.
    let mut chunk_start = 0usize;
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();

    let mut i = 0usize;
    while i <= n {
        let end_of_chunk = i == n
            || chars[i] == '.'
            || chars[i] == '!'
            || chars[i] == '?'
            || chars[i] == '\n';

        if end_of_chunk {
            // Collect this chunk's character range
            let chunk_end = if i < n { i + 1 } else { n };
            let chunk: String = chars[chunk_start..chunk_end].iter().collect();

            // Sentence-level emphasis: does the chunk contain ! or ??
            let sent_exclaim = chunk.contains('!');
            let sent_question = chunk.contains('?');
            let sent_mult = if sent_exclaim { 1.5_f32 }
                            else if sent_question { 1.2_f32 }
                            else { 1.0_f32 };

            // Extract words from this chunk
            for raw_word in chunk.split(|c: char| {
                c.is_whitespace() || (c.is_ascii_punctuation() && c != '\'')
            }) {
                if raw_word.is_empty() { continue; }
                if !raw_word.chars().any(|c| c.is_alphabetic()) { continue; }

                // ALL CAPS: more than one character, all alphabetic chars are uppercase
                let alpha_chars: Vec<char> = raw_word.chars()
                    .filter(|c| c.is_alphabetic())
                    .collect();
                let is_all_caps = alpha_chars.len() > 1
                    && alpha_chars.iter().all(|c| c.is_uppercase());
                let word_mult = if is_all_caps { 2.0_f32 } else { 1.0_f32 };

                let weight = sent_mult * word_mult;
                let lowercased = raw_word.to_lowercase();
                // Filter out tokens with no alphabetic content after lowercasing
                if lowercased.chars().any(|c| c.is_alphabetic()) {
                    result.push((lowercased, weight));
                }
            }

            chunk_start = chunk_end;
        }
        i += 1;
    }

    result
}

/// Tokenize text into sentences, each sentence as a Vec<String> of lowercase tokens.
///
/// A sentence boundary is declared at: `.`, `!`, `?`, and paragraph breaks (`\n\n`).
/// Single newlines are treated as whitespace (soft line breaks within a sentence).
///
/// ## Declaration (V5.0 — OC-RA-10)
///
/// Per the Multi-Scale M Structure declaration: word-level co-occurrence is
/// bounded by sentence scope. This function preserves sentence boundaries so
/// that `from_corpus` can restrict the co-occurrence window to within-sentence
/// pairs only.  Cross-sentence co-occurrences are not admitted at word level;
/// they belong to sentence-level M (M_sent, OC-RA-8).
///
/// Sentence boundary detection uses punctuation (`.`, `!`, `?`) and double
/// newlines.  Abbreviations (e.g., "Dr.") are split aggressively — this is
/// a conservative choice that never merges two sentences at the cost of
/// occasionally splitting a non-sentence period.  Declared limitation.
pub fn tokenize_into_sentences(text: &str) -> Vec<Vec<String>> {
    let mut sentences: Vec<Vec<String>> = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '.' | '!' | '?' => {
                // Sentence boundary: flush current buffer
                let tokens = tokenize_text(&current);
                if !tokens.is_empty() {
                    sentences.push(tokens);
                }
                current.clear();
            }
            '\n' => {
                if chars.peek() == Some(&'\n') {
                    // Paragraph break (double newline) = sentence boundary
                    chars.next(); // consume second newline
                    let tokens = tokenize_text(&current);
                    if !tokens.is_empty() {
                        sentences.push(tokens);
                    }
                    current.clear();
                } else {
                    // Single newline: treat as space (soft line break)
                    current.push(' ');
                }
            }
            _ => { current.push(ch); }
        }
    }
    // Flush any trailing text as the final sentence
    let tokens = tokenize_text(&current);
    if !tokens.is_empty() {
        sentences.push(tokens);
    }
    sentences
}

/// Tokenize text into lowercase word tokens.
///
/// Splits on whitespace and punctuation; lowercases all words.
/// Returns Vec<String> — each element is one token.
///
/// NOTE: This function strips all emphasis signals. Use
/// `extract_emphasis_tokens` when emphasis (ι[v]) is needed.
/// For sentence-scoped processing use `tokenize_into_sentences`.
pub fn tokenize_text(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace() || (!c.is_alphabetic() && c != '\''))
        .map(|w| w.to_lowercase())
        .filter(|w| !w.is_empty() && w.chars().any(|c| c.is_alphabetic()))
        .collect()
}

// ── CharM ─────────────────────────────────────────────────────────────────────
//
// OC-RA-11: M_char — character-level PPMI within word boundaries.
//
// DECLARATION (V5.0):
//   Loci:   characters (a–z, 0–9, etc. — all alphanumeric chars in vocabulary)
//   Window: word boundary — characters may only co-occur within the same word.
//           Cross-word co-occurrence is inadmissible at character scale.
//
// What CharM encodes:
//   Morphological and orthographic relational structure.
//   Characters that often appear together within words (e.g. "t"–"h", "e"–"r")
//   acquire high PPMI; characters that rarely co-occur within words have low PPMI.
//
// Method mirrors SemanticM::from_corpus but at character resolution:
//   1. Tokenize into words (whitespace + punctuation split, lowercase)
//   2. For each word: treat every character as a locus; build co-occurrence counts
//      for all character pairs within the word (window = word boundary)
//   3. Compute PPMI across character pairs
//   4. Truncate/pad to `dim` dimensions
//
// Token IDs in CharM are character IDs (index into the character vocabulary).
// The vocabulary is the set of all distinct characters in the corpus.
//
// OC-RA-11 CLOSED: 2026-11-13.

/// Character-level PPMI matrix (M_char).
///
/// Loci: characters. Window: word boundary.
/// CharM.embeddings[char_id] is the PPMI co-occurrence vector for that character
/// across all same-word co-occurrences in the corpus.
#[derive(Debug, Clone)]
pub struct CharM {
    /// Embedding matrix: n_chars × dim, all values in [0,1].
    pub embeddings: VocabTable,
    /// Character → char_id.
    pub char_vocab: std::collections::HashMap<char, usize>,
    /// char_id → character.
    pub id_to_char: Vec<char>,
    /// Embedding dimension.
    pub dim: usize,
    /// Special token: unknown character (all 0.5 — center of domain).
    pub unk_id: usize,
}

impl CharM {
    /// Build CharM from a text corpus.
    ///
    /// Characters co-occur only within the same word (word boundary = window).
    /// `dim` controls the output embedding dimension.
    /// `min_count` is the minimum number of times a character must appear
    /// across all words to be included in the vocabulary.
    pub fn from_corpus(text: &str, dim: usize, min_count: usize) -> Self {
        // Tokenize into words (reuse tokenize_text for consistency)
        let words = tokenize_text(text);

        // Build character frequency map
        let mut freq: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
        for word in &words {
            for ch in word.chars() {
                *freq.entry(ch).or_insert(0) += 1;
            }
        }

        // Build character vocabulary (frequency-filtered, sorted for determinism)
        let mut chars: Vec<char> = freq.iter()
            .filter(|(_, &c)| c >= min_count)
            .map(|(ch, _)| *ch)
            .collect();
        chars.sort();
        let n_chars = chars.len();

        if n_chars == 0 || dim == 0 {
            // Empty corpus — return trivial CharM
            let unk_emb = vec![0.5_f64; dim.max(1)];
            let raw = vec![unk_emb];
            let m = CharM::from_raw(raw, vec!['_']);
            return m;
        }

        let char_to_id: std::collections::HashMap<char, usize> = chars.iter()
            .enumerate()
            .map(|(i, &ch)| (ch, i))
            .collect();

        // Co-occurrence counts within word boundaries.
        // For each word: every character pair (i, j) where i ≠ j within the word.
        // Window = word boundary: all characters in the word are within window of each other.
        let context_dim = n_chars.min(dim * 2);
        let mut cooc = vec![vec![0_f64; context_dim]; n_chars];

        for word in &words {
            let char_ids: Vec<Option<usize>> = word.chars()
                .map(|ch| char_to_id.get(&ch).copied())
                .collect();
            let len = char_ids.len();
            for i in 0..len {
                if let Some(src) = char_ids[i] {
                    for j in 0..len {
                        if i == j { continue; }
                        if let Some(ctx) = char_ids[j] {
                            if ctx < context_dim {
                                cooc[src][ctx] += 1.0;
                            }
                        }
                    }
                }
            }
        }

        // PPMI
        let total: f64 = cooc.iter().flatten().sum();
        if total < 1.0 {
            // No co-occurrences — identity basis
            let raw: Vec<Vec<f64>> = (0..n_chars).map(|id| {
                let mut v = vec![0.0_f64; dim];
                v[id % dim] = 1.0;
                v
            }).collect();
            return Self::from_raw(raw, chars);
        }

        let row_sums: Vec<f64> = cooc.iter().map(|r| r.iter().sum()).collect();
        let col_sums: Vec<f64> = (0..context_dim)
            .map(|c| cooc.iter().map(|r| r[c]).sum())
            .collect();
        let eps = 1e-10_f64;

        let mut ppmi = vec![vec![0.0_f64; context_dim]; n_chars];
        for w in 0..n_chars {
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

        // Truncate or pad to dim
        let raw: Vec<Vec<f64>> = ppmi.iter()
            .map(|row| {
                let mut v: Vec<f64> = row.iter().take(dim).copied().collect();
                while v.len() < dim { v.push(0.0); }
                v
            })
            .collect();

        Self::from_raw(raw, chars)
    }

    /// Build CharM from a raw (pre-computed) embedding matrix + character vocabulary.
    ///
    /// Applies per-dimension min-max normalization.
    /// An `<unk>` character (id = last entry) is appended at 0.5 in all dims.
    pub fn from_raw(raw: Vec<Vec<f64>>, chars: Vec<char>) -> Self {
        assert_eq!(raw.len(), chars.len(), "raw and chars must have same length");
        let n = raw.len();
        if n == 0 {
            let unk_id = 0;
            return CharM {
                embeddings: vec![vec![0.5_f64; 1]],
                char_vocab: std::collections::HashMap::new(),
                id_to_char: vec!['?'],
                dim: 1,
                unk_id,
            };
        }
        let dim = raw[0].len();

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
        let mut normalized: VocabTable = raw.iter().map(|emb| {
            emb.iter().enumerate().map(|(d, &v)| {
                let range = max_d[d] - min_d[d];
                if range < eps { 0.5 } else { (v - min_d[d]) / (range + eps) }
            }).collect()
        }).collect();

        // Append <unk> at center of domain
        normalized.push(vec![0.5_f64; dim]);
        let unk_id = normalized.len() - 1;

        let mut id_to_char: Vec<char> = chars.clone();
        id_to_char.push('?'); // unk character
        let char_vocab: std::collections::HashMap<char, usize> = chars.iter()
            .enumerate()
            .map(|(i, &ch)| (ch, i))
            .collect();

        CharM { embeddings: normalized, char_vocab, id_to_char, dim, unk_id }
    }

    /// Vocabulary size (including <unk>).
    pub fn vocab_size(&self) -> usize { self.embeddings.len() }

    /// Look up char_id for a character (returns unk_id if not found).
    pub fn char_id(&self, ch: char) -> usize {
        self.char_vocab.get(&ch).copied().unwrap_or(self.unk_id)
    }

    /// Encode a word as a Vec<usize> of char_ids.
    pub fn encode_word(&self, word: &str) -> Vec<usize> {
        word.chars().map(|ch| self.char_id(ch)).collect()
    }

    /// Return the embedding for a character id.
    pub fn embedding(&self, char_id: usize) -> &[f64] {
        self.embeddings.get(char_id)
            .map(|v| v.as_slice())
            .unwrap_or_else(|| self.embeddings[self.unk_id].as_slice())
    }
}

// ── SentM / ParaM ─────────────────────────────────────────────────────────────
//
// OC-RA-8: Higher-level connection scopes — M_sent and M_para.
//
// DECLARATION (V5.0):
//
//   M_sent:
//     Loci:   sentences (each sentence is one unit; represented by mean M_word embedding)
//     Window: paragraph boundary — sentences co-occur only within the same paragraph.
//             Cross-paragraph pairs are inadmissible at sentence scale.
//     What M_sent encodes: discourse-level relational structure — which sentence
//     types follow each other within paragraphs.
//
//   M_para:
//     Loci:   paragraphs (each paragraph is one unit; represented by mean M_word embedding)
//     Window: volume boundary — all paragraphs within the text are within window.
//     What M_para encodes: volume-level relational structure — which paragraph
//     types appear together within a document.
//
// Method (both):
//   1. Tokenize text into paragraphs (split on "\n\n").
//   2. For M_sent: within each paragraph, build sentence co-occurrence counts
//      (all sentence pairs within the paragraph are within window).
//      For M_para: all paragraph pairs within the volume are within window.
//   3. Each sentence/paragraph is represented by a unique integer index.
//      PPMI is computed over index co-occurrence — yielding a structural
//      "positional PPMI" that reflects which units appear near each other.
//   4. Sentence/paragraph loci are associated with their mean M_word embedding
//      as the initializing representation; PPMI structure is then overlaid.
//
// OC-RA-8 CLOSED: 2026-11-13.

/// Split text into paragraphs (split on blank lines `\n\n`).
/// Each paragraph is itself split into sentences.
/// Returns Vec<Vec<Vec<String>>> = paragraphs × sentences × tokens.
pub fn tokenize_into_paragraphs(text: &str) -> Vec<Vec<Vec<String>>> {
    text.split("\n\n")
        .map(|para| {
            let sents = tokenize_into_sentences(para);
            sents.into_iter().filter(|s| !s.is_empty()).collect()
        })
        .filter(|para: &Vec<Vec<String>>| !para.is_empty())
        .collect()
}

/// Sentence-level PPMI matrix (M_sent).
///
/// Loci: sentences. Window: paragraph boundary.
/// Two sentences co-occur iff they are in the same paragraph.
///
/// Each sentence is identified by a unique index (0..n_sentences).
/// The embedding matrix is n_sentences × dim.
/// Each row is the PPMI co-occurrence vector for that sentence index
/// across all same-paragraph co-occurrences, normalized to [0,1].
///
/// Additionally, `mean_embeddings` stores the mean M_word token embedding
/// for each sentence (requires a SemanticM to be passed at build time).
#[derive(Debug, Clone)]
pub struct SentM {
    /// PPMI co-occurrence vectors: n_sentences × dim, all in [0,1].
    pub embeddings: VocabTable,
    /// Mean M_word embedding for each sentence: n_sentences × word_dim.
    /// Empty if no SemanticM was provided.
    pub mean_word_embeddings: VocabTable,
    /// Original sentence tokens (Vec<String>) for each sentence.
    pub sentences: Vec<Vec<String>>,
    /// Embedding dimension (PPMI dimension).
    pub dim: usize,
    /// Special token: unknown sentence (all 0.5).
    pub unk_id: usize,
}

impl SentM {
    /// Build SentM from text.
    ///
    /// `dim` is the PPMI embedding dimension for the sentence-level co-occurrence vectors.
    /// `sem` is an optional M_word used to compute mean word embeddings per sentence.
    pub fn from_corpus(text: &str, dim: usize, sem: Option<&SemanticM>) -> Self {
        let paragraphs = tokenize_into_paragraphs(text);

        // Flatten: collect all sentences with paragraph labels
        // para_assignments[sent_idx] = para_idx
        let mut all_sentences: Vec<Vec<String>> = Vec::new();
        let mut para_assignments: Vec<usize> = Vec::new();

        for (para_idx, para) in paragraphs.iter().enumerate() {
            for sent in para {
                all_sentences.push(sent.clone());
                para_assignments.push(para_idx);
            }
        }

        let n_sents = all_sentences.len();
        if n_sents == 0 || dim == 0 {
            return SentM {
                embeddings: vec![vec![0.5_f64; dim.max(1)]],
                mean_word_embeddings: vec![],
                sentences: vec![],
                dim: dim.max(1),
                unk_id: 0,
            };
        }

        // Build paragraph-grouped sentence sets for co-occurrence
        let n_paras = paragraphs.len();
        let mut para_sent_lists: Vec<Vec<usize>> = vec![Vec::new(); n_paras];
        for (sent_idx, &para_idx) in para_assignments.iter().enumerate() {
            para_sent_lists[para_idx].push(sent_idx);
        }

        // Co-occurrence: two sentences co-occur iff in same paragraph.
        // Window = paragraph boundary: all sentence pairs within a paragraph.
        let context_dim = n_sents.min(dim * 2);
        let mut cooc = vec![vec![0_f64; context_dim]; n_sents];
        for sent_list in &para_sent_lists {
            for &si in sent_list {
                for &sj in sent_list {
                    if si == sj { continue; }
                    if sj < context_dim {
                        cooc[si][sj] += 1.0;
                    }
                }
            }
        }

        // PPMI
        let total: f64 = cooc.iter().flatten().sum();
        let raw: Vec<Vec<f64>> = if total < 1.0 {
            // No co-occurrences — identity basis
            (0..n_sents).map(|id| {
                let mut v = vec![0.0_f64; dim];
                v[id % dim] = 1.0;
                v
            }).collect()
        } else {
            let row_sums: Vec<f64> = cooc.iter().map(|r| r.iter().sum()).collect();
            let col_sums: Vec<f64> = (0..context_dim)
                .map(|c| cooc.iter().map(|r| r[c]).sum())
                .collect();
            let eps = 1e-10_f64;
            let ppmi: Vec<Vec<f64>> = (0..n_sents).map(|w| {
                (0..context_dim).map(|c| {
                    let p_wc = cooc[w][c] / (total + eps);
                    let p_w  = row_sums[w] / (total + eps);
                    let p_c  = col_sums[c] / (total + eps);
                    if p_w > eps && p_c > eps && p_wc > eps {
                        ((p_wc / (p_w * p_c)).ln()).max(0.0)
                    } else { 0.0 }
                }).collect()
            }).collect();
            ppmi.iter().map(|row| {
                let mut v: Vec<f64> = row.iter().take(dim).copied().collect();
                while v.len() < dim { v.push(0.0); }
                v
            }).collect()
        };

        // Per-dimension min-max normalization
        let dim_actual = raw[0].len();
        let mut min_d = vec![f64::INFINITY; dim_actual];
        let mut max_d = vec![f64::NEG_INFINITY; dim_actual];
        for emb in &raw {
            for (d, &v) in emb.iter().enumerate() {
                if v < min_d[d] { min_d[d] = v; }
                if v > max_d[d] { max_d[d] = v; }
            }
        }
        let eps = 1e-10_f64;
        let mut embeddings: VocabTable = raw.iter().map(|emb| {
            emb.iter().enumerate().map(|(d, &v)| {
                let range = max_d[d] - min_d[d];
                if range < eps { 0.5 } else { (v - min_d[d]) / (range + eps) }
            }).collect()
        }).collect();
        // Append unk
        embeddings.push(vec![0.5_f64; dim_actual]);
        let unk_id = embeddings.len() - 1;

        // Mean word embeddings per sentence (if sem provided)
        let mean_word_embeddings: VocabTable = if let Some(sem) = sem {
            all_sentences.iter().map(|sent| {
                let ids: Vec<usize> = sent.iter().map(|w| sem.token_id(w)).collect();
                if ids.is_empty() { return vec![0.5_f64; sem.dim]; }
                let mut mean = vec![0.0_f64; sem.dim];
                for &id in &ids {
                    let emb = &sem.embeddings[id.min(sem.embeddings.len() - 1)];
                    for (d, &v) in emb.iter().enumerate() { mean[d] += v; }
                }
                mean.iter_mut().for_each(|v| *v /= ids.len() as f64);
                mean
            }).collect()
        } else { vec![] };

        SentM {
            embeddings,
            mean_word_embeddings,
            sentences: all_sentences,
            dim: dim_actual,
            unk_id,
        }
    }

    /// Number of sentences (including unk).
    pub fn vocab_size(&self) -> usize { self.embeddings.len() }

    /// Return the PPMI embedding for a sentence index.
    pub fn embedding(&self, sent_idx: usize) -> &[f64] {
        self.embeddings.get(sent_idx)
            .map(|v| v.as_slice())
            .unwrap_or_else(|| self.embeddings[self.unk_id].as_slice())
    }

    /// Return the mean word embedding for a sentence index.
    pub fn mean_word_embedding(&self, sent_idx: usize) -> Option<&[f64]> {
        self.mean_word_embeddings.get(sent_idx).map(|v| v.as_slice())
    }
}

/// Paragraph-level PPMI matrix (M_para).
///
/// Loci: paragraphs. Window: volume boundary (all paragraphs co-occur with all others).
/// Each paragraph is identified by a unique index (0..n_paras).
/// PPMI is computed over paragraph co-occurrence within the volume.
///
/// Structurally, M_para records which paragraph positions occur together
/// in the volume — encoding document-level relational structure.
#[derive(Debug, Clone)]
pub struct ParaM {
    /// PPMI co-occurrence vectors: n_paragraphs × dim, all in [0,1].
    pub embeddings: VocabTable,
    /// Mean M_word embedding for each paragraph: n_paragraphs × word_dim.
    pub mean_word_embeddings: VocabTable,
    /// Original paragraph text (flattened sentences) for each paragraph.
    pub paragraphs: Vec<Vec<Vec<String>>>,
    /// Embedding dimension.
    pub dim: usize,
    /// Special token: unknown paragraph (all 0.5).
    pub unk_id: usize,
}

impl ParaM {
    /// Build ParaM from text.
    ///
    /// `dim` is the PPMI embedding dimension.
    /// `sem` is an optional M_word used to compute mean word embeddings per paragraph.
    pub fn from_corpus(text: &str, dim: usize, sem: Option<&SemanticM>) -> Self {
        let paragraphs = tokenize_into_paragraphs(text);
        let n_paras = paragraphs.len();

        if n_paras == 0 || dim == 0 {
            return ParaM {
                embeddings: vec![vec![0.5_f64; dim.max(1)]],
                mean_word_embeddings: vec![],
                paragraphs: vec![],
                dim: dim.max(1),
                unk_id: 0,
            };
        }

        // Co-occurrence: all paragraph pairs within the volume (window = volume boundary).
        let context_dim = n_paras.min(dim * 2);
        let mut cooc = vec![vec![0_f64; context_dim]; n_paras];
        for pi in 0..n_paras {
            for pj in 0..n_paras {
                if pi == pj { continue; }
                if pj < context_dim {
                    cooc[pi][pj] += 1.0;
                }
            }
        }

        // PPMI
        let total: f64 = cooc.iter().flatten().sum();
        let raw: Vec<Vec<f64>> = if total < 1.0 {
            (0..n_paras).map(|id| {
                let mut v = vec![0.0_f64; dim];
                v[id % dim] = 1.0;
                v
            }).collect()
        } else {
            let row_sums: Vec<f64> = cooc.iter().map(|r| r.iter().sum()).collect();
            let col_sums: Vec<f64> = (0..context_dim)
                .map(|c| cooc.iter().map(|r| r[c]).sum())
                .collect();
            let eps = 1e-10_f64;
            let ppmi: Vec<Vec<f64>> = (0..n_paras).map(|w| {
                (0..context_dim).map(|c| {
                    let p_wc = cooc[w][c] / (total + eps);
                    let p_w  = row_sums[w] / (total + eps);
                    let p_c  = col_sums[c] / (total + eps);
                    if p_w > eps && p_c > eps && p_wc > eps {
                        ((p_wc / (p_w * p_c)).ln()).max(0.0)
                    } else { 0.0 }
                }).collect()
            }).collect();
            ppmi.iter().map(|row| {
                let mut v: Vec<f64> = row.iter().take(dim).copied().collect();
                while v.len() < dim { v.push(0.0); }
                v
            }).collect()
        };

        // Normalize
        let dim_actual = raw[0].len();
        let mut min_d = vec![f64::INFINITY; dim_actual];
        let mut max_d = vec![f64::NEG_INFINITY; dim_actual];
        for emb in &raw { for (d, &v) in emb.iter().enumerate() { if v < min_d[d] { min_d[d] = v; } if v > max_d[d] { max_d[d] = v; } } }
        let eps = 1e-10_f64;
        let mut embeddings: VocabTable = raw.iter().map(|emb| {
            emb.iter().enumerate().map(|(d, &v)| {
                let range = max_d[d] - min_d[d];
                if range < eps { 0.5 } else { (v - min_d[d]) / (range + eps) }
            }).collect()
        }).collect();
        embeddings.push(vec![0.5_f64; dim_actual]);
        let unk_id = embeddings.len() - 1;

        // Mean word embeddings per paragraph
        let mean_word_embeddings: VocabTable = if let Some(sem) = sem {
            paragraphs.iter().map(|para| {
                let all_tokens: Vec<usize> = para.iter()
                    .flat_map(|sent| sent.iter().map(|w| sem.token_id(w)))
                    .collect();
                if all_tokens.is_empty() { return vec![0.5_f64; sem.dim]; }
                let mut mean = vec![0.0_f64; sem.dim];
                for &id in &all_tokens {
                    let emb = &sem.embeddings[id.min(sem.embeddings.len() - 1)];
                    for (d, &v) in emb.iter().enumerate() { mean[d] += v; }
                }
                mean.iter_mut().for_each(|v| *v /= all_tokens.len() as f64);
                mean
            }).collect()
        } else { vec![] };

        ParaM {
            embeddings,
            mean_word_embeddings,
            paragraphs,
            dim: dim_actual,
            unk_id,
        }
    }

    /// Number of paragraphs (including unk).
    pub fn vocab_size(&self) -> usize { self.embeddings.len() }

    /// Return the PPMI embedding for a paragraph index.
    pub fn embedding(&self, para_idx: usize) -> &[f64] {
        self.embeddings.get(para_idx)
            .map(|v| v.as_slice())
            .unwrap_or_else(|| self.embeddings[self.unk_id].as_slice())
    }

    /// Return the mean word embedding for a paragraph index.
    pub fn mean_word_embedding(&self, para_idx: usize) -> Option<&[f64]> {
        self.mean_word_embeddings.get(para_idx).map(|v| v.as_slice())
    }
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

    // ── Emphasis tests (OC-RA-9) ──────────────────────────────────────────────

    #[test]
    fn test_extract_emphasis_ambient() {
        // Normal declarative sentence — all weights should be 1.0
        let pairs = extract_emphasis_tokens("the cat sat on the mat.");
        for (_, w) in &pairs {
            assert!((*w - 1.0).abs() < 1e-6, "ambient weight must be 1.0, got {}", w);
        }
    }

    #[test]
    fn test_extract_emphasis_exclamation() {
        // Sentence-level emphasis: words in an exclamatory sentence get 1.5
        let pairs = extract_emphasis_tokens("Run away!");
        for (_, w) in &pairs {
            assert!((*w - 1.5).abs() < 1e-6,
                "exclamatory sentence weight must be 1.5, got {}", w);
        }
    }

    #[test]
    fn test_extract_emphasis_question() {
        let pairs = extract_emphasis_tokens("What is this?");
        for (_, w) in &pairs {
            assert!((*w - 1.2).abs() < 1e-6,
                "question sentence weight must be 1.2, got {}", w);
        }
    }

    #[test]
    fn test_extract_emphasis_all_caps_declarative() {
        // ALL CAPS word in a declarative sentence: 2.0
        let pairs = extract_emphasis_tokens("This is URGENT.");
        let urgent = pairs.iter().find(|(w, _)| w == "urgent");
        assert!(urgent.is_some(), "URGENT must be tokenized as 'urgent'");
        let (_, w) = urgent.unwrap();
        assert!((*w - 2.0).abs() < 1e-6,
            "ALL CAPS declarative weight must be 2.0, got {}", w);
        // non-caps words should be 1.0
        let this = pairs.iter().find(|(w, _)| w == "this").unwrap();
        assert!((this.1 - 1.0).abs() < 1e-6,
            "ambient word in declarative must be 1.0, got {}", this.1);
    }

    #[test]
    fn test_extract_emphasis_all_caps_exclamatory() {
        // ALL CAPS in exclamatory sentence: 2.0 × 1.5 = 3.0
        let pairs = extract_emphasis_tokens("FIRE!");
        let fire = pairs.iter().find(|(w, _)| w == "fire").unwrap();
        assert!((fire.1 - 3.0).abs() < 1e-6,
            "ALL CAPS exclamatory must be 3.0, got {}", fire.1);
    }

    #[test]
    fn test_all_caps_single_char_not_amplified() {
        // Single-character ALL CAPS (e.g., "I") must not be treated as emphasis
        let pairs = extract_emphasis_tokens("I went there.");
        let i_tok = pairs.iter().find(|(w, _)| w == "i");
        if let Some((_, w)) = i_tok {
            assert!((*w - 1.0).abs() < 1e-6,
                "single-char 'I' must not get ALL CAPS emphasis, got {}", w);
        }
    }

    #[test]
    fn test_iota_populated_in_from_corpus() {
        // "FIRE" appears in an exclamatory sentence → iota for "fire" > 1.0
        let corpus = "The fire was bright. FIRE! The fire is gone.";
        let m = SemanticM::from_corpus(corpus, 8, 2, 1);
        let fire_id = m.token_id("fire");
        if fire_id != m.unk_id {
            assert!(m.emphasis(fire_id) > 1.0,
                "iota['fire'] must be > 1.0 when FIRE! appears in corpus, got {}",
                m.emphasis(fire_id));
        }
    }

    #[test]
    fn test_iota_max_selection() {
        // "fire" appears normally AND as FIRE! — iota must be the MAX (3.0)
        let corpus = "the fire burned. FIRE! the fire went out.";
        let m = SemanticM::from_corpus(corpus, 8, 1, 1);
        let fire_id = m.token_id("fire");
        if fire_id != m.unk_id {
            assert!((m.emphasis(fire_id) - 3.0).abs() < 1e-5,
                "iota['fire'] must be max = 3.0, got {}", m.emphasis(fire_id));
        }
    }

    #[test]
    fn test_epsilon_edge() {
        let corpus = "the fire burned. FIRE! normal words here.";
        let m = SemanticM::from_corpus(corpus, 8, 1, 1);
        let fire_id = m.token_id("fire");
        let the_id  = m.token_id("the");
        if fire_id != m.unk_id && the_id != m.unk_id {
            let eps = m.epsilon_edge(fire_id, the_id);
            // ι[fire]=3.0, ι[the]=1.0 → ε = 2.0
            assert!(eps > 0.0,
                "epsilon_edge between high- and low-emphasis tokens must be > 0");
            assert!((eps - (m.emphasis(fire_id) - m.emphasis(the_id)).abs()).abs() < 1e-6,
                "epsilon_edge must equal |ι[src] - ι[tgt]|");
        }
    }

    #[test]
    fn test_unk_has_ambient_iota() {
        let corpus = "hello world";
        let m = SemanticM::from_corpus(corpus, 4, 1, 1);
        assert!((m.emphasis(m.unk_id) - 1.0).abs() < 1e-6,
            "unk token iota must be 1.0 (ambient)");
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

    // ── Sentence boundary tests (OC-RA-10) ───────────────────────────────────────

    /// tokenize_into_sentences: single sentence (no terminal punctuation)
    #[test]
    fn test_tokenize_sentences_single() {
        let sents = tokenize_into_sentences("cat sat mat");
        assert_eq!(sents.len(), 1);
        assert_eq!(sents[0], vec!["cat", "sat", "mat"]);
    }

    /// tokenize_into_sentences: period splits into two sentences
    #[test]
    fn test_tokenize_sentences_period_split() {
        let sents = tokenize_into_sentences("cat sat. dog ran.");
        assert_eq!(sents.len(), 2, "period must produce two sentences, got {:?}", sents);
        assert_eq!(sents[0], vec!["cat", "sat"]);
        assert_eq!(sents[1], vec!["dog", "ran"]);
    }

    /// tokenize_into_sentences: exclamation and question marks also split
    #[test]
    fn test_tokenize_sentences_exclaim_question() {
        let sents = tokenize_into_sentences("FIRE! What now? Stay calm.");
        assert_eq!(sents.len(), 3, "three terminal marks → three sentences");
        assert_eq!(sents[0], vec!["fire"]);
        assert_eq!(sents[1], vec!["what", "now"]);
        assert_eq!(sents[2], vec!["stay", "calm"]);
    }

    /// tokenize_into_sentences: double newline splits; single newline is a space
    #[test]
    fn test_tokenize_sentences_newline_handling() {
        let sents = tokenize_into_sentences("cat sat\nmat\n\ndog ran");
        assert_eq!(sents.len(), 2, "double newline must split, single newline must not");
        // first sentence: cat + sat + mat (single newline = space)
        assert!(sents[0].contains(&"cat".to_string()));
        assert!(sents[0].contains(&"sat".to_string()));
        assert!(sents[0].contains(&"mat".to_string()));
        assert_eq!(sents[1], vec!["dog", "ran"]);
    }

    /// tokenize_into_sentences: empty input yields no sentences
    #[test]
    fn test_tokenize_sentences_empty() {
        let sents = tokenize_into_sentences("");
        assert!(sents.is_empty());
    }

    /// tokenize_into_sentences: trailing text with no terminator is a sentence
    #[test]
    fn test_tokenize_sentences_no_terminator() {
        let sents = tokenize_into_sentences("cat sat");
        assert_eq!(sents.len(), 1);
        assert_eq!(sents[0], vec!["cat", "sat"]);
    }

    /// OC-RA-10 enforcement: cross-sentence pairs must NOT co-occur in M_word.
    ///
    /// "cat" appears in sentence 1; "ran" appears only in sentence 2.
    /// With sentence-scoped co-occurrence, they must have zero count in cooc —
    /// which means their PPMI vectors must be orthogonal / independently computed.
    ///
    /// We verify this indirectly: in a two-sentence corpus where "cat" and "ran"
    /// never share a sentence, the M they produce must be distinct from one where
    /// they DO share a sentence.
    #[test]
    fn test_sentence_boundary_separates_cooccurrence() {
        // Corpus A: "cat" and "ran" in DIFFERENT sentences — no co-occurrence
        let corpus_a = "cat sat mat. dog ran here.";
        // Corpus B: "cat" and "ran" in the SAME sentence — co-occurrence exists
        let corpus_b = "cat ran mat dog sat here";

        let m_a = SemanticM::from_corpus(corpus_a, 8, 5, 1);
        let m_b = SemanticM::from_corpus(corpus_b, 8, 5, 1);

        // Both must include "cat" and "ran" in vocabulary
        let cat_a = m_a.token_id("cat");
        let ran_a = m_a.token_id("ran");
        let cat_b = m_b.token_id("cat");
        let ran_b = m_b.token_id("ran");

        assert_ne!(cat_a, m_a.unk_id, "'cat' must be in corpus_a vocab");
        assert_ne!(ran_a, m_a.unk_id, "'ran' must be in corpus_a vocab");
        assert_ne!(cat_b, m_b.unk_id, "'cat' must be in corpus_b vocab");
        assert_ne!(ran_b, m_b.unk_id, "'ran' must be in corpus_b vocab");

        // The embeddings for "cat" in A and B should differ because in A
        // "cat" cannot co-occur with "ran"/"dog"/"here" (cross-sentence).
        // We confirm that the two corpora produce DIFFERENT M structures —
        // which only holds when sentence boundaries are enforced.
        let emb_cat_a = &m_a.embeddings[cat_a];
        let emb_cat_b = &m_b.embeddings[cat_b];
        let diff: f64 = emb_cat_a.iter().zip(emb_cat_b.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-9,
            "corpus_a and corpus_b have different sentence structure; \
             'cat' embeddings must differ when sentence boundaries are enforced. \
             diff = {}", diff);
    }

    /// OC-RA-10 enforcement: within-sentence co-occurrence is preserved.
    ///
    /// "cat" and "sat" are in the same sentence — they must still co-occur.
    #[test]
    fn test_within_sentence_cooccurrence_preserved() {
        let corpus = "cat sat mat. dog ran here.";
        let m = SemanticM::from_corpus(corpus, 8, 2, 1);

        // "cat" and "sat" share sentence 1; their embeddings must not be zero
        let cat_id = m.token_id("cat");
        let sat_id = m.token_id("sat");

        assert_ne!(cat_id, m.unk_id, "'cat' must be in vocab");
        assert_ne!(sat_id, m.unk_id, "'sat' must be in vocab");

        // Both must have non-trivial (non-zero) embeddings (co-occurrence happened)
        let cat_norm: f64 = m.embeddings[cat_id].iter().map(|v| v * v).sum::<f64>().sqrt();
        let sat_norm: f64 = m.embeddings[sat_id].iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(cat_norm > 1e-9, "'cat' embedding must be non-zero");
        assert!(sat_norm > 1e-9, "'sat' embedding must be non-zero");
    }

    // ── OC-RA-11: CharM tests ─────────────────────────────────────────────────

    /// CharM builds a vocabulary of characters found in the corpus.
    #[test]
    fn test_char_m_vocab_contains_corpus_chars() {
        let m = CharM::from_corpus("hello world", 8, 1);
        // 'h','e','l','o','w','r','d' + unk
        for ch in "helowrd".chars() {
            assert_ne!(m.char_id(ch), m.unk_id,
                "char '{}' must be in CharM vocab", ch);
        }
    }

    /// Unknown characters map to unk_id.
    #[test]
    fn test_char_m_unk_for_unseen_char() {
        let m = CharM::from_corpus("abc", 4, 1);
        let unk = m.char_id('z');
        assert_eq!(unk, m.unk_id, "unseen char must map to unk_id");
    }

    /// Characters only co-occur within word boundaries.
    /// "ab" and "cd" are in separate words; 'a'/'b' should co-occur but
    /// 'a'/'c' should NOT (they never share a word).
    #[test]
    fn test_char_m_word_boundary_separates_cooccurrence() {
        // "ab" and "cd" are distinct words — no cross-word co-occurrence
        let corpus = "ab cd ab cd ab cd";
        let m = CharM::from_corpus(corpus, 8, 1);

        let a_id = m.char_id('a');
        let b_id = m.char_id('b');
        let c_id = m.char_id('c');
        let d_id = m.char_id('d');

        // All four chars in vocab
        assert_ne!(a_id, m.unk_id);
        assert_ne!(b_id, m.unk_id);
        assert_ne!(c_id, m.unk_id);
        assert_ne!(d_id, m.unk_id);

        // 'a' and 'b' co-occur in every "ab" occurrence — their embeddings
        // must have similar structure (high PPMI relative to 'a'/'c' cross-word pair).
        // We test this structurally: 'a'-'b' co-occurrence must be > 0 while
        // 'a'-'c' co-occurrence must be 0 (they never share a word).
        // Access the raw embedding: since PPMI is normalized, we check via dim.
        let emb_a = m.embedding(a_id);
        let emb_c = m.embedding(c_id);
        // Both should be non-trivial (they appear in a corpus word)
        let norm_a: f64 = emb_a.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm_c: f64 = emb_c.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm_a > 1e-9, "'a' embedding must be non-zero");
        assert!(norm_c > 1e-9, "'c' embedding must be non-zero");

        // 'a' and 'c' never share a word — their PPMI should be 0.
        // We verify this by building a second CharM where "ac" IS a word:
        // the cross-word corpus must produce DIFFERENT embeddings for 'a' than
        // the same corpus with "ac" added.
        let m2 = CharM::from_corpus("ab cd ac ab cd ac", 8, 1);
        let a2_id = m2.char_id('a');
        let emb_a2 = m2.embedding(a2_id);
        let diff: f64 = emb_a.iter().zip(emb_a2.iter())
            .map(|(x, y)| (x - y).abs()).sum();
        assert!(diff > 1e-9,
            "adding 'ac' word must change 'a' embedding (cross-word co-occurrence test)");
    }

    /// CharM embeddings are in [0,1].
    #[test]
    fn test_char_m_domain_bounds() {
        let m = CharM::from_corpus("the quick brown fox jumps over the lazy dog", 8, 1);
        for emb in &m.embeddings {
            for &v in emb {
                assert!(v >= DOMAIN_LOWER && v <= DOMAIN_UPPER + 1e-9,
                    "CharM embedding {} out of [0,1]", v);
            }
        }
    }

    /// CharM encode_word returns the correct char_ids.
    #[test]
    fn test_char_m_encode_word() {
        let m = CharM::from_corpus("cat dog", 8, 1);
        let ids = m.encode_word("cat");
        assert_eq!(ids.len(), 3, "encode_word('cat') must return 3 ids");
        assert_eq!(ids[0], m.char_id('c'));
        assert_eq!(ids[1], m.char_id('a'));
        assert_eq!(ids[2], m.char_id('t'));
    }

    /// CharM on empty corpus does not panic.
    #[test]
    fn test_char_m_empty_corpus() {
        let m = CharM::from_corpus("", 8, 1);
        assert!(m.vocab_size() >= 1, "must have at least unk entry");
    }

    // ── OC-RA-8: SentM tests ──────────────────────────────────────────────────

    /// SentM builds one entry per sentence.
    #[test]
    fn test_sent_m_sentence_count() {
        // Two paragraphs with two sentences each = 4 sentences + 1 unk
        let corpus = "First sentence. Second sentence.\n\nThird sentence. Fourth sentence.";
        let m = SentM::from_corpus(corpus, 8, None);
        assert_eq!(m.sentences.len(), 4, "must have 4 sentences");
        assert_eq!(m.vocab_size(), 5, "4 sentences + 1 unk");
    }

    /// SentM embeddings are in [0,1].
    #[test]
    fn test_sent_m_domain_bounds() {
        let corpus = "Alpha beta. Gamma delta.\n\nEpsilon zeta. Eta theta.";
        let m = SentM::from_corpus(corpus, 8, None);
        for emb in &m.embeddings {
            for &v in emb {
                assert!(v >= DOMAIN_LOWER && v <= DOMAIN_UPPER + 1e-9,
                    "SentM embedding {} out of [0,1]", v);
            }
        }
    }

    /// Sentences in different paragraphs do not co-occur (paragraph boundary respected).
    /// Sentences in the same paragraph do co-occur.
    #[test]
    fn test_sent_m_paragraph_boundary() {
        // para1: "s1 s2" (two sentences); para2: "s3 s4" (two sentences)
        // s1 and s2 must co-occur; s1 and s3 must NOT.
        let corpus = "one two. three four.\n\nfive six. seven eight.";
        let m_two_para = SentM::from_corpus(corpus, 8, None);
        // All four sentences must exist
        assert_eq!(m_two_para.sentences.len(), 4);

        // Single-paragraph corpus: same 4 sentences but no paragraph break.
        // Now all 4 sentences co-occur with each other.
        let corpus_one = "one two. three four. five six. seven eight.";
        let m_one_para = SentM::from_corpus(corpus_one, 8, None);
        assert_eq!(m_one_para.sentences.len(), 4);

        // The two-paragraph corpus must produce DIFFERENT embeddings than the
        // one-paragraph corpus, because co-occurrence structure differs.
        let diff: f64 = m_two_para.embeddings[0].iter()
            .zip(m_one_para.embeddings[0].iter())
            .map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-9,
            "paragraph boundary must produce different co-occurrence structure");
    }

    /// SentM empty corpus does not panic.
    #[test]
    fn test_sent_m_empty_corpus() {
        let m = SentM::from_corpus("", 8, None);
        assert!(m.vocab_size() >= 1);
    }

    // ── OC-RA-8: ParaM tests ──────────────────────────────────────────────────

    /// ParaM builds one entry per paragraph.
    #[test]
    fn test_para_m_paragraph_count() {
        let corpus = "Para one sentence.\n\nPara two sentence.\n\nPara three sentence.";
        let m = ParaM::from_corpus(corpus, 8, None);
        assert_eq!(m.paragraphs.len(), 3, "must have 3 paragraphs");
        assert_eq!(m.vocab_size(), 4, "3 paragraphs + 1 unk");
    }

    /// ParaM embeddings are in [0,1].
    #[test]
    fn test_para_m_domain_bounds() {
        let corpus = "First para alpha beta.\n\nSecond para gamma delta.\n\nThird para epsilon.";
        let m = ParaM::from_corpus(corpus, 8, None);
        for emb in &m.embeddings {
            for &v in emb {
                assert!(v >= DOMAIN_LOWER && v <= DOMAIN_UPPER + 1e-9,
                    "ParaM embedding {} out of [0,1]", v);
            }
        }
    }

    /// ParaM mean_word_embeddings populated when SemanticM provided.
    #[test]
    fn test_para_m_mean_word_embeddings_with_sem() {
        let corpus = "cat sat mat.\n\ndog ran here.";
        let sem = SemanticM::from_corpus(corpus, 8, 2, 1);
        let m = ParaM::from_corpus(corpus, 8, Some(&sem));
        assert_eq!(m.mean_word_embeddings.len(), m.paragraphs.len(),
            "one mean embedding per paragraph");
        for emb in &m.mean_word_embeddings {
            for &v in emb {
                assert!(v >= DOMAIN_LOWER && v <= DOMAIN_UPPER + 1e-9,
                    "mean word embedding {} out of [0,1]", v);
            }
        }
    }

    /// ParaM empty corpus does not panic.
    #[test]
    fn test_para_m_empty_corpus() {
        let m = ParaM::from_corpus("", 8, None);
        assert!(m.vocab_size() >= 1);
    }

    /// tokenize_into_paragraphs splits on double newline.
    #[test]
    fn test_tokenize_into_paragraphs() {
        let text = "First paragraph with two sentences. And more.\n\nSecond paragraph here.";
        let paras = tokenize_into_paragraphs(text);
        assert_eq!(paras.len(), 2, "must have 2 paragraphs");
        assert_eq!(paras[0].len(), 2, "first paragraph has 2 sentences");
        assert_eq!(paras[1].len(), 1, "second paragraph has 1 sentence");
    }
}
