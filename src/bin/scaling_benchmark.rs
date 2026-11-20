// scaling_benchmark.rs — CPU scaling baseline for ABR relational attention.
//
// Measures, on CPU, how cost and memory grow, as a baseline for later GPU work.
//
// Table 1 (sequence length n, fresh field): time per token for AttentionLayer::forward
//   (accumulation + inference), stored pairs, admitted pairs, and the mean number of
//   admitted edges per position (k). Alongside: plain causal softmax attention over the
//   same embeddings (Q = K = V = M(token), scores q·k/√d), O(n²·d).
// Table 2 (long stream, field grows): tokens streamed in chunks of CHUNK tokens with
//   reset_stream() between chunks (MAX_SEQUENCE_LENGTH = 4096), as held_out_benchmark does.
//   Reports time per token for each chunk, split into accumulation and inference, and the
//   field size at the end of the chunk. Flat per-token time = cost independent of history.
//
// Tokens: deterministic Zipf(s = 1.0) over VOCAB ids (an LCG, seeded), so frequent
// tokens repeat as in text. Embeddings: build_vocab_table_seeded(VOCAB, DIM, SEED).
// Timings are wall-clock on whatever CPU runs this; compare rows, not machines.
//
// RUN: cargo run --release --bin scaling_benchmark [-- --max-stream N]

use abr_relational_attention::attention_layer::AttentionLayer;
use abr_relational_attention::declaration::{build_vocab_table_seeded, Locus, VocabTable, THETA_MIN};
use std::time::Instant;

const VOCAB: usize = 2000;
const DIM: usize = 50;
const SEED: u64 = 20261120;
const CHUNK: usize = 4000;

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn zipf_tokens(n: usize, seed: u64) -> Vec<usize> {
    let cdf: Vec<f64> = {
        let w: Vec<f64> = (1..=VOCAB).map(|r| 1.0 / r as f64).collect();
        let total: f64 = w.iter().sum();
        let mut acc = 0.0;
        w.iter().map(|x| { acc += x / total; acc }).collect()
    };
    let mut rng = Lcg(seed);
    (0..n).map(|_| {
        let u = rng.next_f64();
        cdf.partition_point(|&c| c < u).min(VOCAB - 1)
    }).collect()
}

/// Causal softmax attention, Q = K = V = embedding. Returns a checksum so the work is not optimised away.
fn softmax_attention(ids: &[usize], vocab: &VocabTable) -> f64 {
    let n = ids.len();
    let scale = 1.0 / (DIM as f64).sqrt();
    let mut check = 0.0;
    let mut scores = vec![0.0f64; n];
    for j in 0..n {
        let q = &vocab[ids[j]];
        let mut mx = f64::NEG_INFINITY;
        for i in 0..=j {
            let k = &vocab[ids[i]];
            let s: f64 = q.iter().zip(k).map(|(a, b)| a * b).sum::<f64>() * scale;
            scores[i] = s;
            if s > mx { mx = s; }
        }
        let mut z = 0.0;
        for s in scores.iter_mut().take(j + 1) { *s = (*s - mx).exp(); z += *s; }
        let mut out = vec![0.0f64; DIM];
        for i in 0..=j {
            let w = scores[i] / z;
            for (o, v) in out.iter_mut().zip(&vocab[ids[i]]) { *o += w * v; }
        }
        check += out[0];
    }
    check
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let max_stream: usize = args.iter().position(|a| a == "--max-stream")
        .and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(32_000);
    let vocab = build_vocab_table_seeded(VOCAB, DIM, SEED);
    println!("ABR relational attention: CPU scaling baseline");
    println!("vocab {VOCAB}, dim {DIM}, Zipf(1.0) tokens, theta {THETA_MIN}, window 64\n");

    println!("Table 1. Sequence length n, fresh field");
    println!("{:>6} {:>14} {:>14} {:>10} {:>10} {:>12}", "n", "ABR us/token", "softmax us/tok", "stored", "admitted", "mean k");
    for &n in &[256usize, 512, 1024, 2048, 4000] {
        let ids = zipf_tokens(n, SEED + n as u64);
        let mut layer = AttentionLayer::new();
        let t = Instant::now();
        let out = layer.forward(&ids, &vocab);
        let abr = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        let mean_k = out.iter().map(|o| o.n_admitted as f64).sum::<f64>() / n as f64;
        let t = Instant::now();
        let c = softmax_attention(&ids, &vocab);
        let sm = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        std::hint::black_box(c);
        println!("{:>6} {:>14.2} {:>14.2} {:>10} {:>10} {:>12.1}", n, abr, sm, layer.field_size(), layer.admitted_pairs(), mean_k);
    }

    println!("\nTable 2. Long stream, field persists across chunks of {CHUNK} tokens");
    println!("{:>9} {:>16} {:>16} {:>10} {:>10}", "streamed", "accum us/token", "infer us/token", "stored", "admitted");
    let ids = zipf_tokens(max_stream, SEED);
    let mut layer = AttentionLayer::new();
    let mut streamed = 0;
    for chunk in ids.chunks(CHUNK) {
        let t = Instant::now();
        for &tok in chunk {
            let pos = layer.field.frontier();
            layer.field.arrive(Locus::new(pos, tok), &vocab);
        }
        let acc = t.elapsed().as_secs_f64() * 1e6 / chunk.len() as f64;
        let t = Instant::now();
        let mut check = 0.0;
        for pos in 0..chunk.len() {
            let (r, _) = layer.field.inference_at(pos, &vocab, layer.theta);
            check += r.first().copied().unwrap_or(0.0);
        }
        std::hint::black_box(check);
        let inf = t.elapsed().as_secs_f64() * 1e6 / chunk.len() as f64;
        streamed += chunk.len();
        println!("{:>9} {:>16.2} {:>16.2} {:>10} {:>10}", streamed, acc, inf, layer.field_size(), layer.admitted_pairs());
        layer.reset_stream();
    }
}
