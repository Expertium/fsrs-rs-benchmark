// Profiling-only cost probe (NOT compiled into the crate; lives in profiling/, excluded from
// complexity.py). Question: can a PORTABLE, dependency-free batched exp (a [f32;8] poly that
// LLVM auto-vectorizes to SSE2/NEON — no AVX-512, constraint 7) beat scalar libm f32::exp?
// The iter11 lesson was that SCALAR fast-exp is ~2x SLOWER than libm (latency on a serial dep
// chain); the SIMD premise is that doing 8 at once amortizes that latency. If exp8 >> scalar
// here, the forward SIMD rewrite is worth it; if not, abort it.
// Build:  rustc -O profiling/bench_simd_exp.rs -o profiling/bench_simd_exp.exe
//   (also try: rustc -O -C target-cpu=x86-64-v2 ... to see the AVX/SSE4 ceiling)
// Run:    profiling/bench_simd_exp.exe
use std::hint::black_box;
use std::time::Instant;

const LOG2E: f32 = std::f32::consts::LOG2_E;
const LN2: f32 = std::f32::consts::LN_2;

#[inline(always)]
fn exp1(x: f32) -> f32 {
    let x = if x > 88.0 { 88.0 } else if x < -87.0 { -87.0 } else { x };
    let n = (x * LOG2E).round();
    let r = x - n * LN2;
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0))))));
    let two_n = f32::from_bits((((n as i32) + 127) as u32) << 23);
    p * two_n
}

// Batched over 8 lanes; the fixed-size loop is what LLVM should auto-vectorize.
#[inline(always)]
fn exp8(x: &[f32; 8]) -> [f32; 8] {
    let mut out = [0.0f32; 8];
    for i in 0..8 {
        out[i] = exp1(x[i]);
    }
    out
}

#[inline(never)]
fn run(n: usize, xs: &[f32]) {
    let m = xs.len();

    // (1) scalar libm exp
    let t = Instant::now();
    let mut a = 0.0f32;
    for i in 0..n {
        a += black_box(xs[i % m]).exp();
    }
    black_box(a);
    let libm = t.elapsed().as_nanos() as f64 / n as f64;

    // (2) scalar poly exp (the iter11 fast_exp — expected ~2x slower than libm)
    let t = Instant::now();
    let mut b = 0.0f32;
    for i in 0..n {
        b += exp1(black_box(xs[i % m]));
    }
    black_box(b);
    let scalar_poly = t.elapsed().as_nanos() as f64 / n as f64;

    // (3) batched poly exp8 (8 lanes); per-element time = total / 8
    let t = Instant::now();
    let mut c = 0.0f32;
    let chunks = n / 8;
    for k in 0..chunks {
        let base = (k * 8) % m;
        let mut buf = [0.0f32; 8];
        for j in 0..8 {
            buf[j] = black_box(xs[(base + j) % m]);
        }
        let e = exp8(&buf);
        for j in 0..8 {
            c += e[j];
        }
    }
    black_box(c);
    let simd_poly = t.elapsed().as_nanos() as f64 / (chunks * 8) as f64;

    println!("  per-exp ns:  libm={libm:.2}   scalar_poly={scalar_poly:.2}   batched8_poly={simd_poly:.2}");
    println!("  speedup batched8 vs libm = {:.2}x", libm / simd_poly);
}

fn main() {
    // arguments spanning the forgetting-curve/stability exp ranges actually seen
    let xs: Vec<f32> = (0..1024)
        .map(|i| {
            let u = (i as f32 / 1024.0) * 2.0 - 1.0; // [-1,1]
            u * 6.0 // [-6, 6] typical exponent magnitudes
        })
        .collect();
    let n = 80_000_000usize;
    println!("warmup...");
    run(2_000_000, &xs);
    println!("measured (n={n}):");
    run(n, &xs);
}
