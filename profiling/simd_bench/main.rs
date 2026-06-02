// Profiling-only premise check (profiling/ is excluded from complexity.py). EXPLICIT SIMD this
// time: does wide::f32x8 (real vectorization, not LLVM auto-vec which failed in bench_simd_exp.rs)
// do 8 exps faster than 8x scalar libm f32::exp? If yes, a vectorized forward (8 cards/lane) is
// worth a multi-iteration rewrite; the forward is ~92% transcendentals and ~95% of median time.
// Build/run:  cd profiling/simd_bench && cargo run --release
use std::hint::black_box;
use std::time::Instant;
use wide::{f32x8, i32x8};

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

#[inline(always)]
fn exp8(x: f32x8) -> f32x8 {
    let x = x.fast_max(f32x8::splat(-87.0)).fast_min(f32x8::splat(88.0));
    let n = (x * f32x8::splat(LOG2E)).round();
    let r = x - n * f32x8::splat(LN2);
    let c = |v: f32| f32x8::splat(v);
    let p = c(1.0)
        + r * (c(1.0)
            + r * (c(0.5) + r * (c(1.0 / 6.0) + r * (c(1.0 / 24.0) + r * (c(1.0 / 120.0) + r * c(1.0 / 720.0))))));
    let ni: i32x8 = n.round_int();
    let bits: i32x8 = (ni + i32x8::splat(127)) << 23;
    let two_n: f32x8 = bytemuck::cast(bits);
    p * two_n
}

fn main() {
    let xs: Vec<f32> = (0..1024).map(|i| ((i as f32 / 1024.0) * 2.0 - 1.0) * 6.0).collect();
    let vs: Vec<f32x8> = (0..128)
        .map(|j| {
            let b = j * 8;
            f32x8::from([xs[b], xs[b + 1], xs[b + 2], xs[b + 3], xs[b + 4], xs[b + 5], xs[b + 6], xs[b + 7]])
        })
        .collect();
    let n = 80_000_000usize;
    let m = xs.len();

    // accuracy check
    let mut worst = 0.0f64;
    for &x in &xs {
        let a = exp8(f32x8::splat(x)).to_array()[0] as f64;
        let b = (x as f64).exp();
        worst = worst.max(((a - b) / b).abs());
    }
    println!("exp8 worst rel err vs libm = {worst:e}");

    // (1) scalar libm
    let t = Instant::now();
    let mut a = 0.0f32;
    for i in 0..n {
        a += black_box(xs[i % m]).exp();
    }
    black_box(a);
    let libm = t.elapsed().as_nanos() as f64 / n as f64;

    // (2) scalar poly
    let t = Instant::now();
    let mut b = 0.0f32;
    for i in 0..n {
        b += exp1(black_box(xs[i % m]));
    }
    black_box(b);
    let scalar = t.elapsed().as_nanos() as f64 / n as f64;

    // (3) wide f32x8 poly (8 lanes); per-exp = total / (chunks*8)
    let chunks = n / 8;
    let t = Instant::now();
    let mut acc = f32x8::splat(0.0);
    for k in 0..chunks {
        acc += exp8(black_box(vs[k % 128]));
    }
    black_box(acc.to_array());
    let simd = t.elapsed().as_nanos() as f64 / (chunks * 8) as f64;

    println!("per-exp ns:  libm={libm:.2}   scalar_poly={scalar:.2}   wide_f32x8={simd:.2}");
    println!("wide_f32x8 vs libm = {:.2}x   (>1 means SIMD wins -> vectorized forward is worth it)", libm / simd);
}
