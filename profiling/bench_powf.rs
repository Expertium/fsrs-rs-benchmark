// Profiling-only cost probe (NOT compiled into the crate; lives in profiling/, excluded from
// complexity.py). Question: is `(e * x.ln()).exp()` ~ as cheap as `x.powf(e)` for scalar f32?
// The ln-cache idea rewrites each forward powf as (e*ln x).exp(), caches ln x, and reuses it
// in the backward (killing a recomputed ln). Net per powf-site:
//   current   (fwd+bwd) = powf + ln
//   ln-cache  (fwd+bwd) = (ln+exp) + 0   [the backward ln is now the cached forward ln]
// => WIN iff  lnexp < powf + ln   (forward penalty < backward saving).
// Build:  rustc -O profiling/bench_powf.rs -o profiling/bench_powf.exe
// Run:    profiling/bench_powf.exe
use std::hint::black_box;
use std::time::Instant;

#[inline(never)]
fn run(n: usize, xs: &[f32], es: &[f32]) {
    // powf
    let t = Instant::now();
    let mut a = 0.0f32;
    for i in 0..n {
        let x = black_box(xs[i % xs.len()]);
        let e = black_box(es[i % es.len()]);
        a += x.powf(e);
    }
    black_box(a);
    let powf = t.elapsed().as_nanos() as f64 / n as f64;

    // ln + exp (the ln-cache forward form)
    let t = Instant::now();
    let mut b = 0.0f32;
    for i in 0..n {
        let x = black_box(xs[i % xs.len()]);
        let e = black_box(es[i % es.len()]);
        b += (e * x.ln()).exp();
    }
    black_box(b);
    let lnexp = t.elapsed().as_nanos() as f64 / n as f64;

    // ln alone (the backward term we'd save by caching)
    let t = Instant::now();
    let mut c = 0.0f32;
    for i in 0..n {
        let x = black_box(xs[i % xs.len()]);
        c += x.ln();
    }
    black_box(c);
    let ln = t.elapsed().as_nanos() as f64 / n as f64;

    // exp alone (reference)
    let t = Instant::now();
    let mut d = 0.0f32;
    for i in 0..n {
        let e = black_box(es[i % es.len()]);
        d += e.exp();
    }
    black_box(d);
    let exp = t.elapsed().as_nanos() as f64 / n as f64;

    println!("  powf   = {powf:.3} ns/op");
    println!("  ln+exp = {lnexp:.3} ns/op");
    println!("  ln     = {ln:.3} ns/op");
    println!("  exp    = {exp:.3} ns/op");
    let fwd_penalty = lnexp - powf;
    let bwd_saving = ln;
    let net = bwd_saving - fwd_penalty; // >0 => ln-cache is a net win per powf-site
    println!("  fwd penalty (lnexp-powf) = {fwd_penalty:+.3} ns;  bwd saving (ln) = {bwd_saving:.3} ns");
    println!("  NET per powf-site (saving-penalty) = {net:+.3} ns  => {}",
        if net > 0.0 { "ln-cache WINS (proceed)" } else { "ln-cache LOSES (skip)" });
}

fn main() {
    let n = 80_000_000usize;
    // Representative FSRS magnitudes: bases are stabilities/curve bases (>=1, up to ~1e4);
    // exponents are decays/weights in ~[-1, 1].
    let xs: Vec<f32> = (0..997)
        .map(|i| {
            let r = i as f32 / 997.0;
            (0.5 + r * 4000.0).max(1.0001) // 1.0001 .. ~4000
        })
        .collect();
    let es: Vec<f32> = (0..991)
        .map(|i| -0.95 + (i as f32 / 991.0) * 1.9) // -0.95 .. 0.95
        .collect();
    println!("scalar f32, n={n}:");
    run(n, &xs, &es);
}
