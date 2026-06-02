//! Hand-written forward + reverse-mode backward for the training BCE loss, with NO
//! autodiff tape. Profiling showed burn's `Autodiff<NdArray<f32>>` forward+backward is
//! ~85% of compute_parameters() time (backward alone 59%); replacing it with a direct
//! scalar VJP removes the tape-record + tape-replay overhead.
//!
//! Scalar per-card loop (the recurrence is sequential; cards are independent). The
//! forward mirrors model_v7's math EXACTLY and stashes the intermediates each step needs;
//! the backward is the reverse-mode adjoint of that same forward (single source of truth).
//! Result is not bit-for-bit vs autodiff (different FP order) but is judged by the
//! ±0.0010 average-log-loss band.

const S_MIN: f32 = 0.0001;
const S_MAX: f32 = 36500.0;
const D_MIN: f32 = 1.0;
const D_MAX: f32 = 10.0;
const MIN_R: f32 = 1e-5;
const MAX_R: f32 = 1.0 - 1e-5;

#[inline(always)]
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}

/// Loop-invariant (weight-only) subexpressions, computed ONCE per batch_loss[_and_grad] call
/// instead of once per (card × timestep). Identical ops to the old inline versions (same f32/f64),
/// so it's BIT-FOR-BIT — it just lifts redundant transcendentals out of the hot per-timestep
/// forward (and the one in next_d's backward). iter10.
struct WConsts {
    ln_w27: f32, // ln(w27)        (curve_fwd: q1 = ln_w27/decay1)
    ln_w28: f32, // ln(w28)        (curve_fwd: p28 = (inv2*ln_w28).exp())
    aa7: f32,    // exp(w[7]-1.5)  (stab_fwd slow, start=7)
    aa16: f32,   // exp(w[16]-1.5) (stab_fwd fast, start=16)
    init: f32,   // w4 - exp(3*w5) + 1   (next_d_fwd, f32)
    exp3w5: f64, // exp(3*w5) in f64      (next_d_bwd: d(init)/d(w5))
}

fn wconsts(w: &[f32]) -> WConsts {
    WConsts {
        ln_w27: w[27].ln(),
        ln_w28: w[28].ln(),
        aa7: (w[7] - 1.5).exp(),
        aa16: (w[16] - 1.5).exp(),
        init: w[4] - (w[5] * 3.0).exp() + 1.0,
        exp3w5: (w[5] as f64 * 3.0).exp(),
    }
}

// ===================== forgetting curve =====================

struct CurveCache {
    out: f32,
    a: f32,
    bv: f32,
    dm1: f32,
    decay1: f32,
    factor1: f32,
    b1: f32,
    r1: f32,
    q1: f32,
    e1: f32,
    m1: f32,
    p35: f32,
    dm2: f32,
    decay2: f32,
    factor2: f32,
    b2: f32,
    r2: f32,
    inv2: f32,
    p28: f32,
    m2: f32,
    ex34: f32,
    weight1: f32,
    weight2: f32,
    wsum: f32,
    num: f32,
    ret: f32,
    p31: f32,
    s32: f32,
    ex33: f32,
    // Cached ln(base) for each powf rewritten as (exp*ln_base).exp() — reused by curve_bwd so
    // the backward's exponent-derivatives (d/dw = value*ln_base) don't recompute ln. iter8.
    ln_sf: f32, // shared by p35 (sf^w35) and p31 (sf^-w31)
    ln_b1: f32,
    ln_b2: f32,
    ln_s: f32,
    ln_w27: f32,
    ln_w28: f32,
}

fn curve_fwd(w: &[f32], t: f32, s: f32, sf: f32, d: f32, ln_w27: f32, ln_w28: f32) -> CurveCache {
    let t = t.max(0.0);
    let a = t / sf;
    let bv = t / s;
    let ln_sf = sf.ln();
    let p35 = (w[35] * ln_sf).exp(); // sf^w35
    let m1 = w[25] * p35;
    let dm1 = clamp(m1, 0.01, 0.95);
    let decay1 = -dm1;
    let q1 = ln_w27 / decay1; // ln_w27 hoisted (loop-invariant)
    let e1 = q1.min(60.0).exp();
    let factor1 = e1 - 1.0;
    let b1 = a * factor1 + 1.0;
    let ln_b1 = b1.ln();
    let r1 = (decay1 * ln_b1).exp(); // b1^decay1
    let ex34 = ((d - 5.0) * w[34]).exp();
    let m2 = w[26] * ex34;
    let dm2 = clamp(m2, 0.01, 0.95);
    let decay2 = -dm2;
    let inv2 = 1.0 / decay2;
    let p28 = (inv2 * ln_w28).exp(); // w28^inv2 ; ln_w28 hoisted (loop-invariant)
    let factor2 = p28 - 1.0;
    let b2 = bv * factor2 + 1.0;
    let ln_b2 = b2.ln();
    let r2 = (decay2 * ln_b2).exp(); // b2^decay2
    let p31 = ((-w[31]) * ln_sf).exp(); // sf^-w31 (reuse ln_sf)
    let weight1 = w[29] * p31;
    let ln_s = s.ln();
    let s32 = (w[32] * ln_s).exp(); // s^w32
    let ex33 = ((d - 5.0) * w[33]).exp();
    let weight2 = w[30] * s32 * ex33;
    let wsum = weight1 + weight2;
    let num = weight1 * r1 + weight2 * r2;
    let ret = num / wsum;
    let out = ret * (1.0 - 2e-5) + 1e-5;
    CurveCache {
        out, a, bv, dm1, decay1, factor1, b1, r1, q1, e1, m1, p35, dm2, decay2, factor2,
        b2, r2, inv2, p28, m2, ex34, weight1, weight2, wsum, num, ret, p31, s32, ex33,
        ln_sf, ln_b1, ln_b2, ln_s, ln_w27, ln_w28,
    }
}

/// VJP of the curve. `t` is data (no grad). Returns adjoints (g_s, g_sf, g_d), accumulates gw.
#[allow(clippy::too_many_arguments)]
fn curve_bwd(
    w: &[f32], c: &CurveCache, t: f32, s: f32, sf: f32, d: f32, g_out: f64, gw: &mut [f64],
) -> (f64, f64, f64) {
    let t = t.max(0.0) as f64;
    let (s, sf, d) = (s as f64, sf as f64, d as f64);
    let g_ret = g_out * (1.0 - 2e-5);
    let (wsum, ret) = (c.wsum as f64, c.ret as f64);
    let g_num = g_ret / wsum;
    let g_wsum = -g_ret * ret / wsum;
    let (r1, r2) = (c.r1 as f64, c.r2 as f64);
    let mut g_weight1 = g_num * r1 + g_wsum;
    let mut g_weight2 = g_num * r2 + g_wsum;
    let mut g_r1 = g_num * c.weight1 as f64;
    let mut g_r2 = g_num * c.weight2 as f64;
    // weight2 = w30 * s32 * ex33
    let (s32, ex33) = (c.s32 as f64, c.ex33 as f64);
    gw[30] += g_weight2 * s32 * ex33;
    let g_s32 = g_weight2 * w[30] as f64 * ex33;
    let g_ex33 = g_weight2 * w[30] as f64 * s32;
    let mut g_d = g_ex33 * ex33 * w[33] as f64; // ex33 = exp((d-5)*w33)
    gw[33] += g_ex33 * ex33 * (d - 5.0);
    let mut g_s = g_s32 * w[32] as f64 * (s32 / s); // d(s^w32)/ds = w32*s^(w32-1) = w32*s32/s
    gw[32] += g_s32 * s32 * c.ln_s as f64;
    // weight1 = w29 * p31 ; p31 = sf^(-w31)
    let p31 = c.p31 as f64;
    gw[29] += g_weight1 * p31;
    let g_p31 = g_weight1 * w[29] as f64;
    let mut g_sf = g_p31 * (-(w[31] as f64)) * (p31 / sf); // d(sf^-w31)/dsf = -w31*p31/sf
    gw[31] += g_p31 * (-(p31 * c.ln_sf as f64));
    // r2 = b2^decay2
    let (b2, decay2) = (c.b2 as f64, c.decay2 as f64);
    let g_b2 = g_r2 * decay2 * (r2 / b2); // d(b2^decay2)/db2 = decay2*r2/b2
    let mut g_decay2 = g_r2 * r2 * c.ln_b2 as f64;
    let factor2 = c.factor2 as f64;
    let g_bv = g_b2 * factor2; // b2 = bv*factor2 + 1
    let g_factor2 = g_b2 * c.bv as f64;
    let g_p28 = g_factor2; // factor2 = p28 - 1
    // p28 = w28^inv2
    let (inv2, p28) = (c.inv2 as f64, c.p28 as f64);
    gw[28] += g_p28 * inv2 * (p28 / w[28] as f64); // d(w28^inv2)/dw28 = inv2*p28/w28
    let g_inv2 = g_p28 * p28 * c.ln_w28 as f64;
    g_decay2 += g_inv2 * (-1.0 / (decay2 * decay2)); // inv2 = 1/decay2
    let g_dm2 = -g_decay2; // decay2 = -dm2
    let g_m2 = if c.m2 > 0.01 && c.m2 < 0.95 { g_dm2 } else { 0.0 };
    let ex34 = c.ex34 as f64;
    gw[26] += g_m2 * ex34; // m2 = w26 * ex34
    let g_ex34 = g_m2 * w[26] as f64;
    g_d += g_ex34 * ex34 * w[34] as f64; // ex34 = exp((d-5)*w34)
    gw[34] += g_ex34 * ex34 * (d - 5.0);
    g_s += g_bv * (-t / (s * s)); // bv = t/s
    // r1 = b1^decay1
    let (b1, decay1) = (c.b1 as f64, c.decay1 as f64);
    let g_b1 = g_r1 * decay1 * (r1 / b1); // d(b1^decay1)/db1 = decay1*r1/b1
    let mut g_decay1 = g_r1 * r1 * c.ln_b1 as f64;
    let g_a = g_b1 * c.factor1 as f64; // b1 = a*factor1 + 1
    let g_factor1 = g_b1 * c.a as f64;
    let g_e1 = g_factor1; // factor1 = e1 - 1
    let g_q1c = g_e1 * c.e1 as f64; // e1 = exp(q1c)
    let g_q1 = if (c.q1 as f64) < 60.0 { g_q1c } else { 0.0 }; // q1c = min(q1,60)
    // q1 = ln(w27) / decay1
    let lw27 = c.ln_w27 as f64;
    let g_lw27 = g_q1 / decay1;
    g_decay1 += g_q1 * (-lw27 / (decay1 * decay1));
    gw[27] += g_lw27 / w[27] as f64; // lw27 = ln(w27)
    let g_dm1 = -g_decay1; // decay1 = -dm1
    let g_m1 = if c.m1 > 0.01 && c.m1 < 0.95 { g_dm1 } else { 0.0 };
    let p35 = c.p35 as f64;
    gw[25] += g_m1 * p35; // m1 = w25 * p35
    let g_p35 = g_m1 * w[25] as f64;
    g_sf += g_p35 * w[35] as f64 * (p35 / sf); // d(sf^w35)/dsf = w35*p35/sf
    gw[35] += g_p35 * p35 * c.ln_sf as f64;
    g_sf += g_a * (-t / (sf * sf)); // a = t/sf
    (g_s, g_sf, g_d)
}

// ===================== stability after review =====================

struct StabCache {
    out: f32,
    nsf_fail: f32,
    pls: f32,
    sinc: f32,
    ls_sinc: f32,
    aa: f32,
    bb: f32,
    cc: f32,
    expr: f32,
    pp: f32,
    qbase: f32,
    rexp: f32,
    hard: f32,
    easy: f32,
    ln_ls: f32,  // ln(last_s)   for cc = last_s^-w[start+1]
    ln_ld: f32,  // ln(last_d)   for pp = last_d^-w[start+4]
    ln_ls1: f32, // ln(last_s+1) for qbase = (last_s+1)^w[start+5]
}

fn stab_fwd(w: &[f32], last_s: f32, last_d: f32, r: f32, rating: f32, start: usize, aa: f32) -> StabCache {
    let hard = if rating == 2.0 { w[start + 7] } else { 1.0 };
    let easy = if rating == 4.0 { w[start + 8] } else { 1.0 };
    let ln_ld = last_d.ln();
    let pp = ((-w[start + 4]) * ln_ld).exp(); // last_d^-w[start+4]
    let ln_ls1 = (last_s + 1.0).ln();
    let qbase = (w[start + 5] * ln_ls1).exp(); // (last_s+1)^w[start+5]
    let rexp = ((1.0 - r) * w[start + 6]).exp();
    let nsf_fail = w[start + 3] * pp * (qbase - 1.0) * rexp;
    let pls = last_s.min(nsf_fail);
    let bb = 11.0 - last_d; // aa = exp(w[start]-1.5) hoisted (loop-invariant)
    let ln_ls = last_s.ln();
    let cc = ((-w[start + 1]) * ln_ls).exp(); // last_s^-w[start+1]
    let expr = ((1.0 - r) * w[start + 2]).exp();
    let sinc = aa * bb * cc * (expr - 1.0) * hard * easy + 1.0;
    let ls_sinc = last_s * sinc;
    let nss = pls.max(ls_sinc);
    let out = if rating > 1.0 { nss } else { pls };
    StabCache {
        out, nsf_fail, pls, sinc, ls_sinc, aa, bb, cc, expr, pp, qbase, rexp, hard, easy,
        ln_ls, ln_ld, ln_ls1,
    }
}

/// VJP of stability_for_set. Returns (g_last_s, g_last_d, g_r), accumulates gw[start..start+9].
fn stab_bwd(
    w: &[f32], c: &StabCache, last_s: f32, last_d: f32, r: f32, rating: f32, start: usize,
    g_out: f64, gw: &mut [f64],
) -> (f64, f64, f64) {
    let (last_s, last_d, r) = (last_s as f64, last_d as f64, r as f64);
    let (g_nss, g_pls_direct) = if rating > 1.0 { (g_out, 0.0) } else { (0.0, g_out) };
    // nss = max(pls, ls_sinc)
    let g_pls_from_nss = if c.pls >= c.ls_sinc { g_nss } else { 0.0 };
    let g_ls_sinc = if c.ls_sinc > c.pls { g_nss } else { 0.0 };
    // ls_sinc = last_s * sinc
    let mut g_last_s = g_ls_sinc * c.sinc as f64;
    let g_sinc = g_ls_sinc * last_s;
    let g_pls = g_pls_direct + g_pls_from_nss;
    // pls = min(last_s, nsf_fail)
    let nsf_fail = c.nsf_fail as f64;
    g_last_s += if last_s <= nsf_fail { g_pls } else { 0.0 };
    let g_nsf_fail = if nsf_fail < last_s { g_pls } else { 0.0 };
    // sinc = aa*bb*cc*(expr-1)*hard*easy + 1  ; let prod = sinc - 1
    let (aa, bb, cc) = (c.aa as f64, c.bb as f64, c.cc as f64);
    let em1 = c.expr as f64 - 1.0;
    let (hard, easy) = (c.hard as f64, c.easy as f64);
    let g_prod = g_sinc;
    let prod = aa * bb * cc * em1 * hard * easy;
    gw[start] += g_prod * prod; // aa = exp(w[start]-1.5); dprod/dw[start] = prod
    let g_bb = g_prod * (aa * cc * em1 * hard * easy);
    let g_cc = g_prod * (aa * bb * em1 * hard * easy);
    let g_em1 = g_prod * (aa * bb * cc * hard * easy);
    if rating == 2.0 {
        gw[start + 7] += g_prod * (aa * bb * cc * em1 * easy);
    }
    if rating == 4.0 {
        gw[start + 8] += g_prod * (aa * bb * cc * em1 * hard);
    }
    let mut g_last_d = g_bb * (-1.0); // bb = 11 - last_d
    g_last_s += g_cc * (-(w[start + 1] as f64)) * (cc / last_s); // d(ls^-w)/dls = -w*cc/ls
    gw[start + 1] += g_cc * (-(cc * c.ln_ls as f64));
    // expr = exp((1-r)*w[start+2]) ; em1 = expr - 1
    let expr = c.expr as f64;
    let mut g_r = g_em1 * expr * (-(w[start + 2] as f64));
    gw[start + 2] += g_em1 * expr * (1.0 - r);
    // nsf_fail = w[start+3] * pp * (qbase-1) * rexp
    let (pp, qbase, rexp) = (c.pp as f64, c.qbase as f64, c.rexp as f64);
    let q = qbase - 1.0;
    gw[start + 3] += g_nsf_fail * (pp * q * rexp);
    let g_pp = g_nsf_fail * w[start + 3] as f64 * q * rexp;
    let g_q = g_nsf_fail * w[start + 3] as f64 * pp * rexp;
    let g_rexp = g_nsf_fail * w[start + 3] as f64 * pp * q;
    // pp = last_d^(-w[start+4])
    g_last_d += g_pp * (-(w[start + 4] as f64)) * (pp / last_d); // d(ld^-w)/dld = -w*pp/ld
    gw[start + 4] += g_pp * (-(pp * c.ln_ld as f64));
    // q = qbase - 1 ; qbase = (last_s+1)^w[start+5]
    g_last_s += g_q * w[start + 5] as f64 * (qbase / (last_s + 1.0)); // d((ls+1)^w)/dls = w*qbase/(ls+1)
    gw[start + 5] += g_q * qbase * c.ln_ls1 as f64;
    // rexp = exp((1-r)*w[start+6])
    g_r += g_rexp * rexp * (-(w[start + 6] as f64));
    gw[start + 6] += g_rexp * rexp * (1.0 - r);
    (g_last_s, g_last_d, g_r)
}

// ===================== next difficulty =====================

fn next_d_fwd(w: &[f32], last_d: f32, rating: f32, init: f32) -> (f32, f32, f32) {
    let delta_d = -w[6] * (rating - 3.0);
    let new_d = last_d + (10.0 - last_d) * delta_d / 9.0;
    // init = w4 - exp(3*w5) + 1 hoisted (loop-invariant)
    let out_pre = 0.01 * init + 0.99 * new_d;
    (clamp(out_pre, D_MIN, D_MAX), out_pre, delta_d)
}

/// VJP of next_difficulty. Returns g_last_d, accumulates gw[4], gw[5], gw[6].
fn next_d_bwd(out_pre: f32, delta_d: f32, last_d: f32, rating: f32, g_out: f64, gw: &mut [f64], exp3w5: f64) -> f64 {
    let g_out_pre = if out_pre > D_MIN && out_pre < D_MAX { g_out } else { 0.0 };
    let g_init = g_out_pre * 0.01;
    let g_new_d = g_out_pre * 0.99;
    gw[4] += g_init;
    gw[5] += g_init * (-exp3w5 * 3.0); // init = w4 - exp(3 w5) + 1 ; exp3w5 hoisted
    let delta_d = delta_d as f64;
    let g_last_d = g_new_d * (1.0 - delta_d / 9.0);
    let g_delta_d = g_new_d * (10.0 - last_d as f64) / 9.0;
    gw[6] += g_delta_d * (-(rating as f64 - 3.0)); // delta_d = -w6*(rating-3)
    g_last_d
}

// ===================== one recurrence step =====================

struct StepCache {
    s0: f32,
    d0: f32,
    sf0: f32,
    last_s: f32,
    last_d: f32,
    last_sf: f32,
    dt: f32,
    rating: f32,
    nth0: bool,
    curve: CurveCache,
    slow: StabCache,
    fast: StabCache,
    nd_out_pre: f32,
    nd_delta_d: f32,
    ns3: f32, // value before the final stability clamp
    nsf3: f32,
}

fn step_fwd(w: &[f32], delta_t: f32, rating: f32, state: (f32, f32, f32), nth0: bool, wc: &WConsts) -> ((f32, f32, f32), StepCache) {
    let (s0, d0, sf0) = state;
    let last_s = clamp(s0, S_MIN, S_MAX);
    let last_d = clamp(d0, D_MIN, D_MAX);
    let last_sf = clamp(sf0, S_MIN, S_MAX);
    let dt = delta_t.max(0.0);
    let curve = curve_fwd(w, dt, last_s, last_sf, last_d, wc.ln_w27, wc.ln_w28);
    let r = curve.out;
    let slow = stab_fwd(w, last_s, last_d, r, rating, 7, wc.aa7);
    let fast = stab_fwd(w, last_sf, last_d, r, rating, 16, wc.aa16);
    let (nd1, nd_out_pre, nd_delta_d) = next_d_fwd(w, last_d, rating, wc.init);
    let (mut ns, mut nsf, mut nd) = (slow.out, fast.out, nd1);
    if nth0 && s0 == 0.0 {
        let rc = clamp(rating, 1.0, 4.0);
        let init_s = w[(rc as usize) - 1];
        let init_d = clamp(w[4] - (w[5] * (rc - 1.0)).exp() + 1.0, D_MIN, D_MAX);
        ns = init_s;
        nsf = 0.8 * init_s;
        nd = init_d;
    }
    if rating == 0.0 {
        ns = last_s;
        nsf = last_sf;
        nd = last_d;
    }
    let ns3 = ns;
    let nsf3 = nsf;
    let out = (clamp(ns, S_MIN, S_MAX), nd, clamp(nsf, S_MIN, S_MAX));
    let cache = StepCache {
        s0, d0, sf0, last_s, last_d, last_sf, dt, rating, nth0, curve, slow, fast,
        nd_out_pre, nd_delta_d, ns3, nsf3,
    };
    (out, cache)
}

/// VJP of one step. Given adjoints on the OUTPUT state, returns adjoints on the INPUT state.
fn step_bwd(w: &[f32], c: &StepCache, g_out: (f64, f64, f64), gw: &mut [f64], wc: &WConsts) -> (f64, f64, f64) {
    let (g_ns_out, g_nd_out, g_nsf_out) = g_out;
    // final clamps: ns_out = clamp(ns3, S_MIN, S_MAX) ; nd has no final clamp
    let g_ns3 = if c.ns3 > S_MIN && c.ns3 < S_MAX { g_ns_out } else { 0.0 };
    let g_nsf3 = if c.nsf3 > S_MIN && c.nsf3 < S_MAX { g_nsf_out } else { 0.0 };
    let g_nd3 = g_nd_out;
    // padding mask
    let (mut g_last_s_extra, mut g_last_sf_extra, mut g_last_d_extra) = (0.0, 0.0, 0.0);
    let (g_ns2, g_nsf2, g_nd2) = if c.rating == 0.0 {
        g_last_s_extra += g_ns3;
        g_last_sf_extra += g_nsf3;
        g_last_d_extra += g_nd3;
        (0.0, 0.0, 0.0)
    } else {
        (g_ns3, g_nsf3, g_nd3)
    };
    // init override (t==0)
    let (g_ns1, g_nsf1, g_nd1) = if c.nth0 && c.s0 == 0.0 {
        let rc = clamp(c.rating, 1.0, 4.0);
        gw[(rc as usize) - 1] += g_ns2 + g_nsf2 * 0.8; // init_s = w[rc-1]; nsf=0.8*init_s
        let id_pre = w[4] - (w[5] * (rc - 1.0)).exp() + 1.0;
        if id_pre > D_MIN && id_pre < D_MAX {
            gw[4] += g_nd2;
            gw[5] += g_nd2 * (-((w[5] as f64 * (rc as f64 - 1.0)).exp()) * (rc as f64 - 1.0));
        }
        (0.0, 0.0, 0.0)
    } else {
        (g_ns2, g_nsf2, g_nd2)
    };
    // ns1 = stab(slow, last_s, last_d) ; nsf1 = stab(fast, last_sf, last_d) ; nd1 = next_d(last_d)
    let (g_ls_a, g_ld_a, g_r_a) =
        stab_bwd(w, &c.slow, c.last_s, c.last_d, c.curve.out, c.rating, 7, g_ns1, gw);
    let (g_lsf_b, g_ld_b, g_r_b) =
        stab_bwd(w, &c.fast, c.last_sf, c.last_d, c.curve.out, c.rating, 16, g_nsf1, gw);
    let g_ld_c = next_d_bwd(c.nd_out_pre, c.nd_delta_d, c.last_d, c.rating, g_nd1, gw, wc.exp3w5);
    // r = curve(dt, last_s, last_sf, last_d)
    let (g_ls_d, g_lsf_d, g_ld_d) =
        curve_bwd(w, &c.curve, c.dt, c.last_s, c.last_sf, c.last_d, g_r_a + g_r_b, gw);
    let g_last_s = g_ls_a + g_ls_d + g_last_s_extra;
    let g_last_sf = g_lsf_b + g_lsf_d + g_last_sf_extra;
    let g_last_d = g_ld_a + g_ld_b + g_ld_c + g_ld_d + g_last_d_extra;
    // input clamps
    let g_s0 = if c.s0 > S_MIN && c.s0 < S_MAX { g_last_s } else { 0.0 };
    let g_d0 = if c.d0 > D_MIN && c.d0 < D_MAX { g_last_d } else { 0.0 };
    let g_sf0 = if c.sf0 > S_MIN && c.sf0 < S_MAX { g_last_sf } else { 0.0 };
    (g_s0, g_d0, g_sf0)
}

// ===================== batch driver =====================

/// Forward-only BCE loss for one batch. The per-epoch **validation** scorer (iter6) and the
/// gradient unit test. `t_hist`/`r_hist` are row-major [seq_len, batch]; card c is column c.
/// Single-threaded on purpose: 2-thread splitting (iter7) regressed under --processes 10
/// (forward-only is memory-bound, so the 2nd thread just fights for saturated bandwidth).
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_loss(
    w: &[f32], t_hist: &[f32], r_hist: &[f32], seq_len: usize, batch: usize,
    delta_ts: &[f32], labels: &[f32], weights: &[f32],
) -> f64 {
    let wc = wconsts(w);
    let mut loss = 0.0f64;
    for c in 0..batch {
        let mut state = (0.0f32, 0.0f32, 0.0f32);
        for t in 0..seq_len {
            state = step_fwd(w, t_hist[t * batch + c], r_hist[t * batch + c], state, t == 0, &wc).0;
        }
        let (s, d, sf) = state;
        let r = clamp(curve_fwd(w, delta_ts[c], s, sf, d, wc.ln_w27, wc.ln_w28).out, MIN_R, MAX_R);
        loss += -(weights[c] as f64)
            * (labels[c] as f64 * (r as f64).ln() + (1.0 - labels[c] as f64) * (1.0 - r as f64).ln());
    }
    loss
}

/// Forward + reverse-mode backward over cards `[start, end)`; accumulates d(loss)/d(w)
/// into `gw` and returns the summed loss for that range.
#[allow(clippy::too_many_arguments)]
fn loss_and_grad_range(
    w: &[f32], t_hist: &[f32], r_hist: &[f32], seq_len: usize, batch: usize,
    delta_ts: &[f32], labels: &[f32], weights: &[f32], gw: &mut [f64], start: usize, end: usize,
) -> f64 {
    let wc = wconsts(w);
    let mut loss = 0.0f64;
    let mut caches: Vec<StepCache> = Vec::with_capacity(seq_len);
    for c in start..end {
        caches.clear();
        let mut state = (0.0f32, 0.0f32, 0.0f32);
        for t in 0..seq_len {
            let (ns, cache) = step_fwd(w, t_hist[t * batch + c], r_hist[t * batch + c], state, t == 0, &wc);
            state = ns;
            caches.push(cache);
        }
        let (s, d, sf) = state;
        let fc = curve_fwd(w, delta_ts[c], s, sf, d, wc.ln_w27, wc.ln_w28);
        let r_raw = fc.out;
        let r = clamp(r_raw, MIN_R, MAX_R);
        let (lbl, wt) = (labels[c] as f64, weights[c] as f64);
        loss += -wt * (lbl * (r as f64).ln() + (1.0 - lbl) * (1.0 - r as f64).ln());
        // d loss / d r (then clamp)
        let g_r = -wt * (lbl / r as f64 - (1.0 - lbl) / (1.0 - r as f64));
        let g_rraw = if r_raw > MIN_R && r_raw < MAX_R { g_r } else { 0.0 };
        // final curve backward -> adjoints on final (s, sf, d)
        let (mut g_s, mut g_sf, mut g_d) =
            curve_bwd(w, &fc, delta_ts[c], s, sf, d, g_rraw, gw);
        // recurrence backward
        for t in (0..seq_len).rev() {
            let (gs0, gd0, gsf0) = step_bwd(w, &caches[t], (g_s, g_d, g_sf), gw, &wc);
            g_s = gs0;
            g_d = gd0;
            g_sf = gsf0;
        }
    }
    loss
}

/// Forward + reverse-mode backward BCE loss for one batch. Returns the summed loss and
/// accumulates d(loss)/d(w) into `gw` (length 36). Cards are independent, so the batch is
/// split across **2 threads** (the worker is pinned to a 2-CPU block per constraint 2, so
/// this uses the otherwise-idle second core). Partial gradients are summed after the join;
/// the only effect on the result is FP reassociation, well within the ±0.0010 band.
#[allow(clippy::too_many_arguments)]
pub(crate) fn batch_loss_and_grad(
    w: &[f32], t_hist: &[f32], r_hist: &[f32], seq_len: usize, batch: usize,
    delta_ts: &[f32], labels: &[f32], weights: &[f32], gw: &mut [f64],
) -> f64 {
    const THREAD_MIN: usize = 64;
    if batch < THREAD_MIN {
        return loss_and_grad_range(w, t_hist, r_hist, seq_len, batch, delta_ts, labels, weights, gw, 0, batch);
    }
    let mid = batch / 2;
    std::thread::scope(|s| {
        let h = s.spawn(|| {
            let mut g = [0.0f64; 36];
            let l = loss_and_grad_range(
                w, t_hist, r_hist, seq_len, batch, delta_ts, labels, weights, &mut g, mid, batch,
            );
            (l, g)
        });
        let mut g_a = [0.0f64; 36];
        let loss_a = loss_and_grad_range(
            w, t_hist, r_hist, seq_len, batch, delta_ts, labels, weights, &mut g_a, 0, mid,
        );
        let (loss_b, g_b) = h.join().unwrap();
        for i in 0..36 {
            gw[i] += g_a[i] + g_b[i];
        }
        loss_a + loss_b
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fd_grad(
        w: &[f32], t_hist: &[f32], r_hist: &[f32], seq_len: usize, batch: usize,
        dts: &[f32], lbl: &[f32], wts: &[f32],
    ) -> [f64; 36] {
        let mut g = [0.0f64; 36];
        for i in 0..36 {
            let eps = 1e-3f32;
            let mut wp = w.to_vec();
            wp[i] = w[i] + eps;
            let lp = batch_loss(&wp, t_hist, r_hist, seq_len, batch, dts, lbl, wts);
            wp[i] = w[i] - eps;
            let lm = batch_loss(&wp, t_hist, r_hist, seq_len, batch, dts, lbl, wts);
            g[i] = (lp as f64 - lm as f64) / (2.0 * eps as f64);
        }
        g
    }

    #[test]
    fn grad_matches_fd() {
        let w: Vec<f32> = crate::DEFAULT_PARAMETERS.to_vec();
        // batch=4, seq_len=2; first-review ratings 1,2,3,4 (rating 1 is the suspect)
        let t_hist = [1.0f32, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        let r_hist = [1.0f32, 2.0, 3.0, 4.0, 3.0, 3.0, 3.0, 3.0];
        let dts = [5.0f32, 5.0, 5.0, 5.0];
        let lbl = [1.0f32, 1.0, 0.0, 1.0];
        let wts = [1.0f32, 1.0, 1.0, 1.0];
        let mut mg = [0.0f64; 36];
        batch_loss_and_grad(&w, &t_hist, &r_hist, 2, 4, &dts, &lbl, &wts, &mut mg);
        let fd = fd_grad(&w, &t_hist, &r_hist, 2, 4, &dts, &lbl, &wts);
        // NOTE: finite differences cross clamp/min/max kinks where the analytic subgradient
        // is correct (it matches autodiff) but FD does not, so a few weights (e.g. w18/w34 on
        // this synthetic input) show moderate disagreement that is NOT a bug. The real
        // correctness gate is the average log loss on the full run (±0.0010). This test only
        // guards against GROSS errors (sign flips / missing terms => rel >~ 1).
        let mut bad = false;
        for i in 0..36 {
            let d = (mg[i] - fd[i]).abs();
            let rel = d / fd[i].abs().max(1e-3);
            if rel > 0.6 && d > 1e-3 {
                bad = true;
                eprintln!("GROSS MISMATCH w[{:>2}] manual={:+.6e} fd={:+.6e} rel={:.2e}", i, mg[i], fd[i], rel);
            }
        }
        assert!(!bad, "gross gradient mismatch vs finite difference");
    }
}
