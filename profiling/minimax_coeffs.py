"""Compute minimax (Remez) polynomial coefficients for the analytic.rs SIMD transcendentals.

Profiling-only (constraint 10): writes nothing to result/, never imported by the timed path.
Goal: replace the degree-6 *Taylor* poly in exp8 (and the degree-4-in-u atanh series in ln8)
with lower-degree *minimax* polys at equal-or-better accuracy, so the forward does fewer FMAs.

exp8 reduces x = n*ln2 + r, r in [-ln2/2, ln2/2], then exp(x) = 2^n * P(r).  We want a
RELATIVE-minimax P (weight 1/exp(r)) since the result's rel err = rel err of P vs exp(r).

ln8 reduces x = m*2^e, m in [1,2), t = (m-1)/(m+1) in [0,1/3], ln(x) = e*ln2 + 2*t*Q(t^2)
where Q(u) ~ atanh(sqrt u)/sqrt u = 1 + u/3 + u^2/5 + ...  We minimax Q in u=t^2, weighting by
t (since the ln error is 2*t*|dQ|) and report the reconstructed absolute ln error.
"""

import mpmath as mp

mp.mp.dps = 60


def remez(f, a, b, deg, weight=None, iters=60):
    """Minimax poly of given degree approximating f on [a,b], minimizing max weight(x)|p-f|.
    Returns (coeffs low->high, max_weighted_err, control_nodes)."""
    if weight is None:
        weight = lambda x: mp.mpf(1)
    # initial nodes: Chebyshev extrema mapped to [a,b]
    n = deg + 2
    nodes = [
        (a + b) / 2 + (b - a) / 2 * mp.cos(mp.pi * k / (n - 1)) for k in range(n)
    ]
    nodes = sorted(nodes)
    last_E = None
    for _ in range(iters):
        # Solve: sum_j c_j x_i^j - (-1)^i E / w(x_i) = f(x_i)
        A = mp.zeros(n, n)
        rhs = mp.zeros(n, 1)
        for i, x in enumerate(nodes):
            for j in range(deg + 1):
                A[i, j] = x**j
            A[i, deg + 1] = -(mp.mpf(-1) ** i) / weight(x)
            rhs[i] = f(x)
        sol = mp.lu_solve(A, rhs)
        coeffs = [sol[j] for j in range(deg + 1)]
        E = sol[deg + 1]

        def perr(x):  # weighted error
            p = sum(coeffs[j] * x**j for j in range(deg + 1))
            return weight(x) * (p - f(x))

        # find extrema of perr on a fine grid, refine, pick n alternating
        N = 4000
        grid = [a + (b - a) * k / N for k in range(N + 1)]
        vals = [perr(x) for x in grid]
        ext = []  # (x, err) local extrema incl endpoints
        ext.append((grid[0], vals[0]))
        for k in range(1, N):
            if (vals[k] - vals[k - 1]) * (vals[k + 1] - vals[k]) < 0:
                # local extremum near grid[k]; golden-refine
                lo, hi = grid[k - 1], grid[k + 1]
                for _ in range(40):
                    m1 = lo + (hi - lo) * 0.382
                    m2 = lo + (hi - lo) * 0.618
                    if abs(perr(m1)) > abs(perr(m2)):
                        hi = m2
                    else:
                        lo = m1
                xm = (lo + hi) / 2
                ext.append((xm, perr(xm)))
        ext.append((grid[N], vals[N]))
        # greedily pick n alternating-sign extrema of largest magnitude
        # collapse consecutive same-sign by keeping the larger
        pruned = [ext[0]]
        for x, e in ext[1:]:
            if e == 0:
                continue
            if (e > 0) == (pruned[-1][1] > 0):
                if abs(e) > abs(pruned[-1][1]):
                    pruned[-1] = (x, e)
            else:
                pruned.append((x, e))
        if len(pruned) >= n:
            # take the n with largest |e| while keeping alternation: trim ends
            while len(pruned) > n:
                if abs(pruned[0][1]) < abs(pruned[-1][1]):
                    pruned = pruned[1:]
                else:
                    pruned = pruned[:-1]
            nodes = [x for x, _ in pruned]
        maxerr = max(abs(e) for _, e in ext)
        if last_E is not None and abs(abs(E) - last_E) < mp.mpf(10) ** (-40):
            break
        last_E = abs(E)
    return coeffs, maxerr, nodes


LN2 = mp.log(2)
half = LN2 / 2

print("=" * 70)
print("exp(r), r in [-ln2/2, ln2/2], RELATIVE-minimax (weight 1/exp(r))")
print("=" * 70)
for deg in (3, 4, 5, 6):
    coeffs, err, _ = remez(mp.exp, -half, half, deg, weight=lambda x: 1 / mp.exp(x))
    print(f"\n  degree {deg}: max REL err = {mp.nstr(err, 4)}")
    for j, c in enumerate(coeffs):
        print(f"    c{j} = {mp.nstr(c, 12)}")

# Taylor baseline rel err for reference (current exp8 = degree-6 Taylor)
taylor = [mp.mpf(1) / mp.factorial(k) for k in range(7)]
def texp(x):
    return sum(taylor[k] * x**k for k in range(7))
terr = max(
    abs((texp(-half + (LN2) * k / 4000) - mp.exp(-half + LN2 * k / 4000)) / mp.exp(-half + LN2 * k / 4000))
    for k in range(4001)
)
print(f"\n  [current degree-6 Taylor rel err = {mp.nstr(terr, 4)}]")

print()
print("=" * 70)
print("ln: Q(u) ~ atanh(sqrt u)/sqrt u on u in [0, 1/9]; ln err = 2*t*|dQ|")
print("=" * 70)
umax = mp.mpf(1) / 9


def Qfun(u):
    if u == 0:
        return mp.mpf(1)
    t = mp.sqrt(u)
    return mp.atanh(t) / t


for deg in (1, 2, 3, 4):
    # unit weight (Remez on Q itself); reconstructed ln err reported empirically below
    coeffs, _, _ = remez(Qfun, mp.mpf(0), umax, deg)
    # reconstructed absolute ln error over m in [1,2)
    def lnapprox(m):
        t = (m - 1) / (m + 1)
        u = t * t
        Q = sum(coeffs[j] * u**j for j in range(deg + 1))
        return 2 * t * Q
    lnerr = max(abs(lnapprox(1 + k / 4000) - mp.log(1 + k / 4000)) for k in range(4001))
    print(f"\n  Q degree {deg} in u (=> ln through t^{2*deg+1}): max ABS ln err = {mp.nstr(lnerr, 4)}")
    for j, c in enumerate(coeffs):
        print(f"    q{j} = {mp.nstr(c, 12)}")

# current ln8 series rel: 1, 1/3, 1/5, 1/7, 1/9 (degree 4 in u)
cur = [mp.mpf(1) / (2 * j + 1) for j in range(5)]
def lncur(m):
    t = (m - 1) / (m + 1)
    u = t * t
    Q = sum(cur[j] * u**j for j in range(5))
    return 2 * t * Q
lncurerr = max(abs(lncur(1 + k / 4000) - mp.log(1 + k / 4000)) for k in range(4001))
print(f"\n  [current degree-4-in-u atanh series abs ln err = {mp.nstr(lncurerr, 4)}]")
