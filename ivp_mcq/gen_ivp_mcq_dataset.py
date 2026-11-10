#!/usr/bin/env python3
"""
Generate IVP-MCQ: a balanced multiple-choice dataset of initial value problems
with MORE conditions than needed, for zero-shot interpretability experiments.

Two variants
------------
B  "distractor"   : n sufficient conditions + 1 pure distractor that carries no
                    new information:
                      duplicate  - one condition restated verbatim
                      rescaled   - one condition restated as k*y(x0) = k*v
                      unrelated  - a condition on a different function z
                                   (with a TRAP option = the solution you get if
                                   you wrongly read z as y)
C  "inconsistent" : n sufficient conditions + 1 or 2 extra conditions that
                    CONTRADICT them. Exactly one option satisfies a subset of
                    the conditions large enough to determine the solution.
                      direct conflict   - an extra condition at the same point
                                          and derivative order as a sufficient
                                          one, with a different value
                      indirect conflict - extra conditions at other points

Construction (why the labels are trustworthy)
---------------------------------------------
Every family has a general solution that is AFFINE in the constants:
    y = C1*phi1(x) + C2*phi2(x) + yp(x)
so every condition y^(d)(x0) = v is a linear equation a.C + b = v. Options are
chosen first (small non-zero INTEGER constants, so no option looks "nicer"),
condition values are derived from them, and achievability is decided by exact
linear algebra: an option is achievable iff the conditions it satisfies contain
a subset of rank n (n = order). Each item is accepted only if exactly one
option is achievable. An independent checker (check_ivp_mcq.py) re-verifies
every item from the rendered TEXT using sympy.dsolve.
"""
import argparse, json, os, random, re, time
from collections import Counter, defaultdict
from fractions import Fraction
from math import gcd
import sympy as sp

x = sp.symbols('x')
yf = sp.Function('y')
YS = sp.Symbol('y')          # plain symbol used only for pretty-printing RHS
PI = sp.pi
PRIME = "'"
YP, YPP = "y'", "y''"

PTS_DEFAULT = [sp.Integer(0), sp.Integer(1), sp.Integer(-1)]
PTS_TRIG = [sp.Integer(0), PI / 2, PI]
PTS_POS = [sp.Integer(1), sp.Integer(2), sp.Integer(3)]


# ------------------------------- formatting --------------------------------- #
def txt(e) -> str:
    s = sp.sstr(sp.expand(e)).replace('**', '^')
    s = re.sub(r'\bE\b', 'exp(1)', s)       # SymPy prints e as a bare 'E'
    return re.sub(r'\s+', ' ', s).strip()


def lhs_terms(terms):
    """[(coef, "y''"), (p, "y'"), (q, "y")] -> "y'' - 3*y' + 2*y"."""
    out = []
    for c, name in terms:
        c = sp.nsimplify(c)
        if c == 0:
            continue
        mag = abs(c)
        body = name if mag == 1 else f'{txt(mag)}*{name}'
        if not out:
            out.append(body if c > 0 else f'-{body}')
        else:
            out.append(('+ ' if c > 0 else '- ') + body)
    return ' '.join(out)


def point_txt(p):
    return txt(p)


def cond_txt(d, p, v, fn='y', scale=1):
    lhs = f"{fn}{PRIME * d}({point_txt(p)})"
    if scale != 1:
        return f'{scale}*{lhs} = {txt(scale * v)}'
    return f'{lhs} = {txt(v)}'


# ------------------------------- families ----------------------------------- #
def nz(rng, lo, hi):
    return rng.choice([i for i in range(lo, hi + 1) if i != 0])


def fam(rng, cat):
    """Return dict(ode, ode_text, basis, yp, pts, n, ode_type)."""
    D1, D2 = sp.Derivative(yf(x), x), sp.Derivative(yf(x), (x, 2))

    # ---------------- first order: direct integration ----------------
    if cat == 'direct_power':
        k, m = rng.randint(0, 4), nz(rng, -4, 4)
        rhs = m * (k + 1) * x**k
        return dict(ode=sp.Eq(D1, rhs), ode_text=f"y' = {txt(rhs)}",
                    basis=[sp.Integer(1)], yp=m * x**(k + 1), pts=PTS_DEFAULT,
                    ode_type='first_order_direct', n=1)
    if cat == 'direct_trig':
        b, m, f = rng.randint(1, 3), nz(rng, -4, 4), rng.choice(['sin', 'cos'])
        if f == 'sin':
            rhs, yp = m * b * sp.sin(b * x), -m * sp.cos(b * x)
        else:
            rhs, yp = m * b * sp.cos(b * x), m * sp.sin(b * x)
        return dict(ode=sp.Eq(D1, rhs), ode_text=f"y' = {txt(rhs)}",
                    basis=[sp.Integer(1)], yp=yp, pts=PTS_TRIG,
                    ode_type='first_order_direct', n=1)
    if cat == 'direct_exp':
        b, m = nz(rng, -3, 3), nz(rng, -4, 4)
        rhs = m * b * sp.exp(b * x)
        return dict(ode=sp.Eq(D1, rhs), ode_text=f"y' = {txt(rhs)}",
                    basis=[sp.Integer(1)], yp=m * sp.exp(b * x), pts=PTS_DEFAULT,
                    ode_type='first_order_direct', n=1)

    # ---------------- first order: separable ----------------
    if cat == 'separable_exponential':
        k = nz(rng, -5, 5)
        return dict(ode=sp.Eq(D1, k * yf(x)), ode_text=f"y' = {txt(k * YS)}",
                    basis=[sp.exp(k * x)], yp=sp.Integer(0), pts=PTS_DEFAULT,
                    ode_type='first_order_separable', n=1)
    if cat == 'separable_xy':
        k = nz(rng, -3, 3)
        return dict(ode=sp.Eq(D1, 2 * k * x * yf(x)),
                    ode_text=f"y' = {txt(2 * k * x * YS)}",
                    basis=[sp.exp(k * x**2)], yp=sp.Integer(0), pts=PTS_DEFAULT,
                    ode_type='first_order_separable', n=1)
    if cat == 'separable_y_over_x':
        k = nz(rng, -4, 4)
        return dict(ode=sp.Eq(D1, k * yf(x) / x),
                    ode_text=f"y' = {txt(k * YS / x)}",
                    basis=[x**k], yp=sp.Integer(0), pts=PTS_POS,
                    ode_type='first_order_separable', n=1)

    # ---------------- first order: linear ----------------
    if cat == 'linear_const_forcing':
        a, m = nz(rng, -5, 5), nz(rng, -5, 5)
        return dict(ode=sp.Eq(D1 + a * yf(x), a * m),
                    ode_text=f"{lhs_terms([(1, YP), (a, 'y')])} = {txt(a * m)}",
                    basis=[sp.exp(-a * x)], yp=sp.Integer(m), pts=PTS_DEFAULT,
                    ode_type='first_order_linear', n=1)
    if cat == 'linear_poly_forcing':
        a, c = nz(rng, -3, 3), nz(rng, -6, 6)
        yp = sp.Rational(c, a) * x - sp.Rational(c, a**2)
        return dict(ode=sp.Eq(D1 + a * yf(x), c * x),
                    ode_text=f"{lhs_terms([(1, YP), (a, 'y')])} = {txt(c * x)}",
                    basis=[sp.exp(-a * x)], yp=yp, pts=PTS_DEFAULT,
                    ode_type='first_order_linear', n=1)
    if cat == 'linear_exp_forcing':
        while True:
            a, b = nz(rng, -3, 3), nz(rng, -3, 3)
            if a + b != 0:
                break
        m = nz(rng, -3, 3)
        rhs = m * (a + b) * sp.exp(b * x)
        return dict(ode=sp.Eq(D1 + a * yf(x), rhs),
                    ode_text=f"{lhs_terms([(1, YP), (a, 'y')])} = {txt(rhs)}",
                    basis=[sp.exp(-a * x)], yp=m * sp.exp(b * x), pts=PTS_DEFAULT,
                    ode_type='first_order_linear', n=1)

    # ---------------- second order: direct integration ----------------
    if cat == 'direct2_power':
        k, m = rng.randint(0, 3), nz(rng, -3, 3)
        rhs = m * (k + 1) * (k + 2) * x**k
        return dict(ode=sp.Eq(D2, rhs), ode_text=f"y'' = {txt(rhs)}",
                    basis=[x, sp.Integer(1)], yp=m * x**(k + 2), pts=PTS_DEFAULT,
                    ode_type='second_order_direct', n=2)
    if cat == 'direct2_trig':
        b, m, f = rng.randint(1, 3), nz(rng, -3, 3), rng.choice(['sin', 'cos'])
        g = sp.sin if f == 'sin' else sp.cos
        rhs = -m * b**2 * g(b * x)
        return dict(ode=sp.Eq(D2, rhs), ode_text=f"y'' = {txt(rhs)}",
                    basis=[x, sp.Integer(1)], yp=m * g(b * x), pts=PTS_TRIG,
                    ode_type='second_order_direct', n=2)
    if cat == 'direct2_exp':
        b, m = nz(rng, -3, 3), nz(rng, -3, 3)
        rhs = m * b**2 * sp.exp(b * x)
        return dict(ode=sp.Eq(D2, rhs), ode_text=f"y'' = {txt(rhs)}",
                    basis=[x, sp.Integer(1)], yp=m * sp.exp(b * x), pts=PTS_DEFAULT,
                    ode_type='second_order_direct', n=2)

    # ---------------- second order: constant coefficients ----------------
    forced = cat.startswith('nonhom_')
    root = cat.replace('nonhom_', '').replace('hom_', '')
    if root == 'real_distinct':
        while True:
            r1, r2 = sorted(rng.sample(range(-3, 4), 2))
            if not forced or (r1 != 0 and r2 != 0):
                break
        p, q = -(r1 + r2), r1 * r2
        basis, pts = [sp.exp(r1 * x), sp.exp(r2 * x)], PTS_DEFAULT
    elif root == 'repeated':
        r = nz(rng, -3, 3)
        p, q = -2 * r, r * r
        basis, pts = [sp.exp(r * x), x * sp.exp(r * x)], PTS_DEFAULT
    elif root == 'complex':
        a, b = rng.randint(-2, 2), rng.randint(1, 3)
        p, q = -2 * a, a * a + b * b
        basis = [sp.exp(a * x) * sp.cos(b * x), sp.exp(a * x) * sp.sin(b * x)]
        pts = PTS_TRIG
    else:
        raise ValueError(cat)
    if forced:
        m = nz(rng, -3, 3)
        rhs, yp = q * m, sp.Integer(m)
        otype = 'second_order_nonhomogeneous'
    else:
        rhs, yp = 0, sp.Integer(0)
        otype = 'second_order_homogeneous'
    ode = sp.Eq(D2 + p * D1 + q * yf(x), rhs)
    otext = f"{lhs_terms([(1, YPP), (p, YP), (q, 'y')])} = {txt(rhs)}"
    return dict(ode=ode, ode_text=otext, basis=basis, yp=yp, pts=pts,
                ode_type=otype, n=2)


CATEGORIES = {
    'first_order_direct': ['direct_power', 'direct_trig', 'direct_exp'],
    'first_order_separable': ['separable_exponential', 'separable_xy',
                              'separable_y_over_x'],
    'first_order_linear': ['linear_const_forcing', 'linear_poly_forcing',
                           'linear_exp_forcing'],
    'second_order_direct': ['direct2_power', 'direct2_trig', 'direct2_exp'],
    'second_order_homogeneous': ['hom_real_distinct', 'hom_repeated',
                                 'hom_complex'],
    'second_order_nonhomogeneous': ['nonhom_real_distinct', 'nonhom_repeated',
                                    'nonhom_complex'],
}


# ------------------------------- linear algebra ----------------------------- #
def functional(F, d, p):
    """Condition y^(d)(p): returns (coef vector a, offset b)."""
    a = [sp.nsimplify(sp.simplify(sp.diff(phi, x, d).subs(x, p))) for phi in F['basis']]
    b = sp.simplify(sp.diff(F['yp'], x, d).subs(x, p))
    return a, b


def num(e):
    return complex(sp.N(e, 40))


def is_zero(e):
    return abs(num(e)) < 1e-20


def val(a, b, C):
    return sp.simplify(sum(ai * ci for ai, ci in zip(a, C)) + b)


def rank_ok(rows, n):
    """True iff the coefficient rows (list of vectors) contain rank n."""
    if n == 1:
        return any(not is_zero(r[0]) for r in rows)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            det = rows[i][0] * rows[j][1] - rows[i][1] * rows[j][0]
            if not is_zero(det):
                return True
    return False


def satisfies(cond, C):
    return is_zero(val(cond['a'], cond['b'], C) - cond['v'])


def achievable(conds, C, n):
    rows = [c['a'] for c in conds if c['fn'] == 'y' and satisfies(c, C)]
    return rank_ok(rows, n)


def int_kernel(a):
    """Integer vector d != 0 with a.d = 0 for rational 2-vector a (else None)."""
    if not all(sp.nsimplify(ai).is_rational for ai in a):
        return None
    a1, a2 = [Fraction(str(sp.nsimplify(ai))) for ai in a]
    if a1 == 0 and a2 == 0:
        return None
    d1, d2 = a2, -a1
    den = d1.denominator * d2.denominator
    d1, d2 = int(d1 * den), int(d2 * den)
    g = gcd(abs(d1), abs(d2)) or 1
    return [d1 // g, d2 // g]


def rand_consts(rng, n, lo=-5, hi=5):
    return [sp.Integer(nz(rng, lo, hi)) for _ in range(n)]


def sol_expr(F, C):
    return sum(c * phi for c, phi in zip(C, F['basis'])) + F['yp']


def near_miss(rng, F, C, cond, avoid):
    """Integer constants that satisfy `cond` but differ from C (n=2 only)."""
    d = int_kernel(cond['a'])
    if d is None:
        return None
    for t in rng.sample([1, -1, 2, -2, 3, -3], 6):
        D = [C[0] + t * d[0], C[1] + t * d[1]]
        if all(di != 0 and abs(di) <= 9 for di in D) and D not in avoid:
            return [sp.Integer(v) for v in D]
    return None


INT_RE = re.compile(r'(?<![\w.])-?\d+(?![\w.])')


def far_miss(rng, n, conds, C):
    """Distractor constants. To stop 'which option's numbers appear in the
    conditions?' from being a shortcut, coordinates are often drawn from the
    integers visible in the condition texts (natural slips: reading y'(0) as
    y(0), ignoring the particular solution), and for n=2 sometimes copied from
    the correct option so no option is systematically 'central'."""
    pool = sorted({int(t) for c in conds for t in INT_RE.findall(c['text'].split('=', 1)[1])
                   if t not in ('0', '-0') and abs(int(t)) <= 9})
    mode = rng.choice(['pool', 'pool', 'share', 'random']) if n == 2 else \
        rng.choice(['pool', 'pool', 'random'])
    D = rand_consts(rng, n)
    if mode == 'pool' and pool:
        D = [sp.Integer(rng.choice(pool)) if rng.random() < 0.7 else d for d in D]
    elif mode == 'share':
        k = rng.randrange(2)
        D[k] = C[k]
    return D


def make_cond(F, d, p, C, fn='y', scale=1, role=''):
    a, b = functional(F, d, p)
    v = val(a, b, C)
    return dict(d=d, p=p, a=[scale * ai for ai in a], b=scale * b,
                v=scale * v, raw_v=v, fn=fn, scale=scale, role=role,
                text=cond_txt(d, p, v, fn=fn, scale=scale))


def cond_slots(F):
    slots = []
    for p in F['pts']:
        for d in (0, 1):
            a, _ = functional(F, d, p)
            if any(not is_zero(ai) for ai in a):
                slots.append((d, p))
    return slots


# ------------------------------- item builders ------------------------------ #
def pick_sufficient(rng, F, slots):
    n = F['n']
    for _ in range(50):
        S = rng.sample(slots, n)
        rows = [functional(F, d, p)[0] for d, p in S]
        if rank_ok(rows, n) and (n == 1 or len(rows) == 2 and rank_ok(rows, 2)):
            return S
    return None


def build_B(rng, F, kind):
    n, slots = F['n'], cond_slots(F)
    S = pick_sufficient(rng, F, slots)
    if S is None:
        return None
    C = rand_consts(rng, n)
    real = [make_cond(F, d, p, C, role='sufficient') for d, p in S]
    opts = [('correct', C)]
    info = {'distractor_kind': kind}

    if kind == 'unrelated':
        # mirror one sufficient condition onto z; trap = solution under misreading
        order = rng.sample(range(n), n)
        T = None
        for mi in order:
            if n == 1:
                T = rand_consts(rng, 1)
                if T == C:
                    T = None
                    continue
            else:
                other = real[1 - mi]
                T = near_miss(rng, F, C, other, [list(map(int, C))])
            if T is not None:
                break
        if T is None:
            return None
        md, mp = S[mi]
        zc = make_cond(F, md, mp, T, fn='z', role='distractor')
        opts.append(('trap', T))
        distractor = zc
        info['mirrors_condition'] = real[mi]['text']
    else:
        src = rng.choice(real)
        if kind == 'duplicate':
            distractor = dict(src, role='distractor', a=list(src['a']))
        else:  # rescaled
            k = rng.choice([2, 3, -2])
            distractor = make_cond(F, src['d'], src['p'], C, scale=k, role='distractor')
        info['restates_condition'] = src['text']

    # remaining distractor options: near-miss (n=2) where possible, else far-miss
    while len(opts) < 3:
        D = None
        if n == 2 and rng.random() < 0.6:
            D = near_miss(rng, F, C, rng.choice(real), [list(map(int, o[1])) for o in opts])
        if D is None:
            D = far_miss(rng, n, real + [distractor], C)
        if any(D == o[1] for o in opts):
            continue
        opts.append(('distractor', D))

    conds = real + [distractor]
    return conds, opts, info


def build_C(rng, F, n_extra, conflict):
    n, slots = F['n'], cond_slots(F)
    S = pick_sufficient(rng, F, slots)
    if S is None:
        return None
    C = rand_consts(rng, n)
    real = [make_cond(F, d, p, C, role='sufficient') for d, p in S]

    # choose slots for the extra (contradicting) conditions
    free = [s for s in slots if s not in S]
    extra_slots = []
    if conflict == 'direct':
        extra_slots.append(rng.choice(S))
        if n_extra == 2:
            if not free:
                return None
            extra_slots.append(rng.choice(free))
    else:
        if len(free) < n_extra:
            return None
        extra_slots = rng.sample(free, n_extra)

    # n=2: D2 is built first and is the option that satisfies the first extra
    D2 = far_miss(rng, n, real, C) if n == 2 else None
    extras = []
    for i, (d, p) in enumerate(extra_slots):
        src = D2 if (n == 2 and i == 0) else rand_consts(rng, n)  # else phantom
        extras.append(make_cond(F, d, p, src, role='contradicting'))
    # D1: near-miss on a sufficient condition (n=2) or a far-miss whose numbers
    # are drawn from the visible conditions
    D1 = None
    if n == 2 and rng.random() < 0.6:
        D1 = near_miss(rng, F, C, rng.choice(real), [list(map(int, C))])
    if D1 is None:
        D1 = far_miss(rng, n, real + extras, C)
    if D2 is None:
        D2 = far_miss(rng, n, real + extras, C)
    opts = [('correct', C), ('distractor', D1), ('distractor', D2)]
    if len({tuple(o[1]) for o in opts}) < 3:
        return None
    # contradiction must be real: correct option violates every extra
    if any(satisfies(e, C) for e in extras):
        return None
    # no two listed conditions may be identical
    texts = [c['text'] for c in real + extras]
    if len(set(texts)) < len(texts):
        return None
    info = {'n_extra': n_extra, 'conflict': conflict}
    return real + extras, opts, info


# ------------------------------- rendering ---------------------------------- #
PROMPT_B = ("Solve the initial value problem below and choose the correct "
            "solution.\n\n")
PROMPT_C = ("Consider the differential equation below together with the listed "
            "conditions. The conditions contradict each other, so no function "
            "satisfies all of them. Exactly one of the options is a solution of "
            "the differential equation that satisfies a subset of the conditions "
            "large enough to determine the solution uniquely.\n\n")
TAIL = "\nAnswer with the letter of the correct option only."
LETTERS = 'ABC'


def render(variant, ode_text, cond_texts, opt_texts):
    spans = {}
    s = PROMPT_B if variant == 'B' else PROMPT_C
    spans['instruction'] = [0, len(s)]
    s += 'Differential equation: '
    a = len(s); s += ode_text; spans['equation'] = [a, len(s)]
    s += '\nConditions:\n'
    spans['conditions'] = []
    for i, c in enumerate(cond_texts):
        s += f'{i + 1}. '
        a = len(s); s += c; spans['conditions'].append([a, len(s)])
        s += '\n'
    s += '\nOptions:\n'
    spans['options'] = []
    for L, o in zip(LETTERS, opt_texts):
        s += f'{L}) '
        a = len(s); s += o; spans['options'].append([a, len(s)])
        s += '\n'
    a = len(s); s += TAIL.lstrip('\n'); spans['answer_instruction'] = [a, len(s)]
    return s, spans


# ------------------------------- shortcut audit ----------------------------- #
def _credit(scores, ans):
    m = max(scores)
    top = [i for i, v in enumerate(scores) if v == m]
    return (ans in top) / len(top)


def shortcut_credits(it):
    """Expected accuracy (ties split) of two answer-revealing heuristics that
    never look at the maths: (1) pick the option whose constants are shared
    most with the other options; (2) pick the option whose integers appear
    most often in the condition texts. The builder keeps both at chance."""
    ans = LETTERS.index(it['answer'])
    o = it['options']
    h1 = [sum(o[i]['constants'][k] == o[j]['constants'][k]
              for j in range(3) if j != i for k in range(len(o[i]['constants'])))
          for i in range(3)]
    nums = Counter(t for c in it['conditions'] for t in INT_RE.findall(c['text'].split('=', 1)[1]))
    h2 = [sum(nums[t] > 0 for t in INT_RE.findall(oo['text'].split('=', 1)[1])) for oo in o]
    return [_credit(h1, ans), _credit(h2, ans)]


# ------------------------------- main build --------------------------------- #
def make_item(rng, cat, variant, sub, letter, dpos):
    for _ in range(200):
        F = fam(rng, cat)
        out = build_B(rng, F, sub) if variant == 'B' else \
            build_C(rng, F, sub[0], sub[1])
        if out is None:
            continue
        conds, opts, info = out
        n = F['n']
        ycond = [c for c in conds if c['fn'] == 'y']
        # exactly one achievable option, and it is the correct one
        ach = [achievable(ycond, o[1], n) for o in opts]
        if ach != [True, False, False]:
            continue
        if variant == 'B' and not all(satisfies(c, opts[0][1]) for c in ycond):
            continue
        texts = [txt(sol_expr(F, o[1])) for o in opts]
        if len(set(texts)) < 3:
            continue

        # order conditions (B: distractor at a balanced position)
        if variant == 'B':
            real = [c for c in conds if c['role'] != 'distractor']
            dis = [c for c in conds if c['role'] == 'distractor'][0]
            rng.shuffle(real)
            pos = dpos % (len(real) + 1)
            ordered = real[:pos] + [dis] + real[pos:]
        else:
            ordered = conds[:]
            rng.shuffle(ordered)

        # place options: correct at the requested letter
        others = opts[1:]
        rng.shuffle(others)
        li = LETTERS.index(letter)
        placed = others[:li] + [opts[0]] + others[li:]
        opt_texts = ['y = ' + txt(sol_expr(F, o[1])) for o in placed]

        options = []
        for L, (role, Cv), t in zip(LETTERS, placed, opt_texts):
            sat = [i + 1 for i, c in enumerate(ordered)
                   if c['fn'] == 'y' and satisfies(c, Cv)]
            options.append(dict(letter=L, text=t, role=role,
                                constants=[int(c) for c in Cv],
                                satisfies_conditions=sat))
        cond_out = []
        for c in ordered:
            cond_out.append(dict(text=c['text'], role=c['role'], function=c['fn'],
                                 derivative_order=c['d'], point=point_txt(c['p']),
                                 value_complexity=int(sp.count_ops(c['raw_v']))))
        prompt, spans = render(variant, F['ode_text'], [c['text'] for c in ordered],
                               opt_texts)
        if variant == 'B':
            info['distractor_position'] = [c['role'] for c in ordered].index('distractor') + 1
            if sub == 'unrelated':
                info['trap_letter'] = [o['letter'] for o in options if o['role'] == 'trap'][0]
        else:
            info['sufficient_positions'] = [i + 1 for i, c in enumerate(ordered)
                                            if c['role'] == 'sufficient']
            info['contradicting_positions'] = [i + 1 for i, c in enumerate(ordered)
                                               if c['role'] == 'contradicting']
        Ccorr = opts[0][1]
        return dict(
            variant=variant, subtype=sub if variant == 'B' else f'extra{sub[0]}_{sub[1]}',
            ode_type=F['ode_type'], category=cat, order=n,
            n_required=n, n_conditions=len(ordered),
            equation=F['ode_text'], conditions=cond_out, options=options,
            answer=letter, answer_text='y = ' + txt(sol_expr(F, Ccorr)),
            prompt=prompt, spans=spans, info=info,
            sympy_ode=sp.srepr(F['ode']),
            sympy_general_solution=sp.srepr(
                sol_expr(F, sp.symbols(f'C1:{n + 1}'))),
        )
    raise RuntimeError(f'could not build item {cat} {variant} {sub}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-cell', type=int, default=24,
                    help='items per (category, variant); must divide by 12')
    ap.add_argument('--dev-per-cell', type=int, default=2)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--outdir', default='.')
    args = ap.parse_args()
    assert args.per_cell % 12 == 0, 'per-cell must be a multiple of 12'
    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)
    t0 = time.time()

    subs_B = ['duplicate', 'rescaled', 'unrelated']
    subs_C = [(1, 'direct'), (1, 'indirect'), (2, 'direct'), (2, 'indirect')]
    letter_ctr, pos_ctr = Counter(), Counter()
    hsum, hcnt = defaultdict(lambda: [0.0, 0.0]), Counter()
    items, seen = [], set()
    for otype, cats in CATEGORIES.items():
        for cat in cats:
            for variant, subs in (('B', subs_B), ('C', subs_C)):
                per_sub = args.per_cell // len(subs)
                for sub in subs:
                    for _ in range(per_sub):
                        key0 = (variant, otype, str(sub))
                        letter = LETTERS[letter_ctr[key0] % 3]
                        letter_ctr[key0] += 1
                        pkey = (variant, otype, str(sub))   # position balance
                        dpos = pos_ctr[pkey]
                        pos_ctr[pkey] += 1
                        strat = (variant, otype)
                        best, best_dev = None, None
                        for _try in range(12):
                            it = make_item(rng, cat, variant, sub, letter, dpos)
                            key = (it['equation'], tuple(c['text'] for c in it['conditions']))
                            if key in seen:
                                continue
                            cr = shortcut_credits(it)
                            N = hcnt[strat] + 1
                            dev = max(abs((hsum[strat][h] + cr[h]) / N - 1 / 3)
                                      for h in range(len(cr)))
                            if best is None or dev < best_dev:
                                best, best_dev, best_cr = it, dev, cr
                            if dev <= 0.04 or N < 4:
                                break
                        it = best
                        seen.add((it['equation'], tuple(c['text'] for c in it['conditions'])))
                        for h, c in enumerate(best_cr):
                            hsum[strat][h] += c
                        hcnt[strat] += 1
                        items.append(it)
            print(f'  built {cat:24s} total={len(items)}', flush=True)

    # stratified dev/test split
    by_cell = defaultdict(list)
    for it in items:
        by_cell[(it['category'], it['variant'])].append(it)
    dev, test = [], []
    for cell, rows in by_cell.items():
        rng.shuffle(rows)
        dev += rows[:args.dev_per_cell]
        test += rows[args.dev_per_cell:]
    for split, rows in (('dev', dev), ('test', test)):
        for r in rows:
            r['split'] = split
    allrows = dev + test
    rng.shuffle(allrows)
    for i, r in enumerate(sorted(allrows, key=lambda r: (r['split'], r['variant'],
                                                         r['ode_type'], r['category']))):
        r['id'] = f'ivp-{i:04d}'

    def dump(rows, name):
        rows = sorted(rows, key=lambda r: r['id'])
        with open(os.path.join(args.outdir, name), 'w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    dump(allrows, 'ivp_mcq_all.jsonl')
    dump(dev, 'ivp_mcq_dev.jsonl')
    dump(test, 'ivp_mcq_test.jsonl')
    print(f'DONE {len(allrows)} items (dev {len(dev)}, test {len(test)}) '
          f'in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
