#!/usr/bin/env python3
"""
Grade open-ended answers to IVP-OPEN items.

  load(path)                  -> list of items
  extract_answer(reply)       -> (answer string or None, how it was found)
  parse_answer(s)             -> (sympy expression in x or None, status)
  grade_reply(item, reply)    -> dict: correct, status, and what the answer does
                                 (solves the ODE? which conditions does it
                                 satisfy? does it equal the z-as-y trap?)
  evaluate(items, graded)     -> (accuracy, per-slice report)

Answers are found on the last "Answer:" line, else in the last \\boxed{...},
else on the last line containing "y =". Plain text and common LaTeX are both
accepted. Equivalence with the true solution is checked numerically at five
points in the domain (relative tolerance 1e-6), which is exact for these
function classes in practice and never loops on hard simplifications.
"""
import json, re
from collections import defaultdict
import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, convert_xor,
                                        implicit_multiplication_application)

x = sp.Symbol('x')
LOCALS = {'x': x, 'e': sp.E, 'E': sp.E, 'exp': sp.exp, 'sin': sp.sin, 'cos': sp.cos,
          'tan': sp.tan, 'log': sp.log, 'ln': sp.log, 'sqrt': sp.sqrt, 'pi': sp.pi,
          'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh, 'Abs': sp.Abs, 'abs': sp.Abs}
TRANS = standard_transformations + (implicit_multiplication_application, convert_xor)
COND_RE = re.compile(r"^\s*(?:(-?\d+)\*)?([yz])('*)\(([^)]*)\)\s*=\s*(.+?)\s*$")


def load(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f]


# ----------------------------------------------------------------- extraction
def _boxed(s):
    i = s.rfind('\\boxed{')
    if i < 0:
        return None
    j, depth = i + len('\\boxed{'), 1
    k = j
    while k < len(s) and depth:
        depth += {'{': 1, '}': -1}.get(s[k], 0)
        k += 1
    return s[j:k - 1] if depth == 0 else None


def extract_answer(reply):
    if not reply:
        return None, 'none'
    m = list(re.finditer(r'answer\s*(?:is)?\s*\**\s*[:：]', reply, re.I))
    if m:
        rest = reply[m[-1].end():]
        lines = [l for l in rest.split('\n') if l.strip(' *$\\[]()')]
        if lines:
            line = lines[0]
            # "Answer: The solution ... is:" with the expression on a following line
            if '=' not in line and '\\boxed' not in line:
                line = next((l for l in lines[1:4] if '=' in l or '\\boxed' in l), line)
            b = _boxed(line)
            return (b if b is not None else line), 'answer_line'
    b = _boxed(reply)
    if b is not None:
        return b, 'boxed'
    ys = [l for l in reply.split('\n') if re.search(r'\by\s*(\(x\))?\s*=', l)]
    if ys:
        return ys[-1], 'last_y_line'
    return None, 'none'


# ----------------------------------------------------------------- parsing
def latex_to_plain(s):
    s = s.replace('−', '-').replace('·', '*').replace('×', '*').replace('π', 'pi')
    s = re.sub(r'\\(left|right|displaystyle|,|;|!|quad)', ' ', s)
    s = s.replace('\\ ', ' ')
    s = re.sub(r'\\(cdot|times)', '*', s)
    s = re.sub(r'\\[dt]frac', r'\\frac', s)
    for _ in range(10):  # innermost \frac{A}{B} first
        s2 = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'((\1)/(\2))', s)
        if s2 == s:
            break
        s = s2
    s = re.sub(r'\\sqrt\{([^{}]*)\}', r'sqrt(\1)', s)
    s = re.sub(r'\\(exp|sinh|cosh|tanh|sin|cos|tan|ln|log|pi)\b', r' \1', s)
    s = re.sub(r'\|([^|]+)\|', r' Abs(\1)', s)                       # |x| -> Abs(x)
    s = re.sub(r'(?<![A-Za-z])e\^\{', 'exp{', s)                     # e^{...} -> exp(...)
    s = re.sub(r'(?<![A-Za-z])e\^(\(?-?\w+(?:\.\d+)?\)?)', r'exp(\1)', s)  # e^x, e^-x, e^(2x); not "e^x."
    s = s.replace('{', '(').replace('}', ')')
    return s


def parse_answer(s):
    """Answer text -> (sympy expr, 'ok') or (None, reason)."""
    if s is None:
        return None, 'no_answer'
    s = s.strip()
    b = _boxed(s)
    if b is not None:
        s = b
    s = re.sub(r'\*\*|`|\$|\\\(|\\\)|\\\[|\\\]', ' ', s)
    s = latex_to_plain(s)
    if '=' in s:
        s = s.rsplit('=', 1)[1]
    s = re.split(r'\s(?:for|where|with|when)\s', s)[0]
    s = s.strip().rstrip('.;,').strip()
    if not s:
        return None, 'empty'
    try:
        e = parse_expr(s, local_dict=LOCALS, transformations=TRANS, evaluate=True)
    except Exception:
        return None, 'parse_error'
    if not isinstance(e, sp.Expr):
        return None, 'parse_error'
    t = sp.Symbol('t')
    if e.free_symbols == {t}:   # the right kind of answer written in t instead of x: accepted, but flagged
        return e.subs(t, x), 'ok_variable_t'
    if e.free_symbols - {x}:
        return None, 'free_symbols'
    return e, 'ok'


# ----------------------------------------------------------------- checking
def _num(e, v):
    return complex(sp.N(e.subs(x, v), 30))


def _points(item):
    return [1.3, 1.9, 2.6, 3.4, 4.1] if item['category'] == 'separable_y_over_x' else \
        [-0.9, -0.3, 0.4, 1.1, 1.7]


def _close(a, b):
    return abs(a - b) <= 1e-6 * (1 + abs(b))


def equivalent(e, truth, pts):
    try:
        return all(_close(_num(e, p), _num(truth, p)) for p in pts)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError):
        return False


def _ode_residual(equation, f):
    lhs, rhs = equation.split('=')
    D2, D1, Y = sp.symbols('D2 D1 Y')
    tr = lambda t: parse_expr(t.replace("y''", 'D2').replace("y'", 'D1').replace('y', 'Y').replace('^', '**'),
                              local_dict={**LOCALS, 'D2': D2, 'D1': D1, 'Y': Y})
    return (tr(lhs) - tr(rhs)).subs({D2: sp.diff(f, x, 2), D1: sp.diff(f, x), Y: f})


def _cond_value(f, text):
    k, fn, primes, pt, val = COND_RE.match(text).groups()
    lhs = (int(k) if k else 1) * sp.diff(f, x, len(primes)).subs(x, parse_expr(pt, local_dict=LOCALS))
    return lhs, parse_expr(val.replace('^', '**'), local_dict=LOCALS)


def truth_expr(item):
    return sp.sympify(item['answer_text'].split('=', 1)[1].replace('^', '**'), locals=LOCALS)


def grade_reply(item, reply):
    ans, found = extract_answer(reply)
    e, status = parse_answer(ans)
    out = {'found_by': found, 'answer_raw': ans, 'status': status, 'answer_parsed': None,
           'correct': False, 'solves_ode': None, 'satisfies': None, 'z_as_y': None, 'equals_trap': None}
    if e is None:
        return out
    out['answer_parsed'] = sp.sstr(e)
    pts = _points(item)
    out['correct'] = equivalent(e, truth_expr(item), pts)
    try:
        r = _ode_residual(item['equation'], e)
        out['solves_ode'] = all(abs(_num(r, p)) <= 1e-6 * (1 + abs(_num(e, p))) for p in pts)
    except Exception:
        out['solves_ode'] = False
    sat = []
    for k, c in enumerate(item['conditions'], 1):
        try:
            lhs, v = _cond_value(e, c['text'])
            ok = _close(complex(sp.N(lhs, 30)), complex(sp.N(v, 30)))
        except Exception:
            ok = False
        if c['function'] == 'y' and ok:
            sat.append(k)
        if c['function'] == 'z':
            out['z_as_y'] = ok          # the z-condition holds if z is read as y
    out['satisfies'] = sat
    if item['info'].get('trap_solution'):
        trap = sp.sympify(item['info']['trap_solution'].split('=', 1)[1].replace('^', '**'), locals=LOCALS)
        out['equals_trap'] = equivalent(e, trap, pts)
    return out


# ----------------------------------------------------------------- reporting
def evaluate(items, graded):
    slices = defaultdict(lambda: [0, 0])
    status = defaultdict(int)
    unrel = [0, 0, 0]
    for it, g in zip(items, graded):
        status[g['status']] += 1
        for key in ('all', f"ode_type={it['ode_type']}", f"category={it['category']}",
                    f"subtype={it['subtype']}", f"order={it['order']}"):
            slices[key][0] += g['correct']
            slices[key][1] += 1
        if it['subtype'] == 'unrelated':
            unrel[0] += 1
            unrel[1] += bool(g['z_as_y'])
            unrel[2] += bool(g['equals_trap'])
    report = {k: {'acc': round(c / n, 4), 'n': n} for k, (c, n) in sorted(slices.items())}
    report['answer_status'] = dict(status)
    if unrel[0]:
        report['unrelated: answer satisfies z-condition read as y'] = round(unrel[1] / unrel[0], 4)
        report['unrelated: answer equals the trap solution'] = round(unrel[2] / unrel[0], 4)
    return report['all']['acc'], report
