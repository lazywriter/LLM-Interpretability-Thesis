#!/usr/bin/env python3
"""Unit tests for grade_open.py answer extraction and parsing.

  python ivp_open/test_grade_open.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp  # noqa: E402
from grade_open import extract_answer, parse_answer, x  # noqa: E402

E = sp.exp
PARSE = [  # (answer text, expected expression or failure status)
    (r'y = 3*exp(2*x) - x^2', 3 * E(2 * x) - x**2),
    (r'y = 3e^{2x} - x^2', 3 * E(2 * x) - x**2),
    (r'y = -e^{2x} - e^x', -E(2 * x) - E(x)),
    (r'\[ y = -e^{2x} - e^x. \]', -E(2 * x) - E(x)),              # display math + full stop
    (r'$y = e^{-x}$', E(-x)),
    (r'y = e^-x', E(-x)),
    (r'y(x) = \frac{1}{2}x^2 + 3', x**2 / 2 + 3),
    (r'\boxed{y = 2\sin(3x) - \cos x}', 2 * sp.sin(3 * x) - sp.cos(x)),
    (r'y = -1 \cdot e^{2x} - e^x = -e^{2x} - e^x', -E(2 * x) - E(x)),  # last "=" wins
    (r'y = -4 \ln|x| + \frac{1}{8} + 4 \ln 2', -4 * sp.log(sp.Abs(x)) + sp.Rational(1, 8) + 4 * sp.log(2)),
    (r'y = 4 \ln 3 - 2 \ln |x|', 4 * sp.log(3) - 2 * sp.log(sp.Abs(x))),
    (r'y = -2 \cdot |x|^{-4}', -2 * sp.Abs(x)**-4),
    (r'y = 6 \cosh(3x)', 6 * sp.cosh(3 * x)),
    (r'y = 2\sqrt{x} + \pi', 2 * sp.sqrt(x) + sp.pi),
    (r'y = (x + 1) e^{-2x}', (x + 1) * E(-2 * x)),
    (r'y = 3 - cos(3*x) for all x', 3 - sp.cos(3 * x)),
    (r'**y = x^3 - 1**', x**3 - 1),
    (r'y = C_1 e^{x}', 'free_symbols'),
    (r'y = -4 e^{-2t}', -4 * E(-2 * x)),                          # written in t: accepted (status ok_variable_t)
    (r'y = C_1 t + x', 'free_symbols'),
    (r'The particular solution is:', 'parse_error'),
    (None, 'no_answer'),
]
EXTRACT = [  # (reply, expected answer text, how found)
    ('work\nAnswer: y = x^2', ' y = x^2', 'answer_line'),
    ('work\n**Answer:** y = x^2\n', '** y = x^2', 'answer_line'),      # the parser drops the **
    ('Answer:\n$$ y = x^2 $$', '$$ y = x^2 $$', 'answer_line'),
    ('so \\boxed{y = x} done', 'y = x', 'boxed'),
    ('y = x + C\nthen\ny = x + 1', 'y = x + 1', 'last_y_line'),
    ('nothing here', None, 'none'),
    ('**Final Answer:** The solution to the initial value problem is:\n\\[ \\boxed{y = e^x} \\]',
     'y = e^x', 'answer_line'),                                     # expression on the next line
]


def close(a, b):
    return all(abs(complex(sp.N((a - b).subs(x, v)))) < 1e-9 for v in (0.7, 1.3, 2.1))


def main():
    bad = 0
    for s, want in PARSE:
        e, st = parse_answer(s)
        ok = (st == want) if isinstance(want, str) else (st.startswith('ok') and close(e, want))
        if not ok:
            bad += 1
            print('PARSE FAIL', repr(s), '->', st, e, 'expected', want)
    for reply, want, how in EXTRACT:
        got = extract_answer(reply)
        if got != (want, how):
            bad += 1
            print('EXTRACT FAIL', repr(reply), '->', got, 'expected', (want, how))
    n = len(PARSE) + len(EXTRACT)
    print(f'{n - bad}/{n} passed')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
