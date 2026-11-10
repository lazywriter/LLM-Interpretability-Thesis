#!/usr/bin/env python3
"""Accuracy of answer-revealing heuristics that never do the maths (chance = 33.3%)."""
import json, re, sys
from collections import Counter
rows=[json.loads(l) for l in open(sys.argv[1])]
INT=re.compile(r'(?<![\w.])-?\d+(?![\w.])')
def acc(f, sel=lambda r: True):
    rs=[r for r in rows if sel(r)]; hit=0
    for r in rs:
        sc=[f(r,i) for i in range(3)]; m=max(sc); top=[i for i in range(3) if sc[i]==m]
        hit+=('ABC'.index(r['answer']) in top)/len(top)
    return 100*hit/len(rs)
def shared(r,i):
    c=r['options'][i]['constants']
    return sum(c[k]==o['constants'][k] for j,o in enumerate(r['options']) if j!=i for k in range(len(c)))
def nums(r,i):
    n=Counter(t for c in r['conditions'] for t in INT.findall(c['text'].split('=',1)[1]))
    return sum(n[t]>0 for t in INT.findall(r['options'][i]['text'].split('=',1)[1]))
H=[('most-shared-constants',shared),('numbers-in-conditions',nums),
   ('shortest-option',lambda r,i:-len(r['options'][i]['text'])),
   ('longest-option',lambda r,i:len(r['options'][i]['text'])),
   ('always-A',lambda r,i:i==0)]
S=[('all',lambda r:True),('B',lambda r:r['variant']=='B'),('C',lambda r:r['variant']=='C'),
   ('order1',lambda r:r['order']==1),('order2',lambda r:r['order']==2)]
print(f"{'heuristic':24s}"+''.join(f'{n:>8s}' for n,_ in S))
for name,f in H: print(f'{name:24s}'+''.join(f'{acc(f,s):8.1f}' for _,s in S))
print('chance = 33.3')
