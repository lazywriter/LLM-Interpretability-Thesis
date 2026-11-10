#!/usr/bin/env python3
"""
Open-answer version of the easy-arithmetic control set: the same 144
expressions as easy_mcq/easy_mcq_test.jsonl (same ids), but no options. The
model has to write the number itself, so this tests the calculation alone,
without the step of picking a letter.

Two prompts (both kept per item):
  prompt          "... Answer with the number only."        (one step, like the MCQ)
  prompt_working  "... Show your working briefly, then finish with a final
                   line of the form "Answer: <number>"."    (room to compute)

  python gen_easy_open.py      # writes easy_open/easy_open_test.jsonl
"""
import json, os

INSTRUCTION = 'Compute the value of the expression below.'
NUMBER_ONLY = 'Answer with the number only.'
WORKING = 'Show your working briefly, then finish with a final line of the form "Answer: <number>".'


def main():
    src = [json.loads(l) for l in open('easy_mcq/easy_mcq_test.jsonl', encoding='utf-8')]
    os.makedirs('easy_open', exist_ok=True)
    with open('easy_open/easy_open_test.jsonl', 'w', encoding='utf-8') as f:
        for it in src:
            assert eval(it['expression']) == it['value']
            body = f"{INSTRUCTION}\n\nExpression: {it['expression']}\n\n"
            f.write(json.dumps({
                'id': it['id'], 'split': 'test', 'category': it['category'], 'order': it['order'],
                'expression': it['expression'], 'value': it['value'],
                'prompt': body + NUMBER_ONLY, 'prompt_working': body + WORKING,
            }) + '\n')
    print(f'wrote {len(src)} items to easy_open/easy_open_test.jsonl')


if __name__ == '__main__':
    main()
