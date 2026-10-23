"""Collect real multi-turn replies for human scoring. Never labels mock output as passed."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))
from app.services.chat_pipeline import run_chat
from app.services.llm_service import ChatUnavailable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, default=API_ROOT / 'tests/fixtures/chat_acceptance_cases.json')
    parser.add_argument('--output', type=Path, default=API_ROOT / 'work/chat-acceptance/live-results.json')
    args = parser.parse_args()
    results = {'status': 'pending_human_review', 'cases': []}
    if not os.environ.get('DEEPSEEK_API_KEY', '').strip():
        results.update(status='blocked', reason='DEEPSEEK_API_KEY 未配置；未调用模型，不能验收自然度。')
    else:
        for case in json.loads(args.cases.read_text()):
            history = []
            record = {'id': case['id'], 'expect': case['expect'], 'turns': [], 'scores': None}
            results['cases'].append(record)
            for text in case['turns']:
                history.append({'role': 'user', 'content': text})
                try:
                    response = run_chat(history)
                except ChatUnavailable as exc:
                    results.update(status='blocked', reason=str(exc))
                    break
                reply = response['assistant']['content']
                record['turns'].append({'user': text, 'assistant': reply, 'strategy': response['strategy'], 'citations': response['citations']})
                history.append({'role': 'assistant', 'content': reply})
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
                print(f"{case['id']}: {len(record['turns'])}/{len(case['turns'])}", flush=True)
            if results['status'] == 'blocked':
                break
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': results['status'], 'output': str(args.output), 'reason': results.get('reason')}, ensure_ascii=False))
    return 2 if results['status'] == 'blocked' else 0


if __name__ == '__main__':
    raise SystemExit(main())
