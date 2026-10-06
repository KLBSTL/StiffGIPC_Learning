"""Summarize optional one-contact TOI activity from a diagnostic run."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--frame', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    timeline = []
    with (args.run / 'toi_diagnostic.jsonl').open() as source:
        for line in source:
            item = json.loads(line)
            if item['frame'] != args.frame:
                continue
            name = 'watched_contact_after_update' if 'accepted' in item else 'watched_contact_before_update'
            data = item.get('accepted', item)
            watch = data.get(name)
            if watch is not None:
                timeline.append({'outer': item['outer'], 'phase': name, **watch,
                                 'mu': item.get('mu'), 'alpha': data.get('alpha'),
                                 'full_ccd_alpha': data.get('full_ccd_alpha')})
    activated = [row for row in timeline if row['active']]
    result = {'frame': args.frame, 'observations': len(timeline),
              'first_active_outer': min((row['outer'] for row in activated), default=None),
              'last_active_outer': max((row['outer'] for row in activated), default=None),
              'active_observations': len(activated), 'timeline': timeline}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({**{k: result[k] for k in ['frame', 'observations', 'first_active_outer',
        'last_active_outer', 'active_observations']}, 'first_active': activated[:4],
        'last_active': activated[-4:]}))


if __name__ == '__main__':
    main()
