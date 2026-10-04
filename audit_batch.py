import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    rows, hashes, configs = [], [], []
    for i in range(1, 10):
        sid = f'A{i:02d}'
        path = args.directory / f'{sid}.json'
        record = json.loads(path.read_text(encoding='utf-8'))
        assert record['epochs'] == 1000
        history = record['history']
        assert [h['epoch'] for h in history] == list(range(1, 1001))
        assert all(np.isfinite(h['train_loss']) for h in history)
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as pred:
            truth, predicted = pred['y_true'], pred['y_pred']
            assert truth.shape == predicted.shape == (288,)
            assert np.array_equal(np.bincount(truth, minlength=4), [72]*4)
            assert np.isin(predicted, range(4)).all()
            accuracy = float(np.mean(truth == predicted))
        assert accuracy == record['final_test_acc'] == history[-1]['test_acc']
        configs.append({k: record[k] for k in (
            'strict', 'paper_pooling', 'bn_first', 'norm_then_activation',
            'seed', 'epochs', 'device', 'training_protocol')})
        rows.append((sid, accuracy * 100, record['seconds']))
        for suffix in ('.json', '.npz', '.pt'):
            artifact = path.with_suffix(suffix)
            assert artifact.stat().st_size > 0
            hashes.append((artifact.name, hashlib.sha256(artifact.read_bytes()).hexdigest()))
    assert all(c == configs[0] for c in configs)
    values = np.array([row[1] for row in rows])
    summary = json.loads((args.directory / 'summary.json').read_text())
    assert np.isclose(values.mean()/100, summary['mean_final_acc'])
    assert np.isclose(values.std(ddof=1)/100, summary['std_final_acc'])
    lines = ['# Nine-subject pooled variant audit', '',
             'Training code: v1.0.3 (bb3d254). Audit release: v1.0.4.', '',
             'All nine saved prediction files reproduce their final-epoch accuracies.',
             'Configuration matches across subjects; 1000 finite training-loss records each.',
             'Checkpoint presence and hashes verified; checkpoint inference not rerun.', '',
             '| Subject | Accuracy (%) | Recorded runtime (s) |',
             '| --- | ---: | ---: |']
    lines += [f'| {sid} | {acc:.2f} | {sec:.2f} |' for sid, acc, sec in rows]
    lines += ['', f'Mean +/- sample SD: {values.mean():.2f}% +/- {values.std(ddof=1):.2f}%.',
              f'Difference from paper mean 81.17%: {values.mean()-81.17:+.2f} percentage points.',
              f'Total recorded runtime: {sum(r[2] for r in rows):.2f} s.', '',
              'Runtime includes training and per-epoch test evaluation; it is not pure training time.',
              'Pooling and dropout were changed together. Their separate effects are unproven.',
              'This exploratory reconstruction is not a verified original-paper implementation.',
              'A01 test performance informed progression to this batch; results are not a pristine',
              'held-out confirmatory evaluation. Do not tune further on these test results.', '',
              '## Configuration', '', '```json', json.dumps(configs[0], indent=2), '```', '',
              '## Local artifact SHA-256', '', '| File | SHA-256 |', '| --- | --- |']
    lines += [f'| {name} | {digest} |' for name, digest in hashes]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'PASS: 9 subjects; {values.mean():.2f}% +/- {values.std(ddof=1):.2f}%')


if __name__ == '__main__':
    main()
