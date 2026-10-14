"""Completed-work progress, stream framing, and real worker integration."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from progress_reporting import PREFIX, ProgressReporter, ProgressStream, RunProgress, parse_progress
from test_gui import fixture


def record(stage='analysis', completed=1, total=2, detail='Mask Δ'):
    return dict(stage=stage, completed=completed, total=total, detail=detail)


class ProgressTests(unittest.TestCase):
    def test_reporter_throttles_updates_but_flushes_boundaries(self):
        out = io.StringIO()
        now = [0.]
        reporter = ProgressReporter('analysis', 4, True, stream=out, clock=lambda: now[0])
        reporter.update(0, 'Start', force=True)
        reporter.advance('One')
        self.assertEqual(len(out.getvalue().splitlines()), 1)
        now[0] = .11
        reporter.advance('Two')
        reporter.advance('Boundary', force=True)
        reporter.advance('Done')
        events = [parse_progress(line) for line in out.getvalue().splitlines()]
        self.assertEqual([e['completed'] for e in events], [0, 2, 3, 4])
        silent = io.StringIO()
        ProgressReporter('analysis', 1, stream=silent).advance('Done')
        self.assertEqual(silent.getvalue(), '')

    def test_arbitrary_chunks_utf8_warnings_and_final_line(self):
        event = PREFIX + json.dumps(record(), ensure_ascii=False)
        source = ('Warning: cliché\r\n' + event + '\n' + PREFIX + 'bad json\n' + 'Final Δ').encode()
        stream = ProgressStream()
        parsed = []
        for byte in source:
            parsed.extend(stream.feed(bytes([byte])))
        parsed.extend(stream.feed(final=True))
        self.assertEqual(''.join(line for _, line in parsed), source.decode())
        self.assertEqual([r for r, _ in parsed if r is not None], [record()])
        self.assertIsNone(parsed[2][0])
        self.assertEqual(parsed[-1], (None, 'Final Δ'))

    def test_invalid_records_cannot_change_progress(self):
        invalid = [dict(completed=True), dict(completed=-1), dict(completed=3),
                   dict(total=0), dict(total=float('inf')), dict(completed=float('nan')),
                   dict(detail=None), dict(stage='unknown')]
        for changes in invalid:
            with self.subTest(changes=changes):
                self.assertIsNone(parse_progress(PREFIX + json.dumps({**record(), **changes})))
        for value in [[], None, 'text', {'stage': []}, {'stage': {}}]:
            self.assertIsNone(parse_progress(PREFIX + json.dumps(value)))

    def test_weights_transitions_monotonicity_and_verified_completion(self):
        progress = RunProgress()
        progress.accept(record())
        self.assertEqual((progress.overall_percent, progress.stage_percent), (22, 50))
        progress.accept(record(completed=0))
        self.assertEqual(progress.stage_percent, 50)
        progress.complete_stage()
        progress.begin('outputs')
        self.assertEqual((progress.overall_percent, progress.stage_percent), (45, 0))
        self.assertFalse(progress.accept(record()))
        progress.accept(record(stage='outputs'))
        self.assertEqual(progress.overall_percent, 70)
        progress.complete_stage()
        progress.begin('validation')
        self.assertEqual(progress.overall_percent, 95)
        progress.accept(record(stage='validation', completed=2))
        self.assertEqual(progress.overall_percent, 99)
        progress.complete_stage()
        self.assertEqual(progress.overall_percent, 99)
        progress.finish(True)
        self.assertEqual(progress.overall_percent, 100)

    def test_skipped_outputs_and_failure_do_not_fake_completion(self):
        progress = RunProgress(outputs=False)
        progress.accept(record())
        self.assertEqual(progress.overall_percent, 45)
        progress.complete_stage()
        progress.begin('validation')
        self.assertEqual(progress.overall_percent, 90)
        progress.accept(record(stage='validation'))
        progress.finish(False)
        self.assertEqual((progress.overall_percent, progress.stage_percent), (95, 50))
        self.assertFalse(progress.accept(record(stage='validation', completed=2)))
        self.assertEqual(RunProgress().overall_percent, 0)


class WorkerProgressTests(unittest.TestCase):
    def run_worker(self, script, *args, progress=True):
        command = [sys.executable, '-X', 'utf8', str(ROOT / script), *map(str, args)]
        if progress:
            command.append('--progress-json')
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        events = [r for line in result.stdout.splitlines() if (r := parse_progress(line))]
        if progress:
            self.assertGreaterEqual(len(events), 2)
            self.assertEqual(events[0]['completed'], 0)
            self.assertEqual(events[-1]['completed'], events[-1]['total'])
            self.assertTrue(all(e['total'] == events[0]['total'] for e in events))
            fractions = [e['completed'] / e['total'] for e in events]
            self.assertEqual(fractions, sorted(fractions))
        else:
            self.assertEqual(events, [])
        return events

    def test_real_workers_minimal_detailed_and_empty_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, path, p = fixture(root)
            blank = inputs / 'Control_02.TIF'
            tifffile.imwrite(blank, np.zeros((3,128,128), dtype=np.uint16), imagej=True,
                             resolution=(5,5), metadata={'axes':'CYX','unit':'micron'})
            config = root / 'parameters.json'
            config.write_text(json.dumps(p), encoding='utf-8')
            out = root / 'out'
            raw = path.read_bytes(), blank.read_bytes()
            analysis = self.run_worker('analyze_foci.py', '--input-dir', inputs,
                                       '--output-dir', out, '--parameters', config)
            self.assertEqual(analysis[-1]['total'], 7)
            masks = {f.name: f.read_bytes() for f in (out / 'Masks').iterdir()}
            tables = {f.name: f.read_bytes() for f in (out / 'Results').iterdir()}
            for mode in ('minimal', 'detailed'):
                events = self.run_worker('make_outputs.py', '--output-dir', out, '--qc-output', mode)
                self.assertEqual(events[-1]['stage'], 'outputs')
                self.assertEqual(events[-1]['detail'], 'Review reports saved')
                # Two input images, two foci channels, one group, one usable nucleus.
                self.assertEqual(events[-1]['total'], 39 if mode == 'minimal' else 54)
            validation = self.run_worker('validate_results.py', '--output-dir', out)
            self.assertEqual(validation[-1]['total'], 4)
            self.assertEqual({f.name: f.read_bytes() for f in (out / 'Masks').iterdir()}, masks)
            self.assertEqual({f.name: f.read_bytes() for f in (out / 'Results').iterdir()}, tables)
            self.assertEqual((path.read_bytes(), blank.read_bytes()), raw)
            # Turning telemetry off preserves the CLI and the scientific results.
            plain = root / 'plain'
            self.run_worker('analyze_foci.py', '--input-dir', inputs, '--output-dir', plain,
                            '--parameters', config, progress=False)
            for filename in tables:
                pd.testing.assert_frame_equal(pd.read_csv(out/'Results'/filename),
                                               pd.read_csv(plain/'Results'/filename))
            for filename in masks:
                np.testing.assert_array_equal(tifffile.imread(out/'Masks'/filename),
                                               tifffile.imread(plain/'Masks'/filename))


if __name__ == '__main__':
    unittest.main()
