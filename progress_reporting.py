"""Opt-in worker progress and GUI accounting, independent of Qt and analysis."""
import codecs
import json
import math
import sys
import time


PREFIX = 'ORION_PROGRESS '
STAGE_WEIGHTS = {'analysis': 45, 'outputs': 50, 'validation': 5}
STAGE_LABELS = {'analysis': 'Generating masks', 'outputs': 'Making outputs',
                'validation': 'Validating results'}


def add_progress_argument(parser):
    parser.add_argument('--progress-json', action='store_true',
                        help='Emit structured, completed-work progress records on stdout')


class ProgressReporter:
    def __init__(self, stage, total, enabled=False, *, stream=None, clock=None):
        if stage not in STAGE_WEIGHTS or not math.isfinite(total) or total <= 0:
            raise ValueError('Progress needs a known stage and a positive total.')
        self.stage, self.total, self.enabled = stage, total, enabled
        self.completed = 0
        self.stream = stream if stream is not None else sys.stdout
        self.clock = clock if clock is not None else time.monotonic
        self.last_emit = -math.inf

    def update(self, completed, detail, *, force=False):
        if not math.isfinite(completed) or not self.completed <= completed <= self.total:
            raise ValueError('Completed work must increase within the planned total.')
        self.completed = completed
        if not self.enabled:
            return
        now = self.clock()
        if force or completed == self.total or now - self.last_emit >= .1:
            record = dict(stage=self.stage, completed=completed, total=self.total, detail=detail)
            print(PREFIX + json.dumps(record, ensure_ascii=True, allow_nan=False),
                  file=self.stream, flush=True)
            self.last_emit = now

    def advance(self, detail, *, force=False):
        self.update(self.completed + 1, detail, force=force)


def advance(progress, detail, *, force=False):
    """Optional function argument preserves direct callers of rendering helpers."""
    if progress is not None:
        progress.advance(detail, force=force)


def parse_progress(line):
    if not line.startswith(PREFIX):
        return None
    try:
        record = json.loads(line[len(PREFIX):])
        if not isinstance(record, dict) or record.get('stage') not in STAGE_WEIGHTS:
            return None
        done, total = record.get('completed'), record.get('total')
        if (type(done) not in (int, float) or type(total) not in (int, float)
                or not math.isfinite(done) or not math.isfinite(total)
                or not 0 <= done <= total or total <= 0
                or not isinstance(record.get('detail'), str)):
            return None
        return record
    except (ValueError, TypeError, OverflowError, RecursionError):
        return None


class ProgressStream:
    """Handle arbitrary QProcess chunks, including split UTF-8 and final lines."""
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        self.pending = ''

    def feed(self, chunk=b'', *, final=False):
        self.pending += self.decoder.decode(chunk, final=final)
        parts = self.pending.split('\n')
        self.pending = parts.pop()
        lines = [part + '\n' for part in parts]
        if final and self.pending:
            lines.append(self.pending)
            self.pending = ''
        return [(parse_progress(line), line) for line in lines]


class RunProgress:
    """Fixed stage weights; only successful process exits complete the run."""
    def __init__(self, outputs=True):
        self.weights = {stage: weight for stage, weight in STAGE_WEIGHTS.items()
                        if outputs or stage != 'outputs'}
        self.fractions = dict.fromkeys(self.weights, 0.)
        self.finished = set()
        self.stage = 'analysis'
        self.active = True
        self.succeeded = False

    def begin(self, stage):
        if stage not in self.weights:
            raise ValueError('This stage is not part of the run.')
        self.stage = stage

    def accept(self, record):
        if not self.active or record['stage'] != self.stage:
            return False
        fraction = record['completed'] / record['total']
        self.fractions[self.stage] = max(self.fractions[self.stage], fraction)
        return True

    def complete_stage(self):
        self.fractions[self.stage] = 1.
        self.finished.add(self.stage)

    def finish(self, success):
        self.succeeded = bool(success and self.finished == set(self.weights))
        self.active = False

    @property
    def overall_percent(self):
        if self.succeeded:
            return 100
        percent = sum(self.weights[s] * self.fractions[s] for s in self.weights)
        return min(99, math.floor(100 * percent / sum(self.weights.values()) + 1e-9))

    @property
    def stage_percent(self):
        return min(100, math.floor(self.fractions[self.stage] * 100 + 1e-9))
