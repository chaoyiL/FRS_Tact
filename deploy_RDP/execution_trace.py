"""Persist execution-time and plan-switch evidence for offline reproduction."""
from datetime import datetime
import json
import os
from pathlib import Path
import time


class ExecutionTrace:
    def __init__(self, directory, metadata):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f'rdp_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}_{time.time_ns()}.jsonl'
        self._file = self.path.open('x', encoding='utf-8', buffering=1)
        self.write({'type': 'session', **metadata})

    def write(self, record):
        self._file.write(json.dumps(record, allow_nan=False, separators=(',', ':')) + '\n')

    def close(self):
        self._file.close()
