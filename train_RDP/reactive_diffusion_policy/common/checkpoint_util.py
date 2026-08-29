from pathlib import Path
from typing import Optional, Dict
import os
import re


class PeriodicCheckpointManager:
    """Retain chronological checkpoints independently from metric top-k."""

    def __init__(self, save_dir, keep=0, format_str="epoch={epoch:04d}.ckpt"):
        self.save_dir = Path(save_dir)
        self.keep = int(keep)
        self.format_str = str(format_str)
        if self.keep < 0:
            raise ValueError("periodic checkpoint keep must be non-negative")

    def get_ckpt_path(self, epoch):
        if self.keep == 0:
            return None
        return str(self.save_dir / self.format_str.format(epoch=int(epoch)))

    def prune(self, current_path):
        """Remove the oldest completed periodic files beyond the retention cap."""
        if self.keep == 0 or not self.save_dir.is_dir():
            return
        current = Path(current_path)
        candidates = []
        for path in self.save_dir.glob("*.ckpt"):
            if path == current:
                continue
            match = re.search(r"epoch[=_-]?(\d+)", path.name)
            if match:
                candidates.append((int(match.group(1)), path))
        candidates.sort(key=lambda item: item[0])
        # Count the explicitly supplied current checkpoint even if a caller
        # invokes pruning immediately after an atomic/asynchronous writer.
        excess = max(0, len(candidates) + 1 - self.keep)
        for _, path in candidates[:excess]:
            path.unlink(missing_ok=True)

class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()
    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))
        
        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path
        
        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path
