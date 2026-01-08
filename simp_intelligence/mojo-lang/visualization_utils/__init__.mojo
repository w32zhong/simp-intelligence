from .log_layout_tensor import LoggedTensor, example_logged_tensor
from .mock_gpu import block_idx, thread_idx, barrier

from pathlib import Path
import os


fn clear_log_files() raises:
    var p = Path(".")
    var items = p.listdir()
    for i in range(len(items)):
        var item = items[i]
        if String(item).endswith(".log"):
            print('Removing existing log:', item)
            os.remove(item)
