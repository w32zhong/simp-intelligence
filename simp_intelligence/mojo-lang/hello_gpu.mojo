from gpu import (
    WARP_SIZE,
    block_idx,
    thread_idx,
    warp_id,
)

from gpu.host import DeviceContext
from sys import has_accelerator


fn print_threads():
    """Print thread IDs."""

    print("Block index: [",
        block_idx.x,
        "]\tThread index: [",
        thread_idx.x,
        "]\tWarp ID: [",
        warp_id(),
        "]"
    )


def main():
    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()
        print("Found GPU:", ctx.name(), "\t WARP_SIZE:", WARP_SIZE)
        ctx.enqueue_function_checked[print_threads, print_threads](
            grid_dim=2, block_dim=128
        )
        ctx.synchronize()
        print("Program finished")
