"""Optional workaround for native GPU shutdown crashes on affected hosts."""
import os
import sys


def hard_exit_after_success(device):
    torch = sys.modules.get('torch')
    if torch is not None and device.startswith('cuda') and torch.cuda.is_initialized():
        torch.cuda.synchronize(device=device)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
