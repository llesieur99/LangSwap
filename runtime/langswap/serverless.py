import subprocess

import runpod

# NOTE: do NOT import the heavy pipeline (langswap.api → torch/vLLM/transformers/
# sherpa/omnivoice) at module top.  RunPod runs this file and only starts pulling
# jobs once `runpod.serverless.start` is reached; importing the whole pipeline
# first delays handler registration by the full cold-import time.  On the large
# self-contained image that import is slow enough to exceed RunPod's worker-init
# window, so the worker restart-loops and never becomes ready — jobs sit in_queue
# with the worker shown as "running".  Import lazily inside the handler instead:
# the worker registers in seconds, and the heavy import + model load happen on the
# first job (models stay warm across jobs via the process-global pool).

# Set GPU compute mode at startup. Best-effort and time-bounded: on a shared or
# permission-restricted serverless GPU, `nvidia-smi -c 0` can hang or be denied,
# and a hang here would block the worker from ever registering its handler. Never
# let it block startup.
# Catch *everything* (a non-zero exit, a timeout/hang, a missing binary, or an
# exec/permission error like OSError "Exec format error") — setting the compute
# mode is an optimization and must never crash or stall worker startup.
try:
    subprocess.run(['nvidia-smi', '-c', '0'], check=True, timeout=30)
    print("Successfully set GPU compute mode to DEFAULT")
except Exception as e:
    print(f"Warning: could not set GPU compute mode ({type(e).__name__}: {e}); continuing")


def handler(job):
    """RunPod serverless handler that uses main.py functionality"""
    # Lazy import — see the note above on why this is not a module-level import.
    from langswap.api import process_translation, process_update_translation

    input = job['input']
    show_progress = input.get("show_progress", False)

    # Create a progress callback function specific to runpod
    def progress_callback(message):
        if show_progress:
            runpod.serverless.progress_update(job, message)

    # Call the shared process_translation function
    if "update_request" in input:
        return process_update_translation(input, progress_callback)
    return process_translation(input, progress_callback)


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
