from . import task
from tasks.common import ExactStateVerifier


def load(harness=None, device=None):
    del harness, device
    return ExactStateVerifier(task)
