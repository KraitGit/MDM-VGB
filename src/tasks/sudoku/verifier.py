from . import task
from tasks.common import ExactStateVerifier


def load(**kwargs):
    del kwargs
    return ExactStateVerifier(task)
