"""HPC-based one-class detector for neural-network backdoor inferences."""
__version__ = "0.1.0"

from .detector import DetectorConfig, HPCBackdoorDetector  # noqa: E402,F401
from .schema import FEATURES, load_traces  # noqa: E402,F401
