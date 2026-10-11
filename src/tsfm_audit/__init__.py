"""tsfm_audit: model-agnostic covariate-reliance probes for TSFMs."""
from .probes import permute_time, swap_across_series, truncate_context
from .metrics import mase, wql, skill_score
from .planted import PlantedConfig, generate_planted_panel
from .ceiling import ceiling_ridge
from .harness import ProbeResult, run_probe, run_probe_windows

__all__ = [
    "swap_across_series",
    "permute_time",
    "truncate_context",
    "mase",
    "wql",
    "skill_score",
    "PlantedConfig",
    "generate_planted_panel",
    "ceiling_ridge",
    "ProbeResult",
    "run_probe",
    "run_probe_windows",
]

__version__ = "0.1.0"
