"""Extra adapters stay import-light (no heavy third-party imports at module load)."""
import sys


def test_extra_import_is_light():
    import tsfm_audit.adapters_extra as ax

    for mod in ("torch", "tirex", "t0", "toto", "timesfm", "timesfm3", "uni2ts", "gluonts", "tabpfn_time_series"):
        assert mod not in sys.modules, f"{mod} imported at module load — must stay lazy"
    assert hasattr(ax, "make_t0_predict") and hasattr(ax, "load_timesfm3")
