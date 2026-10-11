"""Check TimesFM 2.5 class API: constructor + covariate method."""
import inspect

from timesfm import TimesFM_2p5_200M_torch

print(inspect.signature(TimesFM_2p5_200M_torch.__init__))
print("has forecast_with_covariates:", hasattr(TimesFM_2p5_200M_torch, "forecast_with_covariates"))
print("has forecast:", hasattr(TimesFM_2p5_200M_torch, "forecast"))
print([m for m in dir(TimesFM_2p5_200M_torch) if "cov" in m.lower() or "forecast" in m.lower()])
