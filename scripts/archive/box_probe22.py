"""Find predict_to_numpy.register implementations."""
import subprocess

print(subprocess.run(["grep", "-rn", "predict_to_numpy.register", "/tmp/uv-venv/lib/python3.13/site-packages/"],
                     capture_output=True, text=True).stdout[:1500])
