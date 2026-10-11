"""Where does uni2ts import splitters from? + raw module output shapes."""
import subprocess

print(subprocess.run(["grep", "-rn", "TFTInstanceSplitter\|TestSplitSampler", "/tmp/uv-venv/lib/python3.13/site-packages/uni2ts/model/moirai2/__init__.py"],
                     capture_output=True, text=True).stdout[:800])
print(subprocess.run(["grep", "-rn", "class TFTInstanceSplitter", "/tmp/uv-venv/lib/python3.13/site-packages/gluonts/"],
                     capture_output=True, text=True).stdout[:300])
