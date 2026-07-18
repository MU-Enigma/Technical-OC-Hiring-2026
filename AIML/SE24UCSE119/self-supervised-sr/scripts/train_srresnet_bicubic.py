"""Entry point: train SRResNet-lite, ablation arm A (bicubic-only degradation)."""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "src.train", "--config", "configs/srresnet_bicubic.json"],
        check=True,
    )