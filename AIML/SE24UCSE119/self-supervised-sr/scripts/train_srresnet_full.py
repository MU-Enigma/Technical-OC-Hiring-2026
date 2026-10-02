"""Entry point: train SRResNet-lite, final model (full randomized degradation)."""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "src.train", "--config", "configs/srresnet_full.json"],
        check=True,
    )