"""Entry point: train the SRCNN baseline model."""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, "-m", "src.train", "--config", "configs/srcnn.json"],
        check=True,
    )