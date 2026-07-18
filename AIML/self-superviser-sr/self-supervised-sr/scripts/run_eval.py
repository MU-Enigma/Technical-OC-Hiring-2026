"""Entry point: run the full evaluation (quantitative table + real-photo qualitative grid)."""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run(
        [
            sys.executable, "-m", "src.evaluate",
            "--data_dir", "data/images",
            "--real_dir", "data/real_lowres",
            "--checkpoints_root", "outputs",
            "--scale", "4",
            "--out_dir", "outputs/eval",
        ],
        check=True,
    )