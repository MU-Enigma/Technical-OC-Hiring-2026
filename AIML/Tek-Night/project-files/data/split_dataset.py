from pathlib import Path
import random
import shutil

# ----------------------------
# Paths
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "dataset" / "raw_hr"
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
VAL_DIR = PROJECT_ROOT / "dataset" / "val"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"

# ----------------------------
# Split sizes
# ----------------------------

TRAIN_SIZE = 700
VAL_SIZE = 50
TEST_SIZE = 50

SEED = 42


def copy_images(image_list, destination):
    """Copy images into a destination folder."""

    destination.mkdir(parents=True, exist_ok=True)

    for image in image_list:
        shutil.copy2(image, destination / image.name)


def main():

    image_files = sorted(RAW_DIR.glob("*.png"))

    expected = TRAIN_SIZE + VAL_SIZE + TEST_SIZE

    if len(image_files) != expected:
        raise ValueError(
            f"Expected {expected} images, found {len(image_files)}."
        )

    random.seed(SEED)
    random.shuffle(image_files)

    train_images = image_files[:TRAIN_SIZE]
    val_images = image_files[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    test_images = image_files[TRAIN_SIZE + VAL_SIZE:]

    copy_images(train_images, TRAIN_DIR)
    copy_images(val_images, VAL_DIR)
    copy_images(test_images, TEST_DIR)

    print("\nDataset split complete!\n")
    print(f"Train      : {len(train_images)}")
    print(f"Validation : {len(val_images)}")
    print(f"Test       : {len(test_images)}")


if __name__ == "__main__":
    main()