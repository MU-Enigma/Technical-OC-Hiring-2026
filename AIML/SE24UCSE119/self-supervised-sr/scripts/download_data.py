# Oxford-IIIT Pet is used as the HR source pool: real photos at variable,
# genuinely high native resolution. Images are only ever resized DOWN (to
# cap compute cost); any image whose native resolution is too small to be
# a legitimate HR source is skipped, so no "HR" detail is manufactured by
# upsampling a low-res source.
import os

from torchvision.datasets import OxfordIIITPet
from PIL import Image

MIN_NATIVE = 200   # skip images smaller than this - never treat upsampled pixels as "HR"
MAX_DIM = 400      # cap the larger side - keeps patch extraction / compute light

os.makedirs("data/images", exist_ok=True)

print("downloading Oxford-IIIT Pet (used only as our HR source pool - real photos, native res)...")
pet_ds = OxfordIIITPet(root="data/_raw", split="trainval", download=True)

counter = 0
skipped_too_small = 0
for i in range(len(pet_ds)):
    img, _ = pet_ds[i]
    w, h = img.size
    if min(w, h) < MIN_NATIVE:
        skipped_too_small += 1
        continue
    if max(w, h) > MAX_DIM:
        ratio = MAX_DIM / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.BICUBIC)  # DOWN only
    img.convert("RGB").save(f"data/images/img_{counter:04d}.jpg", format="JPEG", quality=95)
    counter += 1
    if counter >= 800:
        break

print(f"done. saved {counter} images (skipped {skipped_too_small} too-small natives). "
      f"No image was ever upsampled above its native resolution.")