import os
import shutil
import random

random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
VALID_IMG = os.path.join(BASE, "valid", "images")
VALID_LBL = os.path.join(BASE, "valid", "labels")
TEST_IMG  = os.path.join(BASE, "test",  "images")
TEST_LBL  = os.path.join(BASE, "test",  "labels")

# ── Create test folders ────────────────────────────────────────────────────────
os.makedirs(TEST_IMG, exist_ok=True)
os.makedirs(TEST_LBL, exist_ok=True)

# ── Sample 20% of valid images ─────────────────────────────────────────────────
images = [f for f in os.listdir(VALID_IMG) if f.endswith(('.jpg', '.png'))]
test_sample = random.sample(images, int(len(images) * 0.2))

# ── Move them to test ──────────────────────────────────────────────────────────
for img_name in test_sample:
    stem = os.path.splitext(img_name)[0]

    shutil.move(os.path.join(VALID_IMG, img_name),
                os.path.join(TEST_IMG,  img_name))

    lbl_name = stem + ".txt"
    if os.path.exists(os.path.join(VALID_LBL, lbl_name)):
        shutil.move(os.path.join(VALID_LBL, lbl_name),
                    os.path.join(TEST_LBL,  lbl_name))

print(f"Total valid images   : {len(images)}")
print(f"Moved to test        : {len(test_sample)}")
print(f"Remaining in valid   : {len(images) - len(test_sample)}")