import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

BASE       = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMGS = os.path.join(BASE, "train", "images")
TRAIN_LBLS = os.path.join(BASE, "train", "labels")

CLASS_NAMES = ['car', 'threewheel', 'bus', 'truck', 'motorbike', 'van']
COLORS      = ['#e74c3c','#f39c12','#2ecc71','#3498db','#9b59b6','#1abc9c']

# ── 1. Class distribution ──────────────────────────────────────────────────────
counts = {name: 0 for name in CLASS_NAMES}
for lbl in os.listdir(TRAIN_LBLS):
    if not lbl.endswith('.txt'):
        continue
    with open(os.path.join(TRAIN_LBLS, lbl)) as f:
        for line in f:
            cls_id = int(line.strip().split()[0])
            counts[CLASS_NAMES[cls_id]] += 1

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(counts.keys(), counts.values(), color=COLORS, edgecolor='white')
ax.bar_label(bars)
ax.set_title('Training Set — Instance Count per Class', fontweight='bold')
ax.set_xlabel('Class'); ax.set_ylabel('Instances')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'class_distribution.png'), dpi=150)
plt.show()
print("Class counts:", counts)

# ── 2. Sample images with ground-truth boxes ───────────────────────────────────
def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    x1 = (cx - w/2) * img_w
    y1 = (cy - h/2) * img_h
    x2 = (cx + w/2) * img_w
    y2 = (cy + h/2) * img_h
    return x1, y1, x2, y2

samples = random.sample(os.listdir(TRAIN_IMGS), 6)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for ax, img_name in zip(axes.flatten(), samples):
    img_path = os.path.join(TRAIN_IMGS, img_name)
    lbl_path = os.path.join(TRAIN_LBLS, os.path.splitext(img_name)[0] + '.txt')
    img      = np.array(Image.open(img_path).convert('RGB'))
    H, W     = img.shape[:2]
    ax.imshow(img)
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                p  = line.strip().split()
                cid = int(p[0])
                x1,y1,x2,y2 = yolo_to_xyxy(*map(float,p[1:]),W,H)
                rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,
                                         linewidth=2,
                                         edgecolor=COLORS[cid],
                                         facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1-5, CLASS_NAMES[cid], color='white',
                        fontsize=9, fontweight='bold',
                        bbox=dict(facecolor=COLORS[cid], alpha=0.85, pad=1))
    ax.axis('off')

plt.suptitle('Sample Training Images — Ground-Truth Annotations', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'sample_annotations.png'), dpi=150)
plt.show()
print("EDA done — 2 plots saved.")