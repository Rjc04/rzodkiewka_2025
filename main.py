import pydicom
import numpy as np
import matplotlib.pyplot as plt
from smoothn import smoothn   # paste that code into smoothn.py or same notebook

# 1) Load your dicom
ds = pydicom.dcmread("C:\Users\ritaj\Downloads\OCTImage_2025_07_18_08-57-12.dcm")
img = ds.pixel_array.astype(float)

# 2) Smooth (auto s, robust)
img_s, s = smoothn(img, robust=True, return_s=True)

print("Chosen s:", float(s))

# 3) Quick visual check
fig, ax = plt.subplots(1, 3, figsize=(12,4))
ax[0].imshow(img, cmap='gray');     ax[0].set_title('Original')
ax[1].imshow(img_s, cmap='gray');   ax[1].set_title('Smoothed')
ax[2].imshow(img - img_s, cmap='gray'); ax[2].set_title('Residual')
for a in ax: a.axis('off')
plt.tight_layout(); plt.show()
