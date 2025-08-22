"""
radish_full.py
--------------
1) Load ONE DICOM
2) Smooth with smoothn (auto or fixed s)
3) Segment radish (Otsu + largest component)
4) Metrics
5) Contour + full-width top line (spline + linear edge extrapolation)
6) Plots (+ optional PNG saves)

Reqs: pydicom, scikit-image, scipy, matplotlib, imageio
Install: pip install pydicom scikit-image scipy matplotlib imageio
"""

from pathlib import Path
import numpy as np
import pydicom
import matplotlib.pyplot as plt

from smoothn import smoothn
from skimage.filters import threshold_otsu
from skimage.measure import label, find_contours
from skimage.morphology import remove_small_objects
from scipy.interpolate import UnivariateSpline
from imageio import imwrite

# -------------------- USER SETTINGS --------------------
FILE      = r"C:\Users\ritaj\Downloads\OCTImage_2025_07_18_08-57-12.dcm" # path to your dicom
AUTO_S    = True         # True = auto GCV; False = use FIXED_S
FIXED_S   = 1.04         # used only if AUTO_S is False
ROBUST    = True         # robust iteration (slower)
MIN_OBJ   = 5000         # min pixel count for kept object
SAVE_PNG  = True         # save output PNGs
EDGE_NPTS = 30           # how many columns at each end to estimate slope
SPL_SMOOTH_FACTOR = 0.5  # spline smoothing factor multiplier (0 = interpolate)
# -------------------------------------------------------

# ---------- 1) Load ----------
dicom_path = Path(FILE)
ds  = pydicom.dcmread(dicom_path)
img = ds.pixel_array.astype(float)
print("Image:", img.shape, img.dtype)

# ---------- 2) Smooth ----------
if AUTO_S:
    img_s, s = smoothn(img, robust=ROBUST, return_s=True)
    s_used = float(np.asarray(s))
else:
    img_s  = smoothn(img, s=FIXED_S, robust=ROBUST)
    s_used = FIXED_S
print(f"Chosen s: {s_used:.4g}")

# ---------- 3) Residual stats ----------
res = img - img_s
res_ratio = float(np.std(res) / np.std(img))
res_max   = float(np.abs(res).max())
print("Residual std / original std:", res_ratio)
print("Max abs residual:", res_max)

# ---------- 4) Segment (Otsu + largest component) ----------
thr  = threshold_otsu(img_s)
mask = img_s > thr

lab = label(mask)
if lab.max() > 0:
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    biggest = sizes.argmax()
    mask = (lab == biggest)
    mask = remove_small_objects(mask, MIN_OBJ)

mean_intensity = float(img_s[mask].mean())
area_px        = int(mask.sum())
print(f"Mean intensity (inside radish): {mean_intensity:.2f}")
print(f"Area (pixels): {area_px}")

# ---------- 5) Contour & full-width top line ----------
contours = find_contours(mask.astype(float), level=0.5)
contour = max(contours, key=len) if contours else np.empty((0, 2))

rows, cols = np.where(mask)
top_y = {}
for r, c in zip(rows, cols):
    if c not in top_y or r < top_y[c]:
        top_y[c] = r

xs_known = np.array(sorted(top_y.keys()))
ys_known = np.array([top_y[x] for x in xs_known])

W = img_s.shape[1]
xs_full = np.arange(W)

# spline over known region
if xs_known.size > 3:
    spl = UnivariateSpline(xs_known, ys_known,
                           s=len(xs_known)*SPL_SMOOTH_FACTOR)
    ys_spline = spl(xs_full)
else:
    spl = None
    ys_spline = np.interp(xs_full, xs_known, ys_known)

# Linear edge extrapolation using slope from first/last EDGE_NPTS cols
def edge_fit(x_arr, y_arr, side='left'):
    n = min(EDGE_NPTS, x_arr.size)
    if side == 'left':
        xseg = x_arr[:n]; yseg = y_arr[:n]
    else:
        xseg = x_arr[-n:]; yseg = y_arr[-n:]
    m, b = np.polyfit(xseg, yseg, 1)  # slope, intercept
    return m, b

# slopes on the known region (use raw points for stability)
m_left, b_left  = edge_fit(xs_known, ys_known, 'left')
m_right, b_right = edge_fit(xs_known, ys_known, 'right')

ys_full = ys_spline.copy()

left_end  = xs_known.min()
right_end = xs_known.max()

# left extrapolation
xsL = np.arange(0, left_end)
ys_full[xsL] = m_left * xsL + b_left

# right extrapolation
xsR = np.arange(right_end+1, W)
ys_full[xsR] = m_right * xsR + b_right

# clamp to image bounds
ys_full = np.clip(ys_full, 0, img_s.shape[0]-1)

# ---------- 6) Plots ----------
# Main 3-panel
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(img,   cmap='gray'); ax[0].set_title("Original");  ax[0].axis('off')
ax[1].imshow(img_s, cmap='gray'); ax[1].set_title("Smoothed");  ax[1].axis('off')
ax[2].imshow(res,   cmap='gray',
             vmin=np.percentile(res, 1),
             vmax=np.percentile(res, 99))
ax[2].set_title("Residual (1–99% clip)"); ax[2].axis('off')
plt.tight_layout(); plt.show()

# Residual exaggeration
fig2 = plt.figure(figsize=(4,4))
plt.imshow(res*5, cmap='gray'); plt.title('Residual x5'); plt.axis('off')
plt.tight_layout(); plt.show()

# Contour & top line overlay
fig3, ax3 = plt.subplots(figsize=(6,6))
ax3.imshow(img_s, cmap='gray')
if contour.size:
    ax3.plot(contour[:,1], contour[:,0], '-r', lw=1, label='Contour')
ax3.plot(xs_known, ys_known, '.y', ms=2, label='Top raw (mask)')
ax3.plot(xs_full, ys_full, '-c', lw=2, label='Top smooth (edge-to-edge)')
ax3.legend(loc='lower right')
ax3.set_title("Radish outline & full-width top line (improved edges)")
ax3.axis('off')
plt.tight_layout(); plt.show()

# ---------- 7) Optional PNG saves ----------
if SAVE_PNG:
    png1 = dicom_path.with_suffix('.preview.png')
    png2 = dicom_path.with_suffix('.residualx5.png')
    png3 = dicom_path.with_suffix('.contour.png')
    fig.savefig(png1, dpi=200)
    fig2.savefig(png2, dpi=200)
    fig3.savefig(png3, dpi=200)
    print("Figures saved to:", png1, png2, png3)

# Optional cropped image save
ys_idx, xs_idx = np.where(mask)
if ys_idx.size:
    ymin, ymax = ys_idx.min(), ys_idx.max()
    xmin, xmax = xs_idx.min(), xs_idx.max()
    cropped = img_s[ymin:ymax+1, xmin:xmax+1]
    if SAVE_PNG:
        crop_png = dicom_path.with_suffix('.cropped.png')
        imwrite(crop_png, (cropped / cropped.max() * 255).astype(np.uint8))
        print("Cropped image saved to:", crop_png)
