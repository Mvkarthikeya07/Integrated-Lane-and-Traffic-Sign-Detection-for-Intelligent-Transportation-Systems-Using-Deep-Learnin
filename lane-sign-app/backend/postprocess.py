# postprocess.py
import numpy as np
import cv2

def mask_to_lane_polylines(mask, min_pixels=200, poly_deg=2):
    """
    Input:
      mask: binary mask (H,W) with 0 or 255 values
    Output:
      list of polylines; each polyline is Nx2 numpy array of (x,y) points in image coords
    Steps:
      - find connected components of mask
      - for each large component, sample points and fit polynomial x = f(y) (since lanes run vertically)
      - return polylines to draw
    """
    mask_bin = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    polylines = []
    h, w = mask.shape
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        if area < min_pixels:
            continue
        comp_mask = (labels == lab).astype(np.uint8)
        ys, xs = np.where(comp_mask)
        if len(xs) < 50:
            continue
        # Fit polynomial x = a*y^2 + b*y + c
        coeffs = np.polyfit(ys, xs, poly_deg)
        y_vals = np.linspace(0, h-1, num=50)
        x_vals = np.polyval(coeffs, y_vals)
        # Clip to image
        pts = np.stack([x_vals, y_vals], axis=1)
        pts[:,0] = np.clip(pts[:,0], 0, w-1)
        polylines.append(pts.astype(np.int32))
    return polylines

def draw_lanes_on_image(image_bgr, polylines, color=(0,255,0), thickness=6):
    out = image_bgr.copy()
    for pts in polylines:
        pts_int = pts.reshape(-1,2)
        for i in range(len(pts_int)-1):
            x1, y1 = int(pts_int[i][0]), int(pts_int[i][1])
            x2, y2 = int(pts_int[i+1][0]), int(pts_int[i+1][1])
            cv2.line(out, (x1,y1), (x2,y2), color, thickness)
    return out
