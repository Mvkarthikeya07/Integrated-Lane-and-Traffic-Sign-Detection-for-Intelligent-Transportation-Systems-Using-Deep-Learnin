# backend/app.py
# Flask backend with improved curved-lane visualization and normal-based mirroring (no DL required)
# Requirements: pip install flask flask-cors opencv-python numpy ultralytics

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import time
import os
from ultralytics import YOLO

# ------------------- Sign detector (YOLOv8) -------------------
YOLO_DEVICE = os.environ.get('YOLO_DEVICE', 'cpu')

class SignDetector:
    def __init__(self, model_name='yolov8n.pt', device='cpu', conf_threshold=0.35):
        self.model = YOLO(model_name)
        self.device = device
        self.conf_threshold = conf_threshold
        self.names = self.model.names

    def detect(self, frame):
        results = self.model(frame, device=self.device, verbose=False)
        if len(results) == 0:
            return []
        res = results[0]
        boxes = getattr(res, 'boxes', None)
        detected = []
        if boxes is None:
            return detected
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy() if hasattr(boxes.xyxy[i], 'cpu') else np.array(boxes.xyxy[i])
            conf = float(boxes.conf[i].cpu().numpy()) if hasattr(boxes.conf[i], 'cpu') else float(boxes.conf[i])
            cls = int(boxes.cls[i].cpu().numpy()) if hasattr(boxes.cls[i], 'cpu') else int(boxes.cls[i])
            label = self.names.get(cls, str(cls)) if isinstance(self.names, dict) else self.names[cls]
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            detected.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'label': label, 'conf': conf})
        return detected

# ------------------- Lane detection helpers -------------------
_lane_smooth = {"left_fit": None, "right_fit": None, "alpha": 0.15}

def _smooth_fit(prev, curr, alpha):
    if prev is None:
        return curr
    return alpha * curr + (1 - alpha) * prev

def sliding_window_polyfit(binary_warped, nwindows=9, margin=110, minpix=50):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    if histogram.sum() == 0:
        return None, None
    midpoint = histogram.shape[0] // 2
    leftx_base = int(np.argmax(histogram[:midpoint])) if midpoint>0 else 0
    rightx_base = int(np.argmax(histogram[midpoint:]) + midpoint) if midpoint>0 else 0

    window_height = int(binary_warped.shape[0] // nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        if good_left_inds.size > 0:
            left_lane_inds.append(good_left_inds)
        if good_right_inds.size > 0:
            right_lane_inds.append(good_right_inds)

        if good_left_inds.size > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if good_right_inds.size > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # allow one side missing (we'll mirror)
    left_fit = None
    right_fit = None
    if len(left_lane_inds) > 0:
        left_lane_inds = np.concatenate(left_lane_inds)
        leftx = nonzerox[left_lane_inds]; lefty = nonzeroy[left_lane_inds]
        if leftx.size>0 and lefty.size>0:
            left_fit = np.polyfit(lefty, leftx, 2)
    if len(right_lane_inds) > 0:
        right_lane_inds = np.concatenate(right_lane_inds)
        rightx = nonzerox[right_lane_inds]; righty = nonzeroy[right_lane_inds]
        if rightx.size>0 and righty.size>0:
            right_fit = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit

# ------------------- Core detect_lanes with normal-based mirroring -------------------
def detect_lanes(frame):
    """
    Detect lanes and if only one lane exists, mirror the other using per-y normal offsets.
    """
    global _lane_smooth
    h, w = frame.shape[:2]

    # 1) binary thresholding (gradient + color)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.abs(sobelx)
    maxv = abs_sobelx.max() if abs_sobelx.max() != 0 else 1.0
    scaled = np.uint8(255 * abs_sobelx / maxv)
    sxbinary = np.zeros_like(scaled)
    sxbinary[(scaled >= 20) & (scaled <= 200)] = 1

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 95) & (s_channel <= 255)] = 1

    combined = np.zeros_like(sxbinary)
    combined[(s_binary == 1) | (sxbinary == 1)] = 1
    binary = (combined * 255).astype(np.uint8)

    # 2) perspective transform (tuned)
    src = np.float32([
        [w*0.40, h*0.60],
        [w*0.60, h*0.60],
        [w*0.95, h*0.95],
        [w*0.05, h*0.95]
    ])
    dst = np.float32([
        [w*0.25, 0],
        [w*0.75, 0],
        [w*0.75, h],
        [w*0.25, h]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(binary, M, (w, h), flags=cv2.INTER_LINEAR)

    # 3) sliding window polyfit
    left_fit, right_fit = sliding_window_polyfit(warped, nwindows=9, margin=110, minpix=40)

    # if both missing
    if left_fit is None and right_fit is None:
        return frame

    # 4) estimate lane width in pixels (adaptive)
    # Prefer deriving from detected both lanes if available:
    if left_fit is not None and right_fit is not None:
        # compute bottom x positions
        yb = h - 1
        left_xb = left_fit[0]*yb**2 + left_fit[1]*yb + left_fit[2]
        right_xb = right_fit[0]*yb**2 + right_fit[1]*yb + right_fit[2]
        lane_width_pixels = int(abs(right_xb - left_xb))
        # clamp
        lane_width_pixels = max(200, min(lane_width_pixels, int(w*0.5)))
    else:
        # fallback: use fraction of image width (tunable)
        lane_width_pixels = int(w * 0.30)  # ~30% of width; tune (0.25..0.35)
        lane_width_pixels = max(200, min(lane_width_pixels, int(w*0.5)))

    # 5) if only one side present -> mirror across normal per-y
    ploty = np.linspace(0, h-1, num=h)
    if left_fit is None and right_fit is not None:
        # compute right x curve
        a, b, c = right_fit
        right_x = a*ploty**2 + b*ploty + c
        # derivative dx/dy = 2a*y + b
        deriv = 2*a*ploty + b
        # normal offset in x for each y: shift = lane_width_pixels / sqrt(1 + deriv^2)
        shift = (lane_width_pixels / np.sqrt(1.0 + deriv**2))
        # left is typically to the left (smaller x) => subtract shift
        left_x = right_x - shift
        # fit polynomial to left_x(y)
        left_fit = np.polyfit(ploty, left_x, 2)  # fit x = f(y)
        # convert to expected [a,b,c] where we used np.polyfit(y, x)
        # np.polyfit returns [A,B,C] such that x = A*y^2 + B*y + C -> ok
    elif right_fit is None and left_fit is not None:
        a, b, c = left_fit
        left_x = a*ploty**2 + b*ploty + c
        deriv = 2*a*ploty + b
        shift = (lane_width_pixels / np.sqrt(1.0 + deriv**2))
        right_x = left_x + shift
        right_fit = np.polyfit(ploty, right_x, 2)

    # 6) smoothing coefficients
    alpha = _lane_smooth.get("alpha", 0.15)
    left_fit = np.array(left_fit); right_fit = np.array(right_fit)
    prev_left = _lane_smooth.get("left_fit"); prev_right = _lane_smooth.get("right_fit")
    left_fit = _smooth_fit(prev_left, left_fit, alpha)
    right_fit = _smooth_fit(prev_right, right_fit, alpha)
    _lane_smooth["left_fit"] = left_fit; _lane_smooth["right_fit"] = right_fit

    # 7) generate x points and draw
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    color_warp = np.zeros_like(frame)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))], dtype=np.int32)
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))
    cv2.polylines(color_warp, [pts_left.reshape(-1,2)], isClosed=False, color=(0,200,0), thickness=6)
    cv2.polylines(color_warp, [pts_right.reshape(-1,2)], isClosed=False, color=(0,200,0), thickness=6)

    center_x = (left_fitx + right_fitx) / 2.0
    center_pts = np.array([np.transpose(np.vstack([center_x, ploty]))], dtype=np.int32)
    cv2.polylines(color_warp, [center_pts.reshape(-1,2)], isClosed=False, color=(0,0,255), thickness=6)

    newwarp = cv2.warpPerspective(color_warp, Minv, (w, h))
    result = cv2.addWeighted(frame, 0.7, newwarp, 0.6, 0)

    # 8) offset & arrow
    lane_center_x = center_x[-1]
    vehicle_center_x = w / 2.0
    xm_per_pix = 3.7 / float(lane_width_pixels) if lane_width_pixels > 0 else 3.7/700.0
    offset_m = (vehicle_center_x - lane_center_x) * xm_per_pix
    side = "left" if offset_m > 0 else "right"
    cv2.putText(result, f"Offset: {abs(offset_m):.2f} m {side}", (30,60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

    dx = int(lane_center_x - vehicle_center_x)
    start_pt = (int(w*0.12), 100)
    end_pt = (int(w*0.12 + np.clip(dx, -200, 200)), 100)
    cv2.arrowedLine(result, start_pt, end_pt, (0,255,255), 6, tipLength=0.4)

    return result

# ------------------- Flask app & endpoints -------------------
app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app)

sign_detector = SignDetector(model_name='yolov8n.pt', device=YOLO_DEVICE, conf_threshold=0.35)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/process_frame', methods=['POST'])
def process_frame():
    start = time.time()
    if 'frame' not in request.files:
        return jsonify({'error': 'no frame uploaded'}), 400
    f = request.files['frame']
    data = f.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'invalid image'}), 400

    lane_img = detect_lanes(img)

    signs = sign_detector.detect(img)
    for s in signs:
        x1, y1, x2, y2 = s['x1'], s['y1'], s['x2'], s['y2']
        label = s['label']; conf = s['conf']
        cv2.rectangle(lane_img, (x1, y1), (x2, y2), (0,0,255), 2)
        cv2.putText(lane_img, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    _, jpeg = cv2.imencode('.jpg', lane_img)
    b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')
    elapsed = time.time() - start
    return jsonify({'image': b64, 'signs': signs, 'processing_time': elapsed})

@app.route('/debug_mask', methods=['POST'])
def debug_mask():
    if 'frame' not in request.files:
        return jsonify({'error': 'no frame uploaded'}), 400
    f = request.files['frame']
    data = f.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'invalid image'}), 400

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.abs(sobelx)
    maxv = abs_sobelx.max() if abs_sobelx.max() != 0 else 1.0
    scaled = np.uint8(255 * abs_sobelx / maxv)
    sxbinary = np.zeros_like(scaled)
    sxbinary[(scaled >= 20) & (scaled <= 200)] = 1
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= 95) & (s_channel <= 255)] = 1
    combined = np.zeros_like(sxbinary)
    combined[(s_binary == 1) | (sxbinary == 1)] = 1
    binary = (combined * 255).astype(np.uint8)

    binary_col = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    binary_col[:, :, 1] = binary
    overlay = cv2.addWeighted(img, 0.7, binary_col, 0.7, 0)

    _, b_jpeg = cv2.imencode('.jpg', binary)
    _, o_jpeg = cv2.imencode('.jpg', overlay)
    b64_bin = base64.b64encode(b_jpeg.tobytes()).decode('utf-8')
    b64_ov = base64.b64encode(o_jpeg.tobytes()).decode('utf-8')
    return jsonify({'mask': b64_bin, 'overlay': b64_ov})

# ------------------- Run -------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on http://0.0.0.0:{port}  (YOLO device = {YOLO_DEVICE})")
    app.run(host='0.0.0.0', port=port, debug=True)
