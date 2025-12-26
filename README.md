# 🚗 Lane and Traffic Sign Detection Web App

### Lightweight Lane Detection (LaneNet) + YOLOv8 Object/Sign Detection
**Tech Stack:** Flask • OpenCV • PyTorch • JavaScript • HTML/CSS

A real-time **ADAS (Advanced Driver Assistance System)** web application combining:
- **Lane detection** using a lightweight LaneNet segmentation model
- **Traffic sign & object detection** using YOLOv8n
- **Offset estimation**, curved lane overlays, and green lane-region visualization
- **Live camera streaming** and **image upload processing**
- **Full-stack architecture** using Flask (backend) + HTML/CSS/JS (frontend)

This project demonstrates practical computer vision skills using both **deep learning** and **classical image processing techniques**.

---

### **Real-Time Lane + Sign Detection**
<img width="1366" height="768" alt="2025-11-20" src="https://github.com/user-attachments/assets/6bbafed6-1bb8-41bd-bfb4-3088ad686797" />

### **Detection on Uploaded Images**
<img width="1366" height="768" alt="2025-11-19 (20)" src="https://github.com/user-attachments/assets/f842c549-d547-4ba7-85ad-b6620f2d2ea1" />

### **UI (Idle State)**
<img width="1366" height="768" alt="2025-11-19 (19)" src="https://github.com/user-attachments/assets/41ed2a00-6c43-4d59-9b88-9f418ce4da2b" />

---

## ⭐ Features

### 🛣️ Lane Detection (LaneNet)
- Lightweight and fast lane segmentation
- Lane mask → contour extraction → polynomial curve fitting
- Smooth curved-lane rendering
- Accurate offset estimation (e.g., "0.46m left/right")
- Green lane-region shading for visibility

### 🚦 YOLOv8 Traffic Sign & Object Detection
Detects:
- Cars
- Bikes
- Pedestrians
- Road signs
- Traffic lights
- Any YOLO-supported class

Each detection includes:
- Class label
- Confidence score
- Bounding box
- High-contrast color coding

### 🖥️ Clean & Modern Frontend UI
- Dual-panel responsive layout
- Real-time webcam detection
- Road-image upload support
- Adjustable frame interval
- Displays detection count & processing time

---

## 📁 Project Structure
```
lane-sign-app/
│
├── backend/
│   ├── app.py                 # Flask API server
│   ├── lanenet_model.py       # LaneNet model loader (PyTorch)
│   ├── postprocess.py         # Lane mask → lane curves + offset
│   ├── requirements.txt       # Backend dependencies
│   ├── models/
│   │    └── lanenet.pth       # LaneNet pretrained weights
│   └── yolov8n.pt             # YOLOv8n model file
│
├── frontend/
│   ├── index.html             # Main UI layout
│   ├── script.js              # Camera + API request logic
│   ├── styles.css             # Styling
│
└── README.md
```

---

## 🔧 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/lane-sign-app.git
cd lane-sign-app
```

### 2️⃣ Install backend dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3️⃣ Add the required models
Place the model files here:
```
backend/models/lanenet.pth
backend/yolov8n.pt
```

### 4️⃣ Run the backend server
```bash
python app.py
```

### 5️⃣ Open the web application
```url
http://127.0.0.1:5000
```

---

## 🧠 How It Works

### 🔹 LaneNet Pipeline
1. Input frame (webcam or uploaded image)
2. LaneNet → segmentation mask
3. Contour extraction + filtering
4. Polynomial lane curve fitting
5. Lane region shading (green)
6. Offset calculated from frame center
7. Final overlay rendered

### 🔹 YOLOv8 Pipeline
1. Detects vehicles, pedestrians, road signs, and lights
2. Draws bounding boxes with labels
3. Computes confidence scores
4. Returns detection stats

### ✔ Combined Output Includes:
- Lane curves
- Lane-region shading
- Offset measurement
- YOLO detections
- Real-time FPS & processing time

---

## 📦 Backend Requirements
```
flask
flask-cors
opencv-python
numpy
torch
ultralytics
```

---

## 🚀 Future Enhancements
- Lane Departure Warning (LDW)
- Traffic sign classification (speed limit, stop, etc.)
- Traffic light color detection
- Steering angle estimation
- Enhanced night-mode lane detection
- Mobile App (Flutter / React Native)

---

## 🤝 Contributing
Pull requests, issues, and improvements are welcome!

---

📜 License

This project is licensed under the MIT License.
