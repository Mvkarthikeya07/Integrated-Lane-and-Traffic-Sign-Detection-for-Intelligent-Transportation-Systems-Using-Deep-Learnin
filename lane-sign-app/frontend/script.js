// frontend/script.js
const BACKEND = window.location.origin; // assumes backend serves frontend (see app.py)
const video = document.getElementById('video');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const captureBtn = document.getElementById('captureBtn');
const intervalInput = document.getElementById('interval');
const statusSpan = document.getElementById('status');
const fileInput = document.getElementById('fileInput');
const resultImage = document.getElementById('resultImage');
const signCount = document.getElementById('signCount');
const procTime = document.getElementById('procTime');

let stream = null;
let captureInterval = null;
let busy = false;

startBtn.addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    captureBtn.disabled = false;
    statusSpan.textContent = 'Camera started';
    startAutoCapture();
  } catch (e) {
    alert('Camera access error: ' + e.message);
  }
});

stopBtn.addEventListener('click', () => {
  if (stream) {
    const tracks = stream.getTracks();
    tracks.forEach(t => t.stop());
    video.srcObject = null;
    stream = null;
  }
  clearInterval(captureInterval);
  captureInterval = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  captureBtn.disabled = true;
  statusSpan.textContent = 'Camera stopped';
});

captureBtn.addEventListener('click', () => {
  if (!busy) captureAndSend();
});

fileInput.addEventListener('change', (e) => {
  const f = e.target.files[0];
  if (f) sendFile(f);
});

function startAutoCapture() {
  const ms = parseInt(intervalInput.value) || 800;
  if (captureInterval) clearInterval(captureInterval);
  captureInterval = setInterval(() => {
    if (!busy) captureAndSend();
  }, ms);
}

intervalInput.addEventListener('change', startAutoCapture);

async function captureAndSend(){
  if (!video || !video.videoWidth) return;
  const w = video.videoWidth;
  const h = video.videoHeight;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, w, h);
  const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.9));
  await sendBlob(blob);
}

async function sendFile(file){
  await sendBlob(file);
}

async function sendBlob(blob){
  busy = true;
  statusSpan.textContent = 'Processing...';
  const t0 = performance.now();
  const fd = new FormData();
  fd.append('frame', blob, 'frame.jpg');
  try {
    const res = await fetch(`${BACKEND}/process_frame`, {
      method: 'POST',
      body: fd
    });
    const data = await res.json();
    if (data.image) {
      resultImage.src = 'data:image/jpeg;base64,' + data.image;
    }
    signCount.textContent = data.signs ? data.signs.length : 0;
    procTime.textContent = (data.processing_time || 0).toFixed(2);
  } catch (err) {
    console.error(err);
    alert('Error: ' + err.message);
  } finally {
    const t1 = performance.now();
    statusSpan.textContent = 'Idle';
    busy = false;
  }
}
