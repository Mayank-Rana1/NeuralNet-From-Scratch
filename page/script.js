const tabDraw = document.getElementById('tab-draw');
const tabCamera = document.getElementById('tab-camera');
const viewDraw = document.getElementById('draw-mode');
const viewCamera = document.getElementById('camera-mode');
const btnPredict = document.getElementById('predict-btn');
const btnSpinner = document.getElementById('btn-spinner');
const btnText = document.getElementById('btn-text');
const btnClear = document.getElementById('clear-btn');
const resultText = document.getElementById('result-text');

// --- Drawing Setup ---
const drawCanvas = document.getElementById('draw-canvas');
const dCtx = drawCanvas.getContext('2d');
let isDrawing = false;

function clearCanvas() {
    dCtx.fillStyle = "white";
    dCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
}
clearCanvas();

dCtx.lineWidth = 22; 
dCtx.lineCap = "round";
dCtx.lineJoin = "round";
dCtx.strokeStyle = "black";

function startPosition(e) { isDrawing = true; draw(e); }
function endPosition() { isDrawing = false; dCtx.beginPath(); }
function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    
    const rect = drawCanvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    
    const x = clientX - rect.left;
    const y = clientY - rect.top;

    dCtx.lineTo(x, y);
    dCtx.stroke();
    dCtx.beginPath();
    dCtx.moveTo(x, y);
}

drawCanvas.addEventListener('mousedown', startPosition);
drawCanvas.addEventListener('mouseup', endPosition);
drawCanvas.addEventListener('mousemove', draw);
drawCanvas.addEventListener('touchstart', startPosition, {passive: false});
drawCanvas.addEventListener('touchend', endPosition);
drawCanvas.addEventListener('touchmove', draw, {passive: false});
btnClear.addEventListener('click', clearCanvas);

// --- Camera Setup ---
const video = document.getElementById('webcam');
let cameraStream = null;

async function startCamera() {
    if (cameraStream) return;
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: "environment" } 
        });
        video.srcObject = cameraStream;
    } catch (err) {
        console.error("Camera error:", err);
        resultText.innerText = "ERR";
    }
}
function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
}

// --- Tabs ---
let currentMode = 'draw';
tabDraw.addEventListener('click', () => {
    currentMode = 'draw';
    tabDraw.classList.add('active');
    tabCamera.classList.remove('active');
    viewDraw.classList.add('active');
    viewCamera.classList.remove('active');
    stopCamera();
});
tabCamera.addEventListener('click', () => {
    currentMode = 'camera';
    tabCamera.classList.add('active');
    tabDraw.classList.remove('active');
    viewCamera.classList.add('active');
    viewDraw.classList.remove('active');
    startCamera();
});

// --- AI Engine ---
let session;
const aiCanvas = document.getElementById('ai-vision');
const aiCtx = aiCanvas.getContext('2d');

(async function loadModel() {
    try {
        session = await ort.InferenceSession.create('./page/digit_brain.onnx');
        console.log("Model initialized.");
    } catch (e) {
        console.error("Model load failure", e);
    }
})();

btnPredict.addEventListener('click', async () => {
    if (!session) return;
    
    // 1. Instantly trigger the Loading UI
    btnPredict.disabled = true;
    btnSpinner.classList.remove('hidden');
    btnText.innerText = "Processing...";
    
    resultText.innerText = "...";
    resultText.classList.add('processing-pulse');
    
    // 2. Wait exactly 50ms so the browser has time to visually draw the spinner
    // 2. Wait exactly 50ms so the browser has time to visually draw the spinner
    setTimeout(async () => {
        try {
            // ==========================================
            // A. SMOOTH MULTI-STEP DOWNSCALING
            // ==========================================
            if (currentMode === 'draw') {
                aiCtx.drawImage(drawCanvas, 0, 0, 28, 28);
            } else {
                const cw = video.clientWidth;
                const ch = video.clientHeight;
                const vw = video.videoWidth;
                const vh = video.videoHeight;
                const scale = Math.max(cw / vw, ch / vh);
                
                const scaledW = vw * scale;
                const scaledH = vh * scale;
                const offsetX = (cw - scaledW) / 2;
                const offsetY = (ch - scaledH) / 2;
                
                const targetBoxSize = 140; 
                const boxX = (cw - targetBoxSize) / 2;
                const boxY = (ch - targetBoxSize) / 2;
                
                const cropX = (boxX - offsetX) / scale;
                const cropY = (boxY - offsetY) / scale;
                const cropSize = targetBoxSize / scale;
                
                // STEP 1: Crop to 112x112
                const step1 = document.createElement('canvas');
                step1.width = 112; step1.height = 112;
                const ctx1 = step1.getContext('2d');
                ctx1.imageSmoothingEnabled = true;
                ctx1.imageSmoothingQuality = 'high';
                ctx1.drawImage(video, cropX, cropY, cropSize, cropSize, 0, 0, 112, 112);

                // STEP 2: Shrink to 56x56
                const step2 = document.createElement('canvas');
                step2.width = 56; step2.height = 56;
                const ctx2 = step2.getContext('2d');
                ctx2.imageSmoothingEnabled = true;
                ctx2.imageSmoothingQuality = 'high';
                ctx2.drawImage(step1, 0, 0, 112, 112, 0, 0, 56, 56);

                // STEP 3: Shrink to final 28x28 AI Canvas
                aiCtx.imageSmoothingEnabled = true;
                aiCtx.imageSmoothingQuality = 'high';
                aiCtx.drawImage(step2, 0, 0, 56, 56, 0, 0, 28, 28);
            }

            // ==========================================
            // B. LOCAL ADAPTIVE THRESHOLDING (Shadow-Proof)
            // ==========================================
            const imageData = aiCtx.getImageData(0, 0, 28, 28);
            const pixels = imageData.data;
            const inputFloatArray = new Float32Array(28 * 28);
            const grays = new Float32Array(28 * 28);

            // 1. Convert to simple grayscale array
            for (let i = 0; i < pixels.length; i += 4) {
                grays[i / 4] = 0.299 * pixels[i] + 0.587 * pixels[i+1] + 0.114 * pixels[i+2];
            }

            // 2. Scan every pixel using a local neighborhood
            const radius = 4; // Creates a 9x9 grid around every pixel
            
            for (let y = 0; y < 28; y++) {
                for (let x = 0; x < 28; x++) {
                    let sum = 0;
                    let count = 0;

                    // Check the neighboring pixels
                    for (let dy = -radius; dy <= radius; dy++) {
                        for (let dx = -radius; dx <= radius; dx++) {
                            let ny = y + dy;
                            let nx = x + dx;
                            // Make sure we don't look outside the canvas edges
                            if (ny >= 0 && ny < 28 && nx >= 0 && nx < 28) {
                                sum += grays[ny * 28 + nx];
                                count++;
                            }
                        }
                    }

                    // Calculate the average brightness of the paper right next to this pixel
                    let localAvg = sum / count;
                    let pixel = grays[y * 28 + x];
                    let finalColor = 0.0;

                    // THE MAGIC: Is this pixel a sharp, dark line compared to its immediate neighbors?
                    // (The '8' is our buffer to ignore faint notebook lines and paper texture)
                    if (pixel < localAvg - 8) {
                        
                        // It's ink! Calculate how dark it is and boost it to bright white.
                        let darkness = (localAvg - pixel) / 20.0; 
                        finalColor = Math.min(1.0, darkness);
                        
                    } else {
                        // It's paper, a shadow, or a notebook line. Crush it to pitch black.
                        finalColor = 0.0;
                    }

                    inputFloatArray[y * 28 + x] = finalColor;

                    // Update UI Canvas so you can see it working
                    let vColor = finalColor * 255;
                    let idx = (y * 28 + x) * 4;
                    pixels[idx] = vColor;
                    pixels[idx+1] = vColor;
                    pixels[idx+2] = vColor;
                    pixels[idx+3] = 255;
                }
            }
            aiCtx.putImageData(imageData, 0, 0);

            // ==========================================
            // C. INFERENCE
            // ==========================================
            const tensor = new ort.Tensor('float32', inputFloatArray, [1, 1, 28, 28]);
            const feeds = { 'input': tensor };
            const results = await session.run(feeds);
            const output = results.output.data;
            
            let bestGuess = 0;
            let highestScore = -Infinity;
            
            for (let i = 0; i < output.length; i++) {
                if (output[i] > highestScore) {
                    highestScore = output[i];
                    bestGuess = i;
                }
            }
            
            resultText.innerText = bestGuess;

        } catch (e) {
            console.error("Inference failed", e);
            resultText.innerText = "ERR";
        } finally {
            btnPredict.disabled = false;
            btnSpinner.classList.add('hidden');
            btnText.innerText = "Run Inference";
            resultText.classList.remove('processing-pulse');
        }
    }, 50);
});