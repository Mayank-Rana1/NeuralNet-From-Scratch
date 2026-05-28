<div align="center">

# Digit Inference Engine
**From-Scratch Neural Network to Edge Deployment**

[![Live Demo](https://img.shields.io/badge/Live_Demonstration-Test_the_Inference_Engine_Here-blue?style=for-the-badge&logo=googlechrome&logoColor=white)](https://mayank-rana1.github.io/NeuralNet-From-Scratch/)

</div>

<br>

## Project Philosophy
The objective of this project was to completely demystify the "black box" of machine learning. Rather than relying on modern frameworks from the start, this project was built by hand-coding the foundational linear algebra and calculus required for a neural network to learn. Once the raw mathematics were proven, the architecture was translated into PyTorch, optimized for edge-compute, and deployed via a custom Vanilla JavaScript computer vision pipeline.

## The Engineering Pipeline

The repository is structured sequentially to demonstrate the progression from mathematical theory to a production-ready web application.

### Phase 1: The Mathematical Foundation (Zero Dependencies)
**File:** `01_Neural_Network_From_Scratch.ipynb`
* **Architecture:** Single-layer neural network.
* **Implementation:** Engineered entirely from scratch using only standard Python and NumPy. 
* **Core Mechanics:** Hand-coded matrix multiplication for the forward pass, raw calculus chain rule derivatives for backward propagation, and Stochastic Gradient Descent (SGD) for weight updates.
* **Results & Limitations:** Achieved ~89% accuracy. The model hit a "glass ceiling" because a flat layer can only read raw pixels, lacking the capacity to detect complex geometric shapes.

### Phase 2: Deepening the Network
**File:** `02_Two_Layer_Neural_Network.ipynb`
* **Architecture:** Multi-Layer Perceptron (Deep Learning).
* **Implementation:** Upgraded the foundational math to support hidden layers.
* **Core Mechanics:** Implemented non-linear activation functions (ReLU, Softmax) and managed vanishing gradients during deep backpropagation.
* **Results:** Accuracy improved to ~97%.

### Phase 3: Framework Translation 
**File:** `03_PyTorch_Baseline.ipynb`
* **Implementation:** The custom NumPy architecture was explicitly translated into PyTorch. This validated that the hand-coded mathematics perfectly mirrored professional framework behavior before introducing complex convolutions.

### Phase 4: CNN Architecture & Model Optimization
**File:** `04_Optimized_CNN.py`
* **Architecture Upgrade:** Migrated to a Convolutional Neural Network (CNN) to handle spatial anomalies (tilted handwriting, off-center numbers). Utilized 2x `Conv2d` layers paired with `MaxPool2d` to reduce the final linear connections from 25,088 to just 1,568, drastically cutting memory requirements.
* **Training Optimizations:** * Implemented strict Data Augmentation (`RandomRotation(20)`, `RandomAffine(translate=0.1, scale=0.9-1.1)`) to force the model to learn messy, real-world handwriting.
  * Added a `Dropout(0.3)` layer to prevent memorization/overfitting.
* **Quantization & Export:** The trained PyTorch model was dynamically quantized to 8-bit integers (`QUInt8`), compressing the ONNX file size by ~4x (from roughly 900KB down to ~225KB) with zero noticeable loss in inference accuracy.

### Phase 5: Client-Side Computer Vision & Inference (The Web App)
**Files:** `index.html` | `script.js` | `style.css`

The front-end does not rely on a backend server. The model runs locally in the user's browser, processing a 1x1x28x28 float32 tensor mapped from a live camera feed. 

**Front-End Tech Stack:** HTML5, CSS3 (CSS Variables, Flexbox, Backdrop-Filter), Vanilla JavaScript (ES6+), HTML5 Canvas API, WebRTC (MediaDevices API), ONNX Runtime Web (`ort.min.js`).

**Custom Image Processing Algorithms:**
To bridge the gap between a high-resolution 1080p webcam and a tiny 28x28 neural network input, two custom algorithms were engineered in JavaScript:
1. **Smooth Multi-Step Downscaling:** Instead of shrinking the 140x140 CSS target box directly to 28x28 (which causes browsers to drop pixels and erases thin pen lines), the Canvas API dynamically funnels the image through intermediate states (112x112 → 56x56 → 28x28). This preserves the continuous stroke of a 1px ballpoint pen.
2. **Local Adaptive Thresholding:** Global contrast adjustments fail when shadows fall across paper. A custom algorithm was written to scan a 9x9 pixel grid (Radius: 4) around every single pixel to calculate the local average brightness of the paper. It strictly isolates ink that is mathematically darker than its immediate neighbors (using an 8-shade buffer), effectively destroying shadows and horizontal blue notebook lines, forcing the background to pure black (`0.0`) and the ink to pure white (`1.0`).

---

## Local Execution
To run the web application locally for testing:
1. Clone this repository to your local machine.
2. Ensure the quantized `digit_brain.onnx` file is located in the `page/` subfolder alongside the web assets.
3. Open `index.html` using a local server environment (e.g., VS Code Live Server, Python `http.server`) to bypass browser cross-origin restrictions on local camera access.
