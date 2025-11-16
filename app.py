from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import subprocess
from pathlib import Path
from cnnClassifier.pipeline.prediction import PredictionPipeline
from cnnClassifier.utils.common import decodeImage
import uvicorn
import traceback

# Set environment variables
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = FastAPI(title="Chest Cancer Classification API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = None  # Don't load model at startup
    
    def get_classifier(self):
        """Lazy load the classifier only when needed"""
        if self.classifier is None:
            try:
                print(f"Loading model for prediction...")
                self.classifier = PredictionPipeline(self.filename)
                print(f"Model loaded successfully!")
            except Exception as e:
                error_msg = f"Failed to load model: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                raise Exception(error_msg)
        return self.classifier

# Initialize client app (without loading model)
clApp = ClientApp()

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chest Cancer Classification</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                max-width: 900px;
                width: 100%;
                padding: 40px;
                animation: fadeIn 0.5s ease-in;
            }
            
            @keyframes fadeIn {
                from {
                    opacity: 0;
                    transform: translateY(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2.5em;
            }
            
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
                font-size: 1.1em;
            }
            
            .upload-section {
                border: 3px dashed #667eea;
                border-radius: 15px;
                padding: 40px;
                text-align: center;
                margin-bottom: 30px;
                transition: all 0.3s ease;
                background: #f8f9ff;
            }
            
            .upload-section:hover {
                border-color: #764ba2;
                background: #f0f2ff;
            }
            
            .upload-section.dragover {
                border-color: #764ba2;
                background: #e8ebff;
                transform: scale(1.02);
            }
            
            .upload-icon {
                font-size: 4em;
                color: #667eea;
                margin-bottom: 20px;
            }
            
            .file-input-wrapper {
                position: relative;
                display: inline-block;
            }
            
            input[type="file"] {
                display: none;
            }
            
            .file-label {
                display: inline-block;
                padding: 15px 40px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 50px;
                cursor: pointer;
                font-size: 1.1em;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }
            
            .file-label:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
            }
            
            .preview-section {
                margin: 30px 0;
                text-align: center;
            }
            
            .image-preview {
                max-width: 100%;
                max-height: 400px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                margin: 20px 0;
                display: none;
            }
            
            .image-preview.show {
                display: inline-block;
            }
            
            .button-group {
                display: flex;
                gap: 15px;
                justify-content: center;
                margin-top: 30px;
                flex-wrap: wrap;
            }
            
            button {
                padding: 15px 40px;
                border: none;
                border-radius: 50px;
                font-size: 1.1em;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            }
            
            .predict-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            .predict-btn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
            }
            
            .train-btn {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
            }
            
            .train-btn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(245, 87, 108, 0.6);
            }
            
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }
            
            .result-section {
                margin-top: 30px;
                padding: 25px;
                border-radius: 15px;
                display: none;
                animation: slideIn 0.5s ease-out;
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .result-section.show {
                display: block;
            }
            
            .result-normal {
                background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
                color: #155724;
            }
            
            .result-cancer {
                background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                color: #721c24;
            }
            
            .result-title {
                font-size: 1.5em;
                font-weight: 700;
                margin-bottom: 10px;
            }
            
            .result-description {
                font-size: 1.1em;
                margin-top: 10px;
            }
            
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            
            .loading.show {
                display: block;
            }
            
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .file-name {
                margin-top: 15px;
                color: #666;
                font-size: 0.9em;
            }
            
            .info-box {
                background: #e3f2fd;
                border-left: 4px solid #2196f3;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            
            .info-box p {
                margin: 5px 0;
                color: #1565c0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🫁 Chest Cancer Classification</h1>
            <p class="subtitle">AI-Powered Medical Image Analysis</p>
            
            <div class="info-box">
                <p><strong>Model:</strong> VGG16 Transfer Learning</p>
                <p><strong>Classes:</strong> Normal | Adenocarcinoma Cancer</p>
                <p><strong>Image Size:</strong> 224x224 pixels</p>
            </div>
            
            <div class="upload-section" id="uploadSection">
                <div class="upload-icon">📁</div>
                <h3>Upload Chest CT Scan Image</h3>
                <p style="color: #666; margin: 15px 0;">Drag and drop an image or click to browse</p>
                <div class="file-input-wrapper">
                    <input type="file" id="fileInput" accept="image/*">
                    <label for="fileInput" class="file-label">Choose Image</label>
                </div>
                <div class="file-name" id="fileName"></div>
            </div>
            
            <div class="preview-section">
                <img id="imagePreview" class="image-preview" alt="Preview">
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 15px; color: #666;">Processing image...</p>
            </div>
            
            <div class="button-group">
                <button class="predict-btn" id="predictBtn" onclick="predictImage()" disabled>🔍 Predict</button>
                <button class="train-btn" id="trainBtn" onclick="trainModel()">🚀 Train Model</button>
            </div>
            
            <div class="result-section" id="resultSection">
                <div class="result-title" id="resultTitle"></div>
                <div class="result-description" id="resultDescription"></div>
            </div>
        </div>
        
        <script>
            const fileInput = document.getElementById('fileInput');
            const imagePreview = document.getElementById('imagePreview');
            const fileName = document.getElementById('fileName');
            const predictBtn = document.getElementById('predictBtn');
            const uploadSection = document.getElementById('uploadSection');
            const resultSection = document.getElementById('resultSection');
            const loading = document.getElementById('loading');
            
            let selectedFile = null;
            
            // File input change
            fileInput.addEventListener('change', function(e) {
                handleFile(e.target.files[0]);
            });
            
            // Drag and drop
            uploadSection.addEventListener('dragover', function(e) {
                e.preventDefault();
                uploadSection.classList.add('dragover');
            });
            
            uploadSection.addEventListener('dragleave', function(e) {
                e.preventDefault();
                uploadSection.classList.remove('dragover');
            });
            
            uploadSection.addEventListener('drop', function(e) {
                e.preventDefault();
                uploadSection.classList.remove('dragover');
                const file = e.dataTransfer.files[0];
                if (file && file.type.startsWith('image/')) {
                    handleFile(file);
                } else {
                    alert('Please drop an image file');
                }
            });
            
            function handleFile(file) {
                if (file && file.type.startsWith('image/')) {
                    selectedFile = file;
                    fileName.textContent = `Selected: ${file.name}`;
                    
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        imagePreview.src = e.target.result;
                        imagePreview.classList.add('show');
                        predictBtn.disabled = false;
                        resultSection.classList.remove('show');
                    };
                    reader.readAsDataURL(file);
                } else {
                    alert('Please select a valid image file');
                }
            }
            
            async function predictImage() {
                if (!selectedFile) {
                    alert('Please select an image first');
                    return;
                }
                
                predictBtn.disabled = true;
                loading.classList.add('show');
                resultSection.classList.remove('show');
                
                try {
                    const reader = new FileReader();
                    reader.onload = async function(e) {
                        const base64Image = e.target.result.split(',')[1];
                        
                        const response = await fetch('/predict', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ image: base64Image })
                        });
                        
                        const result = await response.json();
                        
                        loading.classList.remove('show');
                        predictBtn.disabled = false;
                        
                        displayResult(result);
                    };
                    reader.readAsDataURL(selectedFile);
                } catch (error) {
                    console.error('Error:', error);
                    alert('Error predicting image. Please try again.');
                    loading.classList.remove('show');
                    predictBtn.disabled = false;
                }
            }
            
            function displayResult(result) {
                const prediction = result[0].image;
                const resultTitle = document.getElementById('resultTitle');
                const resultDescription = document.getElementById('resultDescription');
                
                resultSection.classList.remove('result-normal', 'result-cancer');
                
                if (prediction === 'Normal') {
                    resultSection.classList.add('result-normal');
                    resultTitle.textContent = '✅ Normal - No Cancer Detected';
                    resultDescription.textContent = 'The CT scan appears to be normal. However, please consult with a medical professional for a definitive diagnosis.';
                } else {
                    resultSection.classList.add('result-cancer');
                    resultTitle.textContent = '⚠️ Adenocarcinoma Cancer Detected';
                    resultDescription.textContent = 'The model has detected signs of adenocarcinoma. Please consult with a medical professional immediately for further evaluation and treatment.';
                }
                
                resultSection.classList.add('show');
            }
            
            async function trainModel() {
                const trainBtn = document.getElementById('trainBtn');
                trainBtn.disabled = true;
                trainBtn.textContent = 'Training...';
                
                try {
                    const response = await fetch('/train', {
                        method: 'POST'
                    });
                    
                    const result = await response.text();
                    alert(result || 'Training completed successfully!');
                } catch (error) {
                    console.error('Error:', error);
                    alert('Error training model. Please check the console for details.');
                } finally {
                    trainBtn.disabled = false;
                    trainBtn.textContent = '🚀 Train Model';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict(image_data: dict):
    """
    Predict cancer from base64 encoded image
    """
    try:
        if 'image' not in image_data:
            raise HTTPException(status_code=400, detail="Image data not provided")
        
        # Decode base64 image
        try:
            decodeImage(image_data['image'], clApp.filename)
            print(f"Image decoded and saved to: {clApp.filename}")
        except Exception as e:
            error_msg = f"Error decoding image: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            raise HTTPException(status_code=400, detail=f"Error decoding image: {str(e)}")
        
        # Get classifier (lazy load)
        try:
            classifier = clApp.get_classifier()
        except Exception as e:
            error_msg = f"Model loading error: {str(e)}"
            print(error_msg)
            raise HTTPException(status_code=500, detail=f"Model not available. Please train the model first. Error: {str(e)}")
        
        # Make prediction
        try:
            result = classifier.predict()
            print(f"Prediction successful: {result}")
            return JSONResponse(content=result)
        except FileNotFoundError as e:
            error_msg = f"File not found: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            raise HTTPException(status_code=404, detail=f"Model or image file not found: {str(e)}")
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # This will show in terminal
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/train")
async def train():
    """
    Train the model by running main.py
    """
    try:
        # Run training script
        result = subprocess.run(
            ["python", "main.py"],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            return {"message": "Training completed successfully!", "output": result.stdout}
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Training failed: {result.stderr}"
            )
    
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Training timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)