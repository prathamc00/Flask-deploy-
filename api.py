from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
import os
import sys
from pathlib import Path

# Add inference module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'inference'))
from deepfake_detector import DeepfakeDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize detector with absolute path for reliability
def get_model_path():
    """Get absolute path to model file"""
    current_dir = Path(__file__).parent
    model_path = current_dir / '..' / 'training' / 'models' / 'trained' / 'hybrid_deepfake_detector_improved.pth'
    return str(model_path.resolve())

# Check if model file exists before initializing
model_path = get_model_path()
if not os.path.exists(model_path):
    print(f"❌ Model file not found at: {model_path}")
    print("📁 Available model files:")
    models_dir = Path(__file__).parent / '..' / 'training' / 'models' / 'trained'
    if models_dir.exists():
        for file in models_dir.glob('*.pth'):
            print(f"   - {file.name}")
    else:
        print("   No trained models directory found")
    sys.exit(1)

# Initialize detector
print(f"🔄 Loading model from: {model_path}")
detector = DeepfakeDetector(
    model_path=model_path,
    threshold=0.0001
)
print("✅ Model loaded successfully!")

# Rest of your code remains the same...
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Deepfake Detection API</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
        .result { margin: 20px 0; padding: 20px; border-radius: 5px; }
        .fake { background-color: #ffebee; border-left: 5px solid #f44336; }
        .real { background-color: #e8f5e8; border-left: 5px solid #4caf50; }
        .error { background-color: #fff3cd; border-left: 5px solid #ff9800; }
        button { background-color: #2196f3; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #1976d2; }
        .stats { display: flex; justify-content: space-around; margin: 20px 0; }
        .stat { text-align: center; }
    </style>
</head>
<body>
    <h1>🔍 Deepfake Detection API</h1>
    <p>Upload an image to detect if it's a deepfake or real photo.</p>
    
    <div class="upload-area">
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" id="imageInput" accept="image/*" required>
            <br><br>
            <button type="submit">Analyze Image</button>
        </form>
    </div>
    
    <div id="result"></div>
    
    <div class="stats">
        <div class="stat">
            <h3>Model Performance</h3>
            <p>ROC AUC: <strong>0.8491</strong></p>
            <p>F1-Score: <strong>0.8897</strong></p>
        </div>
        <div class="stat">
            <h3>Optimal Settings</h3>
            <p>Threshold: <strong>0.0001</strong></p>
            <p>Accuracy: <strong>81.36%</strong></p>
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const fileInput = document.getElementById('imageInput');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select an image file');
                return;
            }
            
            formData.append('image', file);
            
            // Show loading
            document.getElementById('result').innerHTML = '<p>🔄 Analyzing image...</p>';
            
            fetch('/detect', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                let resultHtml = '';
                
                if (data.status === 'success') {
                    const resultClass = data.prediction === 'FAKE' ? 'fake' : 'real';
                    const emoji = data.prediction === 'FAKE' ? '🚨' : '✅';
                    
                    resultHtml = `
                        <div class="result ${resultClass}">
                            <h3>${emoji} Result: ${data.prediction}</h3>
                            <div class="stats">
                                <div class="stat">
                                    <strong>Confidence</strong><br>
                                    ${(data.confidence * 100).toFixed(1)}%
                                </div>
                                <div class="stat">
                                    <strong>Fake Probability</strong><br>
                                    ${(data.fake_probability * 100).toFixed(2)}%
                                </div>
                                <div class="stat">
                                    <strong>Real Probability</strong><br>
                                    ${(data.real_probability * 100).toFixed(2)}%
                                </div>
                            </div>
                        </div>
                    `;
                } else {
                    resultHtml = `
                        <div class="result error">
                            <h3>❌ Error</h3>
                            <p>${data.error}</p>
                        </div>
                    `;
                }
                
                document.getElementById('result').innerHTML = resultHtml;
            })
            .catch(error => {
                document.getElementById('result').innerHTML = `
                    <div class="result error">
                        <h3>❌ Network Error</h3>
                        <p>${error.message}</p>
                    </div>
                `;
            });
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Serve the web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    """API endpoint for deepfake detection"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided', 'status': 'error'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected', 'status': 'error'}), 400
    
    # Create uploads directory
    uploads_dir = Path('uploads')
    uploads_dir.mkdir(exist_ok=True)
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = uploads_dir / filename
    file.save(filepath)
    
    try:
        # Run detection
        result = detector.predict(filepath)
        
        # Clean up uploaded file
        filepath.unlink()
        
        return jsonify(result)
    
    except Exception as e:
        # Clean up on error
        if filepath.exists():
            filepath.unlink()
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'threshold': detector.threshold,
        'device': str(detector.device)
    })

if __name__ == '__main__':
    print("🚀 Starting Deepfake Detection API...")
    print("🌐 Web interface available at: http://localhost:5000")
    print("📡 API endpoint: http://localhost:5000/detect")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
