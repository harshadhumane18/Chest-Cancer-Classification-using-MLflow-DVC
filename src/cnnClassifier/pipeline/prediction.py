import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from pathlib import Path


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model = None
        self.model_path = os.path.join("artifacts", "training", "model.keras")
        # Don't load model in __init__, load it when needed
    
    def _load_model(self):
        """Load model if not already loaded"""
        if self.model is None:
            try:
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"Model file not found at: {self.model_path}. Please train the model first by running 'python main.py'")
                print(f"Loading model from: {self.model_path}")
                self.model = load_model(self.model_path)
                print(f"Model loaded successfully!")
            except Exception as e:
                raise Exception(f"Error loading model: {str(e)}")
    
    def predict(self):
        """Make prediction on the image"""
        # Load model if not loaded
        self._load_model()
        
        try:
            # Check if image file exists
            if not os.path.exists(self.filename):
                raise FileNotFoundError(f"Image file not found: {self.filename}")
            
            # Load and preprocess image
            test_image = image.load_img(self.filename, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            
            # Normalize image (IMPORTANT: match training preprocessing)
            test_image = test_image / 255.0
            
            # Expand dimensions for batch
            test_image = np.expand_dims(test_image, axis=0)
            
            # Make prediction
            predictions = self.model.predict(test_image, verbose=0)
            result = np.argmax(predictions, axis=1)
            
            print(f"Prediction probabilities: {predictions}")
            print(f"Predicted class index: {result[0]}")
            
            # Return result based on class index
            if result[0] == 1:
                prediction = 'Normal'
            else:
                prediction = 'Adenocarcinoma Cancer'
            
            return [{"image": prediction}]
            
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")