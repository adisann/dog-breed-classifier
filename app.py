import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
import gdown
import base64

# Page configuration
st.set_page_config(
    page_title="CatXDog",
    page_icon="🐾",
    layout="centered"
)

class BreedClassifier:
    MODEL_PATH = "pet_breed_classifier_20epochs.h5"
    MODEL_FILE_ID = "1VbuDAwGEkJWDriGobrA_OdLrUI3FcaZl"
    
    def __init__(self):
        self.model = None
        self.class_names = []
        self.class_descriptions = {}
        self.class_animals = {}
        self._load_breed_data()
        self._load_model()
    
    def _load_breed_data(self):
        """Load breed data from CSV or create sample data."""
        if os.path.exists('class_mapping.csv'):
            breed_data = pd.read_csv('class_mapping.csv')
        else:
            # Sample data
            breed_data = pd.DataFrame({
                'nama_breed': ['abyssinian', 'american_bulldog', 'american_pit_bull_terrier'],
                'label': [0, 1, 2],
                'description': [
                    'Kucing Abyssinian memiliki tubuh ramping dan berbulu pendek',
                    'Anjing ras American Bulldog yang berotot dan kuat',
                    'Anjing American Pit Bull Terrier yang kuat dan setia'
                ],
                'jenis_hewan': ['kucing', 'anjing', 'anjing']
            })
        
        self.class_names = breed_data['nama_breed'].tolist()
        self.class_descriptions = dict(zip(breed_data['nama_breed'], breed_data['description']))
        self.class_animals = dict(zip(breed_data['nama_breed'], breed_data['jenis_hewan']))
    
    def download_model(self):
        """Download model from Google Drive."""
        url = f"https://drive.usercontent.google.com/download?id=1VbuDAwGEkJWDriGobrA_OdLrUI3FcaZl&authuser=0&confirm=t&uuid=9a559d18-52bd-409c-a202-7914b25cd01c&at=ALoNOglMFu2QaEbcLlFOjTdiiiJN%3A1747881750880"
        try:
            gdown.download(url, self.MODEL_PATH, quiet=False)
            return os.path.exists(self.MODEL_PATH)
        except:
            st.error("Failed to download model")
            return False
    
    def _load_model(self):
        """Load the TensorFlow model."""
        if not os.path.exists(self.MODEL_PATH):
            with st.spinner("Downloading model..."):
                if not self.download_model():
                    return
        
        try:
            with st.spinner("Loading model..."):
                self.model = tf.keras.models.load_model(self.MODEL_PATH, compile=False)
                st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    def preprocess_image(self, image):
        """Preprocess image for model input."""
        if isinstance(image, np.ndarray):
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image.convert("RGB").resize((224, 224))
            image = np.array(image)
        
        arr = np.expand_dims(image, axis=0)
        return tf.keras.applications.resnet.preprocess_input(arr)
    
    def predict(self, image_array):
        """Predict breed from preprocessed image array."""
        if self.model is None:
            return []
        
        try:
            preds = self.model.predict(image_array)[0]
            top_3_idx = np.argsort(preds)[-3:][::-1]
            
            results = []
            for idx in top_3_idx:
                if idx < len(self.class_names):
                    results.append({
                        "key": self.class_names[idx],
                        "confidence": float(preds[idx]) * 100
                    })
            return results
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return []
    
    def format_prediction(self, breed_key, confidence):
        """Format prediction output."""
        if breed_key not in self.class_descriptions:
            return "Tidak Dikenal", "Tidak dapat mengenali breed dari gambar ini."
        
        animal_type = self.class_animals.get(breed_key, 'unknown')
        breed_name = breed_key.replace('_', ' ').title()
        description = self.class_descriptions.get(breed_key, "Tidak ada deskripsi tersedia.")
        
        if animal_type == 'kucing':
            title = f"🐱 Kucing Ras {breed_name}"
        elif animal_type == 'anjing':
            title = f"🐶 Anjing Ras {breed_name}"
        else:
            title = f"🔍 {breed_name}"
            
        return title, description


class PetBreedClassifierUI:
    def __init__(self, classifier):
        self.classifier = classifier
        self.apply_custom_css()
    
    @staticmethod
    def apply_custom_css():
        """Apply custom CSS styles."""
        st.markdown("""
        <style>
        .result-card { 
            padding: 1.5rem; 
            border-radius: 15px; 
            margin-bottom: 1rem; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .result-card-secondary { 
            padding: 1rem; 
            border-radius: 10px; 
            margin-bottom: 1rem; 
            background-color: #f8f9fa;
            border-left: 4px solid #007bff;
        }
        .confidence-bar { 
            height: 25px; 
            border-radius: 12px; 
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def display_logo(self):
        """Display logo or text header."""
        if os.path.exists("Logo.jpg"):
            try:
                with open("Logo.jpg", "rb") as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode()
                st.markdown(
                    f'<div style="text-align: center;"><img src="data:image/jpg;base64,{img_base64}" width="200"/></div>',
                    unsafe_allow_html=True
                )
            except:
                st.markdown('<h1 style="text-align: center;">🐾 CatXDog</h1>', unsafe_allow_html=True)
        else:
            st.markdown('<h1 style="text-align: center;">🐾 CatXDog</h1>', unsafe_allow_html=True)
    
    def get_image_input(self):
        """Get image input from upload or camera."""
        uploaded = st.file_uploader("Upload gambar...", type=["jpg","jpeg","png"])
        camera = st.camera_input("Atau ambil foto dengan kamera")
        
        if uploaded:
            return Image.open(uploaded)
        elif camera:
            return Image.open(camera)
        return None
    
    def display_results(self, predictions):
        """Display prediction results."""
        if not predictions:
            st.warning("Tidak dapat membuat prediksi.")
            return
        
        st.markdown("### 🎯 Hasil Prediksi")
        
        # Top prediction
        top_pred = predictions[0]
        title, description = self.classifier.format_prediction(top_pred['key'], top_pred['confidence'])
        confidence = top_pred['confidence']
        
        color = "#28a745" if confidence > 80 else "#ffc107" if confidence > 60 else "#dc3545"
        
        st.markdown(
            f"""<div class="result-card">
                <h2 style="text-align: center; margin-top: 0;">{title}</h2>
                <p style="text-align: center; margin: 1rem 0;">{description}</p>
                <p style="text-align: center;">Kepercayaan: {confidence:.1f}%</p>
                <div style="background-color: rgba(255,255,255,0.3); width:100%; border-radius:12px;">
                    <div class="confidence-bar" style="width: {confidence:.1f}%; background-color: {color};">
                        {confidence:.1f}%
                    </div>
                </div>
            </div>""", 
            unsafe_allow_html=True
        )
        
        # Other predictions
        if len(predictions) > 1:
            st.markdown("### 🔍 Kemungkinan Lain")
            for i, pred in enumerate(predictions[1:], 1):
                title, description = self.classifier.format_prediction(pred['key'], pred['confidence'])
                confidence = pred['confidence']
                colors = ["#17a2b8", "#fd7e14"]
                color = colors[min(i-1, len(colors)-1)]
                
                st.markdown(
                    f"""<div class="result-card-secondary">
                        <h4 style="color: {color}; margin-top: 0;">{title}</h4>
                        <p>{description}</p>
                        <p style="color: #666;">Kepercayaan: {confidence:.1f}%</p>
                        <div style="background-color: #e9ecef; width:100%; border-radius:10px; height: 20px;">
                            <div class="confidence-bar" style="width: {confidence:.1f}%; background-color: {color}; height: 20px;">
                                {confidence:.1f}%
                            </div>
                        </div>
                    </div>""", 
                    unsafe_allow_html=True
                )
    
    def run(self):
        """Run the application."""
        self.display_logo()
        st.markdown("### 🐾 Pengenal Ras Kucing dan Anjing")
        st.markdown("Upload foto kucing atau anjing untuk mengetahui rasnya! 📸")
        
        img = self.get_image_input()
        
        if img is not None:
            st.image(img, caption="Foto hewan peliharaan", use_column_width=True)
            
            with st.spinner('Menganalisis gambar... 🔍'):
                arr = self.classifier.preprocess_image(img)
                predictions = self.classifier.predict(arr)
                self.display_results(predictions)


def main():
    """Main function to run the application."""
    classifier = BreedClassifier()
    app = PetBreedClassifierUI(classifier)
    app.run()

if __name__ == "__main__":
    main()