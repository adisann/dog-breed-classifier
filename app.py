import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
import gdown
import base64
import time
from pathlib import Path
import json
import h5py
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.models import model_from_config

# Page configuration
st.set_page_config(
    page_title="CatXDog",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

class BreedClassifier:
    """Handle pet breed classification using a pre-trained TensorFlow model."""
    MODEL_PATH = "pet_breed_classifier_final.h5"
    MODEL_FILE_ID = "1GDOwEq3pHwy1ftngOzCQCllXNtawsueI"
    MIN_MODEL_SIZE_MB = 50

    def __init__(self):
        self.model = None
        self.class_names = []
        self.cat_breeds = []
        self.class_descriptions = {}
        self._load_model()
        self._load_class_data()

    @staticmethod
    @st.cache_resource
    def load_model_cached(model_path):
        """Try standard load; if that fails, patch the HDF5 config."""
        try:
            # First, try the simple way
            return tf.keras.models.load_model(model_path, compile=False)
        except ValueError as err:
            # Look for the specific 'batch_shape' complaint
            if "Unrecognized keyword arguments: ['batch_shape']" in str(err):
                # Repair the config JSON in the HDF5
                with h5py.File(model_path, 'r') as f:
                    raw = f.attrs.get('model_config')
                    if raw is None:
                        raise ValueError("Model config not found in HDF5 file")
                    cfg = raw.decode('utf-8')
                # Parse JSON
                config = json.loads(cfg)
                # Find and patch InputLayer
                for layer in config['config']['layers']:
                    if layer['class_name'] == 'InputLayer':
                        if 'batch_shape' in layer['config']:
                            layer['config']['batch_input_shape'] = layer['config'].pop('batch_shape')
                # Create model from config
                model = model_from_config(config, custom_objects={'InputLayer': InputLayer})
                # Load weights
                model.load_weights(model_path)
                return model
            else:
                raise

    def _load_class_data(self):
        """Load class names, categories, and descriptions from CSV."""
        if os.path.exists('class_mapping.csv'):
            df = pd.read_csv('class_mapping.csv')
            self.class_names = df['class_name'].tolist()
            if 'is_cat' in df:
                self.cat_breeds = df[df['is_cat']].class_name.tolist()
            if 'description' in df:
                self.class_descriptions = dict(zip(df['class_name'], df['description']))
        # fallback generic descriptions
        for k in self.class_names:
            self.class_descriptions.setdefault(k, k.replace('_',' ').title())

    def download_model(self, max_retries=3):
        """Download model from Google Drive if missing or corrupt."""
        url = f"https://drive.google.com/uc?id={self.MODEL_FILE_ID}"
        for attempt in range(max_retries):
            try:
                st.info(f"Downloading model (attempt {attempt+1})...")
                out = gdown.download(url, self.MODEL_PATH, quiet=False, fuzzy=True, resume=True)
                if not out or not os.path.exists(self.MODEL_PATH):
                    raise Exception("Download failed")
                size = os.path.getsize(self.MODEL_PATH)
                if size < self.MIN_MODEL_SIZE_MB*1024*1024:
                    os.remove(self.MODEL_PATH)
                    raise Exception("Corrupted download (too small)")
                st.success(f"Model downloaded: {size/(1024*1024):.1f} MB")
                return True
            except Exception as e:
                if attempt < max_retries-1:
                    st.warning(f"{e}, retrying…")
                    time.sleep(5)
                else:
                    st.error("Could not download model.")
                    return False

    def _load_model(self):
        """Ensure model file is present, then load it."""
        if not os.path.exists(self.MODEL_PATH) or os.path.getsize(self.MODEL_PATH) < self.MIN_MODEL_SIZE_MB*1024*1024:
            with st.spinner("Fetching model…"):
                if not self.download_model():
                    return
        try:
            with st.spinner("Loading model…"):
                self.model = self.load_model_cached(self.MODEL_PATH)
            st.success("Model loaded!")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    def preprocess_image(self, image, img_size=(224,224)):
        """Resize / normalize for ResNet."""
        try:
            if isinstance(image, np.ndarray):
                img = cv2.resize(image, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = image.convert("RGB").resize(img_size)
                img = np.array(img)
            arr = np.expand_dims(img, 0)
            return tf.keras.applications.resnet.preprocess_input(arr)
        except Exception as e:
            st.error(f"Image preprocessing error: {e}")
            return None

    def predict(self, arr):
        """Return top-3 (class, confidence)."""
        if self.model is None or arr is None:
            return []
        preds = self.model.predict(arr)
        idxs = np.argsort(preds[0])[-3:][::-1]
        out = []
        for i in idxs:
            if i < len(self.class_names):
                out.append({
                    "key":   self.class_names[i],
                    "conf":  float(preds[0][i]*100)
                })
        return out

    def format_prediction(self, key, conf):
        """Human-friendly title + description."""
        name = key.replace('_',' ').title()
        desc = self.class_descriptions.get(key, "")
        if key in self.cat_breeds:
            title = f"Cat: {name} ({conf:.1f} %)"
        else:
            title = f"Dog: {name} ({conf:.1f} %)"
        return title, desc

class PetBreedClassifierUI:
    """Streamlit UI for uploading images & showing predictions."""
    def __init__(self, clf: BreedClassifier):
        self.clf = clf
        self._apply_css()
        self._show_logo()

    def _apply_css(self):
        st.markdown("""
        <style>
          .result { padding:1rem; margin:0.5rem 0; border-radius:8px; background:#f9f9f9; }
          .bar { height:1rem; background:#4caf50; border-radius:4px; }
        </style>
        """, unsafe_allow_html=True)

    def _show_logo(self):
        logo = "Logo.jpg"
        if os.path.exists(logo):
            b64 = base64.b64encode(open(logo,'rb').read()).decode()
            st.markdown(f'<p align="center"><img src="data:image/jpeg;base64,{b64}" width="180"></p>', unsafe_allow_html=True)
        else:
            st.header("🐾 CatXDog")

    def run(self):
        st.subheader("Upload a cat or dog photo to identify its breed")
        img_file = st.file_uploader("", type=["jpg","jpeg","png"])
        if img_file:
            img = Image.open(img_file)
            st.image(img, use_column_width=True)
            arr = self.clf.preprocess_image(img)
            with st.spinner("Analyzing…"):
                preds = self.clf.predict(arr)
            if not preds:
                st.warning("No predictions—check the model load above.")
            else:
                st.markdown("### Top Prediction")
                title, desc = self.clf.format_prediction(preds[0]["key"], preds[0]["conf"])
                st.markdown(f'<div class="result"><b>{title}</b><p>{desc}</p><div class="bar" style="width:{preds[0]["conf"]:.1f}%"></div></div>', unsafe_allow_html=True)
                if len(preds)>1:
                    st.markdown("### Other Candidates")
                    for p in preds[1:]:
                        t, d = self.clf.format_prediction(p["key"], p["conf"])
                        st.markdown(f'<div class="result"><b>{t}</b><p>{d}</p><div class="bar" style="width:{p["conf"]:.1f}%"></div></div>', unsafe_allow_html=True)

def main():
    clf = BreedClassifier()
    ui  = PetBreedClassifierUI(clf)
    ui.run()
    st.caption("Powered by ResNet101 + transfer learning")

if __name__ == "__main__":
    main()
    

# import streamlit as st
# st.set_page_config(
#     page_title="CatXDog",
#     page_icon="🐾",
#     layout="centered",
#     initial_sidebar_state="collapsed"
# )
# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import os
# import cv2
# from PIL import Image
# import gdown
# import base64
# import time

# class BreedClassifier:
#     """Class to handle pet breed classification using a pre-trained TensorFlow model."""
#     MODEL_PATH = "pet_breed_classifier_final.h5"
    
#     @staticmethod
#     @st.cache_resource
#     def load_model_cached(model_path):
#         """Load model with caching to avoid reloading on each rerun."""
#         return tf.keras.models.load_model(model_path)
    
#     @staticmethod    
#     def download_model(model_path, max_retries=3):
#         """Download model from Google Drive with improved error handling."""
#         # Direct file ID for Google Drive
#         file_id = "1GDOwEq3pHwy1ftngOzCQCllXNtawsueI"
#         # https://drive.usercontent.google.com/download?id=1GDOwEq3pHwy1ftngOzCQCllXNtawsueI&export=download
#         # Use a more reliable download URL format
#         url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"
        
#         for attempt in range(max_retries):
#             try:
#                 st.info(f"Mengunduh model (usaha ke-{attempt+1})...")
                
#                 # Use gdown with more options for reliable downloads
#                 output = gdown.download(
#                     url, 
#                     model_path, 
#                     quiet=False,
#                     fuzzy=True,
#                     resume=True  # Enable resumable downloads
#                 )
                
#                 # Check if download was successful
#                 if output is None:
#                     raise Exception("Download failed with gdown")
                
#                 # Validate file exists and has proper size (at least 50MB)
#                 if os.path.exists(model_path):
#                     file_size = os.path.getsize(model_path)
#                     if file_size < 50 * 1024 * 1024:  # At least 50MB
#                         st.warning(f"File size smaller than expected ({file_size} bytes). Retrying...")
#                         os.remove(model_path)
#                         raise Exception("File too small, possibly corrupted or access denied.")
                    
#                     st.success(f"Model downloaded successfully ({file_size / (1024*1024):.1f} MB)")
#                     return True
#                 else:
#                     raise Exception("File not found after download.")
            
#             except Exception as e:
#                 if attempt < max_retries - 1:
#                     st.warning(f"Download failed: {str(e)}. Retrying in 5 seconds... (Attempt {attempt+1}/{max_retries})")
#                     time.sleep(5)
#                 else:
#                     st.error(f"Failed to download model after {max_retries} attempts: {str(e)}")
                    
#                     # Provide fallback options
#                     st.info("Coba alternatif berikut: 1) Refresh halaman, 2) Coba lagi nanti, 3) Pastikan koneksi internet stabil")
#                     return False
    
#     def _load_model(self):
#         """Load the TensorFlow model and class names with improved error handling."""
#         try:
#             # Check if model exists
#             if not os.path.exists(self.MODEL_PATH):
#                 with st.spinner("Model belum ditemukan. Mengunduh model dari Google Drive..."):
#                     if not self.download_model(self.MODEL_PATH):
#                         st.error("Gagal mengunduh model. Coba refresh halaman atau kembali lagi nanti.")
#                         return False

#             # Validate file before loading model
#             if not os.path.isfile(self.MODEL_PATH):
#                 st.error("File model tidak ditemukan.")
#                 return False
                
#             if os.path.getsize(self.MODEL_PATH) < 50 * 1024 * 1024:  # At least 50MB
#                 st.error("File model tidak valid atau korup. Silakan coba refresh halaman.")
#                 # Try to remove corrupted file
#                 try:
#                     os.remove(self.MODEL_PATH)
#                 except:
#                     pass
#                 return False

#             # Load model with progress indication
#             with st.spinner("Memuat model ke memori..."):
#                 self.model = self.load_model_cached(self.MODEL_PATH)
#                 st.success("Model berhasil dimuat!")
            
#             # Load class mapping if available
#             if os.path.exists('class_mapping.csv'):
#                 df = pd.read_csv('class_mapping.csv')
#                 self.class_names = df['class_name'].tolist()
            
#             return True
        
#         except Exception as e:
#             st.error(f"Error saat memuat model: {str(e)}")
#             # Add more details for debugging in production
#             import traceback
#             st.error(f"Detail error: {traceback.format_exc()}")
#             return False
    
#     def preprocess_image(self, image, img_size=(224, 224)):
#         """Preprocess image for model input."""
#         try:
#             if isinstance(image, np.ndarray):
#                 image = cv2.resize(image, img_size)
#                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             else:
#                 image = image.convert("RGB")  
#                 image = image.resize(img_size)
#                 image = np.array(image)
#             arr = np.expand_dims(image, axis=0)
#             return tf.keras.applications.resnet.preprocess_input(arr)
#         except Exception as e:
#             st.error(f"Error saat memproses gambar: {e}")
#             return None
    
#     def predict(self, image_array):
#         """Predict breed from preprocessed image array."""
#         if image_array is None or self.model is None:
#             return []
#         try:
#             preds = self.model.predict(image_array)
#             idxs = np.argsort(preds[0])[-3:][::-1]  # Get top 3 predictions
#             results = []
#             for i in idxs:
#                 if i < len(self.class_names):  # Make sure index is valid
#                     key = self.class_names[i]
#                     prob = float(preds[0][i]) * 100
#                     results.append({"key": key, "confidence": prob})
#             return results
#         except Exception as e:
#             st.error(f"Error saat melakukan prediksi: {e}")
#             return []
    
#     def format_prediction(self, breed_key, confidence):
#         """Format prediction output with title and description."""
#         if breed_key not in self.class_descriptions:
#             return "Tidak diketahui", "Tidak dapat mengenali ras hewan ini."
#         name = breed_key.replace('_', ' ').title()
#         desc = self.class_descriptions.get(breed_key, "Tidak ada deskripsi tersedia.")
#         if breed_key in self.cat_breeds:
#             title = f"Kucing dengan Ras {name}, Tingkat kepercayaan {confidence:.1f}%"
#         elif breed_key in ["not_catxdog", "garfield", "catdog"]:
#             title = name
#         else:
#             title = f"Anjing dengan Ras {name}, Tingkat kepercayaan {confidence:.1f}%"
#         return title, desc

# class PetBreedClassifierUI:
#     """UI class for handling the Streamlit interface."""
#     def __init__(self, classifier):
#         """Initialize UI with a classifier."""
#         self.classifier = classifier
#         self.apply_custom_css()
#         self.display_logo()
    
#     @staticmethod
#     def get_base64_image(path):
#         """Convert image to base64 string."""
#         try:
#             with open(path, "rb") as img_file:
#                 return base64.b64encode(img_file.read()).decode()
#         except Exception:
#             return None
    
#     def display_logo(self):
#         """Display logo if available, otherwise show text header."""
#         try:
#             # Path to logo image
#             img_path = "Logo.jpg"
#             if os.path.exists(img_path):
#                 img_base64 = self.get_base64_image(img_path)
#                 if img_base64:
#                     st.markdown(
#                         f"""
#                         <div style="text-align: center;">
#                             <img src="data:image/jpg;base64,{img_base64}" width="200"/>
#                         </div>
#                         """,
#                         unsafe_allow_html=True
#                     )
#                 else:
#                     st.markdown('<h1 style="text-align: center;">🐾 CatXDog</h1>', unsafe_allow_html=True)
#             else:
#                 st.markdown('<h1 style="text-align: center;">🐾 CatXDog</h1>', unsafe_allow_html=True)
#         except Exception:
#             st.markdown('<h1 style="text-align: center;">🐾 CatXDog</h1>', unsafe_allow_html=True)
    
#     @staticmethod
#     def apply_custom_css():
#         """Apply custom CSS styles to the UI."""
#         st.markdown("""
#         <style>
#         .main { padding: 0rem 1rem; }
#         .stApp { max-width: 100%; }
#         .custom-header { font-size: 2rem; text-align: center; margin-bottom: 1rem; color: #4CAF50; }
#         .result-card { padding: 1rem; border-radius: 10px; margin-bottom: 1rem; background-color: #f1f1f1; }
#         .confidence-bar { height: 20px; border-radius: 5px; }
#         .footer { text-align: center; margin-top: 2rem; font-size: 0.8rem; color: #666; }
#         </style>
#         """, unsafe_allow_html=True)
    
#     def render_header(self):
#         """Render the application header."""
#         st.markdown('<p class="custom-header">🐾 Cat and Dog Breeds Classifier</p>', unsafe_allow_html=True)
#         st.markdown('Upload foto anjing atau kucing untuk mengetahui breed-nya!')
    
#     def get_image_input(self):
#         """Get image input from upload or camera."""
#         uploaded = st.file_uploader("Upload gambar...", type=["jpg","jpeg","png"])
#         camera = st.camera_input("Atau ambil foto dengan kamera")
#         img = None
        
#         if uploaded:
#             try:
#                 img = Image.open(uploaded)
#             except Exception as e:
#                 st.error(f"Error saat membuka file: {e}")
#         elif camera:
#             try:
#                 img = Image.open(camera)
#             except Exception as e:
#                 st.error(f"Error saat mengakses kamera: {e}")
        
#         return img
    
#     def display_results(self, predictions):
#         """Display prediction results."""
#         if not predictions:
#             st.warning("Tidak dapat melakukan prediksi. Silakan coba gambar lain.")
#             return
        
#         st.markdown("### Hasil Prediksi")
        
#         # Top prediction
#         title, desc = self.classifier.format_prediction(predictions[0]['key'], predictions[0]['confidence'])
#         confidence = predictions[0]['confidence']
        
#         st.markdown(
#             f"""<div class="result-card">
#                 <h3>🏆 {title}</h3>
#                 <p>{desc}</p>
#                 <div style="background-color: #e0e0e0; width:100%; border-radius:5px;">
#                     <div class="confidence-bar" style="width: {confidence:.1f}%; background-color: #4CAF50;">{confidence:.1f}%</div>
#                 </div>
#             </div>""", 
#             unsafe_allow_html=True
#         )
        
#         # Other predictions
#         if len(predictions) > 1:
#             st.markdown("### Kemungkinan Lainnya")
            
#             for p in predictions[1:]:
#                 title, desc = self.classifier.format_prediction(p['key'], p['confidence'])
#                 confidence = p['confidence']
#                 color = "#2196F3" if p == predictions[1] else "#FF9800"
                
#                 st.markdown(
#                     f"""<div class="result-card">
#                         <h4>{title}</h4>
#                         <p>{desc}</p>
#                         <div style="background-color: #e0e0e0; width:100%; border-radius:5px;">
#                             <div class="confidence-bar" style="width: {confidence:.1f}%; background-color: {color};">{confidence:.1f}%</div>
#                         </div>
#                     </div>""", 
#                     unsafe_allow_html=True
#                 )
    
#     def render_footer(self):
#         """Render the application footer."""
#         st.markdown(
#             '<div class="footer">CatXDog menggunakan ResNet101 dengan transfer learning</div>', 
#             unsafe_allow_html=True
#         )
    
#     def run(self):
#         """Run the application."""
#         self.render_header()
#         img = self.get_image_input()
        
#         if img is not None:
#             st.image(img, caption="Foto hewan peliharaan", use_column_width=True)
            
#             with st.spinner('Menganalisis gambar...'):
#                 arr = self.classifier.preprocess_image(img)
                
#                 if arr is not None and self.classifier.model is not None:
#                     predictions = self.classifier.predict(arr)
#                     self.display_results(predictions)
#                 else:
#                     st.error("Tidak dapat menganalisis gambar. Pastikan model sudah dimuat dengan benar.")
        
#         self.render_footer()

# def main():
#     """Main function to run the application."""
#     classifier = BreedClassifier()
#     app = PetBreedClassifierUI(classifier)
#     app.run()

# if __name__ == "__main__":
#     main()