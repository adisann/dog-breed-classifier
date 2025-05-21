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

# Page configuration
st.set_page_config(
    page_title="CatXDog",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

class BreedClassifier:
    """Class to handle pet breed classification using a pre-trained TensorFlow model."""
    
    MODEL_PATH = "pet_breed_classifier_final.h5"
    MODEL_FILE_ID = "1GDOwEq3pHwy1ftngOzCQCllXNtawsueI"
    MIN_MODEL_SIZE_MB = 50
    
    def __init__(self):
        """Initialize the classifier with model and class names."""
        self.model = None
        self.class_names = []
        self.cat_breeds = []  # Will be populated from class_mapping.csv if available
        self.class_descriptions = {}  # Will be populated with breed descriptions
        self._load_model()
        self._load_class_data()
    
    @staticmethod
    @st.cache_resource
    def load_model_cached(model_path):
        """Load model with caching to avoid reloading on each rerun."""
        return tf.keras.models.load_model(model_path)
    
    def _load_class_data(self):
        """Load class names, descriptions, and categories from CSV if available."""
        try:
            if os.path.exists('class_mapping.csv'):
                df = pd.read_csv('class_mapping.csv')
                self.class_names = df['class_name'].tolist()
                
                # If the CSV has these columns, use them
                if 'is_cat' in df.columns:
                    self.cat_breeds = df[df['is_cat'] == True]['class_name'].tolist()
                
                if 'description' in df.columns:
                    self.class_descriptions = dict(zip(df['class_name'], df['description']))
                else:
                    # Create generic descriptions if none provided
                    self.class_descriptions = {breed: f"A beautiful {breed.replace('_', ' ')} breed." 
                                              for breed in self.class_names}
                
                # Add special classifications
                special_classes = {
                    "not_catxdog": "This image does not appear to be a cat or dog.",
                    "garfield": "This appears to be Garfield or a cartoon cat.",
                    "catdog": "This appears to be a CatDog cartoon character."
                }
                self.class_descriptions.update(special_classes)
        except Exception as e:
            st.warning(f"Warning: Could not load class data: {e}")
    
    def download_model(self, max_retries=3):
        """Download model from Google Drive with improved error handling."""
        url = f"https://drive.google.com/uc?id={self.MODEL_FILE_ID}&export=download&confirm=t"
        
        for attempt in range(max_retries):
            try:
                st.info(f"Downloading model (attempt {attempt+1})...")
                
                # Use gdown with more options for reliable downloads
                output = gdown.download(
                    url, 
                    self.MODEL_PATH, 
                    quiet=False,
                    fuzzy=True,
                    resume=True  # Enable resumable downloads
                )
                
                # Check if download was successful
                if output is None:
                    raise Exception("Download failed with gdown")
                
                # Validate file exists and has proper size
                if os.path.exists(self.MODEL_PATH):
                    file_size = os.path.getsize(self.MODEL_PATH)
                    if file_size < self.MIN_MODEL_SIZE_MB * 1024 * 1024:  # At least 50MB
                        st.warning(f"File size smaller than expected ({file_size} bytes). Retrying...")
                        os.remove(self.MODEL_PATH)
                        raise Exception("File too small, possibly corrupted or access denied.")
                    
                    st.success(f"Model downloaded successfully ({file_size / (1024*1024):.1f} MB)")
                    return True
                else:
                    raise Exception("File not found after download.")
            
            except Exception as e:
                if attempt < max_retries - 1:
                    st.warning(f"Download failed: {str(e)}. Retrying in 5 seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(5)
                else:
                    st.error(f"Failed to download model after {max_retries} attempts: {str(e)}")
                    
                    # Provide fallback options
                    st.info("Try these alternatives: 1) Refresh the page, 2) Try again later, 3) Ensure stable internet connection")
                    return False
    
    def _load_model(self):
        """Load the TensorFlow model with improved error handling."""
        try:
            # Check if model exists
            if not os.path.exists(self.MODEL_PATH):
                with st.spinner("Model not found. Downloading from Google Drive..."):
                    if not self.download_model():
                        st.error("Failed to download model. Try refreshing the page or try again later.")
                        return False

            # Validate file before loading model
            if not os.path.isfile(self.MODEL_PATH):
                st.error("Model file not found.")
                return False
                
            if os.path.getsize(self.MODEL_PATH) < self.MIN_MODEL_SIZE_MB * 1024 * 1024:
                st.error("Model file is invalid or corrupted. Please try refreshing the page.")
                # Try to remove corrupted file
                try:
                    Path(self.MODEL_PATH).unlink(missing_ok=True)
                except:
                    pass
                return False

            # Load model with progress indication
            with st.spinner("Loading model into memory..."):
                self.model = self.load_model_cached(self.MODEL_PATH)
                st.success("Model loaded successfully!")
            
            return True
        
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image, img_size=(224, 224)):
        """Preprocess image for model input."""
        try:
            if isinstance(image, np.ndarray):
                image = cv2.resize(image, img_size)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image.convert("RGB")  
                image = image.resize(img_size)
                image = np.array(image)
            arr = np.expand_dims(image, axis=0)
            return tf.keras.applications.resnet.preprocess_input(arr)
        except Exception as e:
            st.error(f"Error processing image: {e}")
            return None
    
    def predict(self, image_array):
        """Predict breed from preprocessed image array."""
        if image_array is None or self.model is None:
            return []
        try:
            preds = self.model.predict(image_array)
            idxs = np.argsort(preds[0])[-3:][::-1]  # Get top 3 predictions
            results = []
            for i in idxs:
                if i < len(self.class_names):  # Make sure index is valid
                    key = self.class_names[i]
                    prob = float(preds[0][i]) * 100
                    results.append({"key": key, "confidence": prob})
            return results
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return []
    
    def format_prediction(self, breed_key, confidence):
        """Format prediction output with title and description."""
        if breed_key not in self.class_descriptions:
            return "Unknown", "Cannot recognize this pet breed."
        
        name = breed_key.replace('_', ' ').title()
        desc = self.class_descriptions.get(breed_key, "No description available.")
        
        if breed_key in ["not_catxdog", "garfield", "catdog"]:
            title = name
        elif breed_key in self.cat_breeds:
            title = f"Cat Breed: {name}, Confidence {confidence:.1f}%"
        else:
            title = f"Dog Breed: {name}, Confidence {confidence:.1f}%"
            
        return title, desc


class PetBreedClassifierUI:
    """UI class for handling the Streamlit interface."""
    
    def __init__(self, classifier):
        """Initialize UI with a classifier."""
        self.classifier = classifier
        self.apply_custom_css()
        self.display_logo()
    
    @staticmethod
    def get_base64_image(path):
        """Convert image to base64 string."""
        try:
            with open(path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        except Exception:
            return None
    
    def display_logo(self):
        """Display logo if available, otherwise show text header."""
        try:
            # Path to logo image
            img_path = "Logo.jpg"
            if os.path.exists(img_path):
                img_base64 = self.get_base64_image(img_path)
                if img_base64:
                    st.markdown(
                        f"""
                        <div style="text-align: center;">
                            <img src="data:image/jpg;base64,{img_base64}" width="200"/>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown('<h1 style="text-align: center;">🐾 CatXDog</h1>', unsafe_allow_html=True)
            else:
                st.markdown('<h1 style="text-align: center;">🐾 CatXDog</h1>', unsafe_allow_html=True)
        except Exception:
            st.markdown('<h1 style="text-align: center;">🐾 CatXDog</h1>', unsafe_allow_html=True)
    
    @staticmethod
    def apply_custom_css():
        """Apply custom CSS styles to the UI."""
        st.markdown("""
        <style>
        .main { padding: 0rem 1rem; }
        .stApp { max-width: 100%; }
        .custom-header { font-size: 2rem; text-align: center; margin-bottom: 1rem; color: #4CAF50; }
        .result-card { padding: 1rem; border-radius: 10px; margin-bottom: 1rem; background-color: #f1f1f1; }
        .confidence-bar { height: 20px; border-radius: 5px; }
        .footer { text-align: center; margin-top: 2rem; font-size: 0.8rem; color: #666; }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render the application header."""
        st.markdown('<p class="custom-header">🐾 Cat and Dog Breeds Classifier</p>', unsafe_allow_html=True)
        st.markdown('Upload a photo of a dog or cat to identify its breed!')
    
    def get_image_input(self):
        """Get image input from upload or camera."""
        uploaded = st.file_uploader("Upload image...", type=["jpg","jpeg","png"])
        camera = st.camera_input("Or take a photo with your camera")
        img = None
        
        if uploaded:
            try:
                img = Image.open(uploaded)
            except Exception as e:
                st.error(f"Error opening file: {e}")
        elif camera:
            try:
                img = Image.open(camera)
            except Exception as e:
                st.error(f"Error accessing camera: {e}")
        
        return img
    
    def display_results(self, predictions):
        """Display prediction results."""
        if not predictions:
            st.warning("Cannot make predictions. Please try another image.")
            return
        
        st.markdown("### Prediction Results")
        
        # Top prediction
        title, desc = self.classifier.format_prediction(predictions[0]['key'], predictions[0]['confidence'])
        confidence = predictions[0]['confidence']
        
        st.markdown(
            f"""<div class="result-card">
                <h3>🏆 {title}</h3>
                <p>{desc}</p>
                <div style="background-color: #e0e0e0; width:100%; border-radius:5px;">
                    <div class="confidence-bar" style="width: {confidence:.1f}%; background-color: #4CAF50;">{confidence:.1f}%</div>
                </div>
            </div>""", 
            unsafe_allow_html=True
        )
        
        # Other predictions
        if len(predictions) > 1:
            st.markdown("### Other Possibilities")
            
            for p in predictions[1:]:
                title, desc = self.classifier.format_prediction(p['key'], p['confidence'])
                confidence = p['confidence']
                color = "#2196F3" if p == predictions[1] else "#FF9800"
                
                st.markdown(
                    f"""<div class="result-card">
                        <h4>{title}</h4>
                        <p>{desc}</p>
                        <div style="background-color: #e0e0e0; width:100%; border-radius:5px;">
                            <div class="confidence-bar" style="width: {confidence:.1f}%; background-color: {color};">{confidence:.1f}%</div>
                        </div>
                    </div>""", 
                    unsafe_allow_html=True
                )
    
    def render_footer(self):
        """Render the application footer."""
        st.markdown(
            '<div class="footer">CatXDog uses ResNet101 with transfer learning</div>', 
            unsafe_allow_html=True
        )
    
    def run(self):
        """Run the application."""
        self.render_header()
        img = self.get_image_input()
        
        if img is not None:
            st.image(img, caption="Pet photo", use_column_width=True)
            
            with st.spinner('Analyzing image...'):
                arr = self.classifier.preprocess_image(img)
                
                if arr is not None and self.classifier.model is not None:
                    predictions = self.classifier.predict(arr)
                    self.display_results(predictions)
                else:
                    st.error("Cannot analyze image. Please ensure the model is loaded correctly.")
        
        self.render_footer()


def main():
    """Main function to run the application."""
    classifier = BreedClassifier()
    app = PetBreedClassifierUI(classifier)
    app.run()

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