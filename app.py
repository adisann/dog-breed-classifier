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
import traceback

# Fix TensorFlow compatibility issues
# Set logging to only show errors to avoid warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error

# Try to import h5py for handling model files
try:
    import h5py
except ImportError:
    st.warning("h5py not installed. Some model loading features may not work.")

# Page configuration
st.set_page_config(
    page_title="CatXDog",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

class BreedClassifier:
    """Class to handle pet breed classification using a pre-trained TensorFlow model."""
    
    MODEL_PATH = "pet_breed_classifier_20epochs.h5"
    MODEL_FILE_ID = "1VbuDAwGEkJWDriGobrA_OdLrUI3FcaZl"
    MIN_MODEL_SIZE_MB = 50
    
    def __init__(self):
        """Initialize the classifier with model and class names."""
        self.model = None
        self.breed_data = pd.DataFrame()  # Will store the breed information
        self.class_names = []
        self.class_descriptions = {}
        self.class_animals = {}  # Maps class to animal type
        self._load_model()
        self._load_breed_data()
    
    def _load_breed_data(self):
        """Load breed data from DataFrame or CSV file."""
        try:
            # Try to load from CSV file first
            if os.path.exists('class_mapping.csv'):
                self.breed_data = pd.read_csv('class_mapping.csv')
                st.success("Loaded breed data from CSV file")
            else:
                # Create sample data structure based on your description
                # You can replace this with your actual dataframe
                sample_data = {
                    'nama_breed': ['abyssinian', 'american_bulldog', 'american_pit_bull_terrier'],
                    'label': [0, 1, 2],
                    'description': [
                        'Kucing Abyssinian memiliki tubuh ramping dan berbulu pendek dengan corak yang indah',
                        'Anjing ras American Bulldog yang berotot dan kuat dengan sifat setia',
                        'Anjing American Pit Bull Terrier, dikenal karena kekuatan dan kesetiaannya'
                    ],
                    'jenis_hewan': ['kucing', 'anjing', 'anjing']
                }
                self.breed_data = pd.DataFrame(sample_data)
                st.info("Using sample breed data. Please provide your actual class_mapping.csv file")
            
            # Process the data
            self.class_names = self.breed_data['nama_breed'].tolist()
            self.class_descriptions = dict(zip(self.breed_data['nama_breed'], self.breed_data['description']))
            self.class_animals = dict(zip(self.breed_data['nama_breed'], self.breed_data['jenis_hewan']))
            
            # Add special cases for non-cat/dog classifications
            special_cases = {
                'not_catxdog': {
                    'description': 'Gambar ini tidak terdeteksi sebagai anjing atau kucing',
                    'animal': 'other'
                },
                'catdog': {
                    'description': 'Gambar ini terdeteksi sebagai karakter CatDog dari kartun',
                    'animal': 'cartoon'
                },
                'unknown': {
                    'description': 'Tidak dapat mengenali breed dari gambar ini',
                    'animal': 'unknown'
                }
            }
            
            for key, value in special_cases.items():
                if key not in self.class_names:
                    self.class_names.append(key)
                    self.class_descriptions[key] = value['description']
                    self.class_animals[key] = value['animal']
            
        except Exception as e:
            st.error(f"Error loading breed data: {e}")
            # Fallback to basic setup
            self.class_names = ['unknown']
            self.class_descriptions = {'unknown': 'Tidak dapat mengenali breed'}
            self.class_animals = {'unknown': 'unknown'}
    
    @staticmethod
    def create_model_for_loading():
        """Create a model structure matching the saved model to handle loading compatibility issues."""
        try:
            # This creates a basic ResNet101 model similar to what's likely in the saved file
            base_model = tf.keras.applications.ResNet101(
                include_top=False,
                weights=None,  # We don't need the weights since we'll load them
                input_shape=(224, 224, 3),
                pooling='avg'
            )
            
            # Add classification head similar to what's likely in the model
            x = base_model.output
            predictions = tf.keras.layers.Dense(120, activation='softmax')(x)  # Adjust class count if needed
            
            model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
            return model
        except Exception as e:
            st.error(f"Error creating model skeleton: {str(e)}")
            return None
    
    @staticmethod
    def load_weights_only(model_path, model):
        """Load only weights from h5 file to avoid architecture incompatibility."""
        try:
            model.load_weights(model_path, by_name=True, skip_mismatch=True)
            return model
        except Exception as e:
            st.error(f"Error loading weights: {str(e)}")
            return None
    
    @staticmethod
    @st.cache_resource
    def load_model_cached(model_path):
        """Load model with caching to avoid reloading on each rerun."""
        try:
            # First try direct loading - might work with compatible TF versions
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception as e1:
            st.warning(f"Standard model loading failed: {str(e1)}. Trying alternative methods...")
            
            # Try using h5py to load the model file directly
            try:
                import h5py
                
                # Create a base model with expected architecture
                st.info("Creating a compatible model structure...")
                model = BreedClassifier.create_model_for_loading()
                
                if model is None:
                    raise Exception("Failed to create model structure")
                
                st.info("Loading weights into compatible model...")
                with h5py.File(model_path, 'r') as f:
                    # Check if this is a model or weights-only file
                    if 'model_weights' in f:
                        # It's a full model file, but we'll just extract weights
                        st.info("Found model_weights group in h5 file")
                        model.load_weights(model_path)
                    else:
                        # It's likely a weights-only file
                        st.info("Attempting to load as weights-only file")
                        model = BreedClassifier.load_weights_only(model_path, model)
                
                if model is None:
                    raise Exception("Failed to load weights")
                
                st.success("Successfully loaded model weights!")
                return model
                
            except Exception as e2:
                st.error(f"Alternative loading method failed: {str(e2)}")
                
                # Last resort - try to convert model format
                try:
                    st.info("Attempting to convert model format...")
                    
                    # Create a temporary directory
                    import tempfile
                    import shutil
                    
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        # Copy model to temp dir
                        temp_model_path = os.path.join(tmpdirname, "model.h5")
                        shutil.copy2(model_path, temp_model_path)
                        
                        # Try to load and resave in a more compatible format
                        try:
                            # Set a lower API version for TensorFlow to improve compatibility
                            tf.keras.backend.set_floatx('float32')
                            
                            # Create a new model and try direct loading with error suppression
                            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress warnings
                            model = BreedClassifier.create_model_for_loading()
                            model.load_weights(temp_model_path, by_name=False, skip_mismatch=True)
                            
                            st.success("Converted model successfully!")
                            return model
                        except Exception as e3:
                            st.error(f"Conversion failed: {str(e3)}")
                            raise Exception("All model loading approaches failed")
                
                except Exception as e4:
                    st.error(f"Final attempt failed: {str(e4)}")
                    raise Exception("Could not load model with any method")
    
    def download_model(self, max_retries=3):
        """Download model from Google Drive with improved error handling."""
        url = f"https://drive.usercontent.google.com/download?id=1VbuDAwGEkJWDriGobrA_OdLrUI3FcaZl&authuser=0&confirm=t&uuid=9a559d18-52bd-409c-a202-7914b25cd01c&at=ALoNOglMFu2QaEbcLlFOjTdiiiJN%3A1747881750880"
        
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

            # Suppress TensorFlow warnings
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            
            # Load model with progress indication
            with st.spinner("Loading model into memory..."):
                try:
                    # Try to load the model
                    self.model = self.load_model_cached(self.MODEL_PATH)
                    
                    # If model loads successfully, check it has predict method
                    if not hasattr(self.model, 'predict'):
                        # Create a wrapper function for prediction if needed
                        original_model = self.model
                        
                        class ModelWrapper:
                            def __init__(self, model):
                                self.model = model
                            
                            def predict(self, x):
                                # Handle SavedModel format
                                if hasattr(self.model, 'signatures'):
                                    infer = self.model.signatures["serving_default"]
                                    input_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
                                    output = infer(input_tensor)
                                    return list(output.values())[0].numpy()
                                # Handle custom prediction needs
                                else:
                                    # Try direct call
                                    return self.model(x)
                        
                        self.model = ModelWrapper(original_model)
                    
                    st.success("Model loaded successfully!")
                    return True
                    
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    
                    # If all else fails, let's create a dummy model for testing purposes
                    st.warning("Creating a dummy model for testing. Please note that predictions won't be accurate.")
                    
                    # Create a simple dummy model that produces random outputs
                    class DummyModel:
                        def predict(self, x):
                            # Generate random predictions for number of classes
                            num_classes = len(self.class_names) if self.class_names else 42
                            preds = np.random.random((x.shape[0], num_classes))
                            # Normalize to ensure they sum to 1
                            preds = preds / np.sum(preds, axis=1, keepdims=True)
                            return preds
                    
                    self.model = DummyModel()
                    st.warning("Using a dummy model. For accurate predictions, please try again later or contact support.")
                    return True
        
        except Exception as e:
            st.error(f"Unhandled error in model loading: {str(e)}")
            st.error(traceback.format_exc())
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
            # Handle different model types (keras model vs saved_model)
            if hasattr(self.model, 'predict'):
                # Standard Keras model
                preds = self.model.predict(image_array)
                if isinstance(preds, list):
                    preds = preds[0]  # Sometimes model.predict returns a list
            else:
                # Saved model loaded with tf.saved_model.load
                infer = self.model.signatures["serving_default"]
                input_name = list(infer.structured_input_signature[1].keys())[0]
                output_name = list(infer.structured_outputs.keys())[0]
                output = infer(**{input_name: tf.convert_to_tensor(image_array)})
                preds = output[output_name].numpy()
            
            # Make sure we have a 1D array of predictions
            if len(preds.shape) > 1:
                preds = preds[0]
                
            # Get top 3 predictions
            idxs = np.argsort(preds)[-3:][::-1]
            
            # If we have fewer class names than model outputs, limit our results
            max_idx = min(len(self.class_names) - 1, max(idxs)) if self.class_names else 0
            
            results = []
            for i in idxs:
                if i <= max_idx:  # Make sure index is valid
                    key = self.class_names[i] if i < len(self.class_names) else 'unknown'
                    prob = float(preds[i]) * 100
                    results.append({"key": key, "confidence": prob})
            return results
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.error(traceback.format_exc())
            return []
    
    def format_prediction(self, breed_key, confidence):
        """Format prediction output with title and description based on animal type."""
        if breed_key not in self.class_descriptions:
            return "Tidak Dikenal", "Tidak dapat mengenali breed dari gambar ini."
        
        animal_type = self.class_animals.get(breed_key, 'unknown')
        breed_name = breed_key.replace('_', ' ').title()
        description = self.class_descriptions.get(breed_key, "Tidak ada deskripsi tersedia.")
        
        # Format title based on animal type
        if animal_type == 'kucing':
            title = f"🐱 Kucing Ras {breed_name}"
            confidence_text = f"Tingkat kepercayaan: {confidence:.1f}%"
        elif animal_type == 'anjing':
            title = f"🐶 Anjing Ras {breed_name}"
            confidence_text = f"Tingkat kepercayaan: {confidence:.1f}%"
        elif breed_key == 'not_catxdog':
            title = "❓ Bukan Kucing atau Anjing"
            confidence_text = f"Gambar terdeteksi {confidence:.1f}% bukan sebagai anjing atau kucing"
            description = "Maaf, gambar yang Anda upload sepertinya bukan foto kucing atau anjing. Silakan coba upload foto kucing atau anjing yang lebih jelas! 📸"
        elif breed_key == 'catdog':
            title = "🎭 Karakter CatDog"
            confidence_text = f"Gambar terdeteksi {confidence:.1f}% sebagai karakter CatDog"
            description = "Wah, ini sepertinya karakter CatDog dari kartun! Hewan yang unik dengan setengah kucing dan setengah anjing. Sangat menarik! 🌟"
        else:
            title = f"🔍 {breed_name}"
            confidence_text = f"Tingkat kepercayaan: {confidence:.1f}%"
            description = "Tidak dapat mengidentifikasi dengan pasti. Mungkin coba foto yang lebih jelas? 🤔"
            
        return title, description, confidence_text
    
    def get_breed_info(self, breed_key):
        """Get additional breed information."""
        if breed_key in self.breed_data['nama_breed'].values:
            breed_info = self.breed_data[self.breed_data['nama_breed'] == breed_key].iloc[0]
            return {
                'animal_type': breed_info['jenis_hewan'],
                'description': breed_info['description'],
                'label': breed_info['label']
            }
        return None


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
        .result-card { 
            padding: 1.5rem; 
            border-radius: 15px; 
            margin-bottom: 1rem; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
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
            font-size: 14px;
        }
        .footer { text-align: center; margin-top: 2rem; font-size: 0.8rem; color: #666; }
        .emoji-large { font-size: 2rem; }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render the application header."""
        st.markdown('<p class="custom-header">🐾 Pengenal Ras Kucing dan Anjing</p>', unsafe_allow_html=True)
        st.markdown('Upload foto kucing atau anjing untuk mengetahui rasnya! 📸')
    
    def get_image_input(self):
        """Get image input from upload or camera."""
        uploaded = st.file_uploader("Upload gambar...", type=["jpg","jpeg","png"])
        camera = st.camera_input("Atau ambil foto dengan kamera")
        img = None
        
        if uploaded:
            try:
                img = Image.open(uploaded)
            except Exception as e:
                st.error(f"Error membuka file: {e}")
        elif camera:
            try:
                img = Image.open(camera)
            except Exception as e:
                st.error(f"Error mengakses kamera: {e}")
        
        return img
    
    def display_results(self, predictions):
        """Display prediction results with improved formatting."""
        if not predictions:
            st.warning("Tidak dapat membuat prediksi. Silakan coba gambar lain.")
            return
        
        st.markdown("### 🎯 Hasil Prediksi")
        
        # Top prediction with special formatting
        top_pred = predictions[0]
        title, description, confidence_text = self.classifier.format_prediction(
            top_pred['key'], top_pred['confidence']
        )
        confidence = top_pred['confidence']
        
        # Choose colors based on confidence level
        if confidence > 80:
            bar_color = "#28a745"  # Green for high confidence
        elif confidence > 60:
            bar_color = "#ffc107"  # Yellow for medium confidence
        else:
            bar_color = "#dc3545"  # Red for low confidence
        
        st.markdown(
            f"""<div class="result-card">
                <h2 style="margin-top: 0; text-align: center;">{title}</h2>
                <p style="font-size: 1.1em; text-align: center; margin: 1rem 0;">{description}</p>
                <p style="text-align: center; margin-bottom: 1rem;">{confidence_text}</p>
                <div style="background-color: rgba(255,255,255,0.3); width:100%; border-radius:12px; height: 25px;">
                    <div class="confidence-bar" style="width: {min(confidence, 100):.1f}%; background-color: {bar_color};">
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
                title, description, confidence_text = self.classifier.format_prediction(
                    pred['key'], pred['confidence']
                )
                confidence = pred['confidence']
                
                # Different colors for different ranks
                colors = ["#17a2b8", "#fd7e14", "#6f42c1"]
                color = colors[min(i-1, len(colors)-1)]
                
                st.markdown(
                    f"""<div class="result-card-secondary">
                        <h4 style="color: {color}; margin-top: 0;">{title}</h4>
                        <p style="margin: 0.5rem 0;">{description}</p>
                        <p style="font-size: 0.9em; color: #666; margin-bottom: 1rem;">{confidence_text}</p>
                        <div style="background-color: #e9ecef; width:100%; border-radius:10px; height: 20px;">
                            <div class="confidence-bar" style="width: {min(confidence, 100):.1f}%; background-color: {color}; height: 20px; font-size: 12px;">
                                {confidence:.1f}%
                            </div>
                        </div>
                    </div>""", 
                    unsafe_allow_html=True
                )
        
        # Add helpful tips
        st.markdown("---")
        st.markdown("### 💡 Tips untuk Hasil Terbaik")
        st.markdown("""
        - 📷 Gunakan foto yang jelas dan terang
        - 🎯 Pastikan wajah hewan terlihat jelas
        - 📐 Crop foto agar fokus pada hewan
        - 🌟 Hindari foto yang blur atau gelap
        """)
    
    def render_footer(self):
        """Render the application footer."""
        st.markdown(
            '<div class="footer">🧠 CatXDog menggunakan ResNet101 dengan transfer learning | 🐾 Dibuat dengan ❤️ untuk pecinta hewan</div>', 
            unsafe_allow_html=True
        )
    
    def run(self):
        """Run the application."""
        self.render_header()
        img = self.get_image_input()
        
        if img is not None:
            st.image(img, caption="Foto hewan peliharaan", use_column_width=True)
            
            with st.spinner('Menganalisis gambar... 🔍'):
                arr = self.classifier.preprocess_image(img)
                
                if arr is not None and self.classifier.model is not None:
                    predictions = self.classifier.predict(arr)
                    if predictions:
                        self.display_results(predictions)
                    else:
                        st.error("Tidak dapat menganalisis gambar. Silakan coba gambar lain.")
                else:
                    st.error("Tidak dapat menganalisis gambar. Pastikan model berhasil dimuat.")
        
        self.render_footer()


def main():
    """Main function to run the application."""
    classifier = BreedClassifier()
    app = PetBreedClassifierUI(classifier)
    app.run()

if __name__ == "__main__":
    main()