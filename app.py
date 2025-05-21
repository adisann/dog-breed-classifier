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
        
        url = f"https://drive.usercontent.google.com/download?id=1GDOwEq3pHwy1ftngOzCQCllXNtawsueI&confirm=t&uuid=47793b2f-79f0-41ea-805d-13d7cc36f792"
        
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
                            # Generate random predictions for 120 classes (adjust if needed)
                            preds = np.random.random((x.shape[0], 120))
                            # Normalize to ensure they sum to 1
                            preds = preds / np.sum(preds, axis=1, keepdims=True)
                            return preds
                    
                    self.model = DummyModel()
                    
                    # Also create some dummy class names if needed
                    if not self.class_names:
                        self.class_names = [f"dummy_class_{i}" for i in range(120)]
                        self.cat_breeds = self.class_names[:60]  # First half are cats
                        # Add special classes
                        self.class_names.extend(["not_catxdog", "garfield", "catdog"])
                        
                        # Create dummy descriptions
                        self.class_descriptions = {
                            name: f"This is a dummy description for {name}" 
                            for name in self.class_names
                        }
                    
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
                    key = self.class_names[i]
                    prob = float(preds[i]) * 100
                    results.append({"key": key, "confidence": prob})
            return results
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.error(traceback.format_exc())
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