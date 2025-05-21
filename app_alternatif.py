# Set page config must be the first Streamlit command
import streamlit as st
st.set_page_config(
    page_title="CatXDog",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
import gdown
import base64

class BreedClassifier:
    """Class to handle pet breed classification using a pre-trained TensorFlow model."""
    MODEL_PATH = "pet_breed_classifier_final.h5"
    
    @staticmethod
    @st.cache_resource
    def load_model_cached(model_path):
        """Load model with caching to avoid reloading on each rerun."""
        return tf.keras.models.load_model(model_path)
    
    @staticmethod
    def download_model(model_path):
"""Download model from Google Drive with virus scan bypass and validation."""
        file_id = "1GDOwEq3pHwy1ftngOzCQCllXNtawsueI"
        url = f"https://drive.usercontent.google.com/download?id=1GDOwEq3pHwy1ftngOzCQCllXNtawsueI&confirm=t&uuid=47793b2f-79f0-41ea-805d-13d7cc36f792"  # Add &confirm=t
        
        for attempt in range(max_retries):
            try:
                st.info(f"Mengunduh model (usaha ke-{attempt+1})...")
                gdown.download(url, model_path, quiet=False)
                
                # Validasi ukuran file (minimal 1MB)
                if os.path.exists(model_path):
                    file_size = os.path.getsize(model_path)
                    if file_size < 1024 * 1024:  # Jika kurang dari 1MB
                        os.remove(model_path)
                        raise Exception("File terlalu kecil, kemungkinan korup atau akses ditolak.")
                    return True
                else:
                    raise Exception("File tidak ditemukan setelah download.")
            
            except Exception as e:
                if attempt < max_retries - 1:
                    st.warning(f"Download gagal: {e}, mencoba lagi dalam 5 detik...")
                    time.sleep(5)
                else:
                    st.error(f"Gagal mengunduh model setelah {max_retries} usaha: {e}")
                    return False

    def __init__(self):
        """Initialize the classifier with model and breed information."""
        self.class_descriptions = self._build_class_descriptions()
        self.cat_breeds = {k for k, d in self.class_descriptions.items() if d.startswith("Kucing")}
        self.model = None
        self.class_names = list(self.class_descriptions.keys())
        self._load_model()
    
    def _build_class_descriptions(self):
        """Build dictionary of class descriptions."""
        return {
            "american_bulldog": "Anjing ras American Bulldog yang berotot dan kuat, biasanya memiliki rahang besar dan tubuh kekar.",
            "american_pit_bull_terrier": "Anjing American Pit Bull Terrier, dikenal karena tubuh atletis dan karakter yang berani.",
            "abyssinian": "Kucing Abyssinian dengan bulu pendek dan berwarna hangat, dikenal karena kecerdasan dan keaktifannya.",
            "basset_hound": "Anjing Basset Hound bertelinga panjang dan tubuh pendek, dikenal karena indra penciumannya yang tajam.",
            "beagle": "Anjing kecil hingga sedang, Beagle, dikenal karena hidung tajam dan sifat ramah.",
            "bengal": "Kucing Bengal dengan pola bulu mirip macan tutul dan karakter energik.",
            "great_pyrenees": "Anjing Great Pyrenees berukuran besar dengan bulu tebal putih, biasanya anjing penjaga ternak.",
            "english_setter": "Anjing English Setter yang anggun dan berbulu panjang, cocok untuk perburuan atau keluarga.",
            "german_shorthaired": "German Shorthaired Pointer adalah anjing berburu serbaguna dengan tubuh atletis dan cerdas.",
            "english_cocker_spaniel": "Anjing English Cocker Spaniel dengan telinga panjang dan karakter ceria.",
            "boxer": "Anjing Boxer berotot dan aktif, terkenal dengan ekspresi wajah yang khas dan ramah anak.",
            "bombay": "Kucing Bombay dengan bulu hitam pekat dan mata kuning keemasan yang mencolok.",
            "birman": "Kucing Birman berbulu panjang, biasanya krem dengan ujung kaki putih dan mata biru cerah.",
            "egyptian_mau": "Kucing Egyptian Mau dengan bulu berbintik alami dan gerakan cepat.",
            "chihuahua": "Anjing Chihuahua berukuran sangat kecil, berani, dan suka menempel dengan pemiliknya.",
            "british_shorthair": "Kucing British Shorthair dengan tubuh bulat, bulu tebal, dan sifat tenang.",
            "havanese": "Anjing Havanese kecil berbulu panjang dan lembut, sangat cocok sebagai hewan peliharaan keluarga.",
            "japanese_chin": "Anjing kecil Japanese Chin dengan wajah datar dan ekspresi lucu.",
            "keeshond": "Keeshond adalah anjing berbulu tebal, dikenal karena senyuman khas dan kepribadian ceria.",
            "newfoundland": "Newfoundland adalah anjing besar, kuat, dan suka air, terkenal karena sifat lembutnya.",
            "miniature_pinscher": "Miniature Pinscher adalah anjing kecil, gesit, dan penuh percaya diri.",
            "pomeranian": "Anjing kecil berbulu tebal dan wajah seperti rubah, dikenal karena energi tinggi.",
            "pug": "Anjing Pug dengan wajah datar, tubuh kecil, dan kepribadian menggemaskan.",
            "persian": "Kucing Persia dengan wajah datar dan bulu panjang, sering dipelihara karena keanggunannya.",
            "leonberger": "Anjing besar dan berbulu lebat, Leonberger dikenal karena kekuatan dan kesetiaannya.",
            "maine_coon": "Kucing Maine Coon adalah salah satu ras terbesar, berbulu tebal dan sangat ramah.",
            "saint_bernard": "Anjing Saint Bernard yang besar dan lembut, sering dikaitkan dengan misi penyelamatan di pegunungan.",
            "ragdoll": "Kucing Ragdoll dikenal karena tubuhnya yang lemas saat digendong dan sifat yang sangat jinak.",
            "russian_blue": "Kucing Russian Blue dengan bulu abu-abu kebiruan dan mata hijau cerah.",
            "scottish_terrier": "Scottish Terrier atau Scottie dikenal karena tubuh kecil dan karakter keras kepala.",
            "shiba_inu": "Anjing kecil asal Jepang, Shiba Inu, dikenal dengan wajah rubah dan kepribadian mandiri.",
            "samoyed": "Anjing Samoyed berbulu putih lebat dan senyum khas, sangat ramah dan energik.",
            "siamese": "Kucing Siamese berbadan ramping, bermata biru, dan sangat vokal.",
            "yorkshire_terrier": "Yorkshire Terrier kecil dan elegan, sering dihias dengan pita di atas kepala.",
            "staffordshire_bull_terrier": "Staffordshire Bull Terrier adalah anjing kuat namun penuh kasih, cocok untuk keluarga.",
            "wheaten_terrier": "Soft Coated Wheaten Terrier memiliki bulu lembut seperti gandum dan kepribadian bersahabat.",
            "sphynx": "Kucing Sphynx tidak berbulu dengan kulit keriput dan kepribadian penuh rasa ingin tahu.",
            # Special classes
            "not_catxdog": "Gambar tidak terdeteksi sebagai anjing atau kucing. Kemungkinan adalah manusia, kartun, atau hewan lain.",
            "garfield": "Gambar dikenali menyerupai karakter kartun Garfield. Mungkin ini gambar kartun atau fan art.",
            "catdog": "Gambar terdeteksi mengandung dua jenis hewan (kucing dan anjing) dalam satu gambar atau objek hybrid seperti karakter CatDog."
        }
    
    def _load_model(self):
        """Load the TensorFlow model and class names."""
        try:
            # Check if model exists, download if not
            if not os.path.exists(self.MODEL_PATH):
                with st.spinner("Model belum ditemukan. Mengunduh model dari Google Drive..."):
                    if not self.download_model(self.MODEL_PATH):
                        st.error("Gagal mengunduh model. Aplikasi tidak dapat berjalan.")
                        return False
            
            # Load model using cache
            self.model = self.load_model_cached(self.MODEL_PATH)
            
            # Load class mapping if available
            if os.path.exists('class_mapping.csv'):
                df = pd.read_csv('class_mapping.csv')
                self.class_names = df['class_name'].tolist()
            
            return True
            
        except Exception as e:
            st.error(f"Error saat memuat model: {e}")
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
            st.error(f"Error saat memproses gambar: {e}")
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
            st.error(f"Error saat melakukan prediksi: {e}")
            return []
    
    def format_prediction(self, breed_key, confidence):
        """Format prediction output with title and description."""
        if breed_key not in self.class_descriptions:
            return "Tidak diketahui", "Tidak dapat mengenali ras hewan ini."
            
        name = breed_key.replace('_', ' ').title()
        desc = self.class_descriptions.get(breed_key, "Tidak ada deskripsi tersedia.")
        
        if breed_key in self.cat_breeds:
            title = f"Kucing dengan Ras {name}, Tingkat kepercayaan {confidence:.1f}%"
        elif breed_key in ["not_catxdog", "garfield", "catdog"]:
            title = name
        else:
            title = f"Anjing dengan Ras {name}, Tingkat kepercayaan {confidence:.1f}%"
            
        return title, desc


class PetBreedClassifierUI:
    """UI class for handling the Streamlit interface."""
    
    def __init__(self, classifier):
        """Initialize UI with a classifier."""
        self.classifier = classifier
        # self.setup_page_config()
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
    
    # @staticmethod
    # def setup_page_config():
    #     """Configure Streamlit page settings."""
    #     st.set_page_config(
    #         page_title="CatXDog",
    #         page_icon="🐾",
    #         layout="centered",
    #         initial_sidebar_state="collapsed"
    #     )
    
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
        st.markdown('Upload foto anjing atau kucing untuk mengetahui breed-nya!')
    
    def get_image_input(self):
        """Get image input from upload or camera."""
        uploaded = st.file_uploader("Upload gambar...", type=["jpg","jpeg","png"])
        camera = st.camera_input("Atau ambil foto dengan kamera")
        
        img = None
        if uploaded:
            try:
                img = Image.open(uploaded)
            except Exception as e:
                st.error(f"Error saat membuka file: {e}")
        elif camera:
            try:
                img = Image.open(camera)
            except Exception as e:
                st.error(f"Error saat mengakses kamera: {e}")
                
        return img
    
    def display_results(self, predictions):
        """Display prediction results."""
        if not predictions:
            st.warning("Tidak dapat melakukan prediksi. Silakan coba gambar lain.")
            return
            
        st.markdown("### Hasil Prediksi")
        
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
            st.markdown("### Kemungkinan Lainnya")
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
            '<div class="footer">CatXDog menggunakan ResNet101 dengan transfer learning</div>', 
            unsafe_allow_html=True
        )
    
    def run(self):
        """Run the application."""
        self.render_header()
        img = self.get_image_input()
        
        if img is not None:
            st.image(img, caption="Foto hewan peliharaan", use_column_width=True)
            
            with st.spinner('Menganalisis gambar...'):
                arr = self.classifier.preprocess_image(img)
                
                if arr is not None and self.classifier.model is not None:
                    predictions = self.classifier.predict(arr)
                    self.display_results(predictions)
                else:
                    st.error("Tidak dapat menganalisis gambar. Pastikan model sudah dimuat dengan benar.")
        
        self.render_footer()


def main():
    """Main function to run the application."""
    classifier = BreedClassifier()
    app = PetBreedClassifierUI(classifier)
    app.run()

if __name__ == "__main__":
    main()
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import os
# import cv2
# from PIL import Image
# import gdown


# class BreedClassifier:
#     """Main class for handling pet breed classification."""
    
#     MODEL_PATH = "pet_breed_classifier_final.h5"
    
#     def __init__(self):
#         """Initialize the classifier with model and class names."""
#         self.model = None
#         self.class_names = None
#         self.model_loaded = False
#         self.class_descriptions = self._build_class_descriptions()
#         self.cat_breeds = {k for k, d in self.class_descriptions.items() if d.startswith("Kucing")}
    
#     def _build_class_descriptions(self):
#         """Build dictionary of class descriptions."""
#         return {
#             "american_bulldog": "Anjing ras American Bulldog yang berotot dan kuat, biasanya memiliki rahang besar dan tubuh kekar.",
#             "american_pit_bull_terrier": "Anjing American Pit Bull Terrier, dikenal karena tubuh atletis dan karakter yang berani.",
#             "abyssinian": "Kucing Abyssinian dengan bulu pendek dan berwarna hangat, dikenal karena kecerdasan dan keaktifannya.",
#             "basset_hound": "Anjing Basset Hound bertelinga panjang dan tubuh pendek, dikenal karena indra penciumannya yang tajam.",
#             "beagle": "Anjing kecil hingga sedang, Beagle, dikenal karena hidung tajam dan sifat ramah.",
#             "bengal": "Kucing Bengal dengan pola bulu mirip macan tutul dan karakter energik.",
#             "great_pyrenees": "Anjing Great Pyrenees berukuran besar dengan bulu tebal putih, biasanya anjing penjaga ternak.",
#             "english_setter": "Anjing English Setter yang anggun dan berbulu panjang, cocok untuk perburuan atau keluarga.",
#             "german_shorthaired": "German Shorthaired Pointer adalah anjing berburu serbaguna dengan tubuh atletis dan cerdas.",
#             "english_cocker_spaniel": "Anjing English Cocker Spaniel dengan telinga panjang dan karakter ceria.",
#             "boxer": "Anjing Boxer berotot dan aktif, terkenal dengan ekspresi wajah yang khas dan ramah anak.",
#             "bombay": "Kucing Bombay dengan bulu hitam pekat dan mata kuning keemasan yang mencolok.",
#             "birman": "Kucing Birman berbulu panjang, biasanya krem dengan ujung kaki putih dan mata biru cerah.",
#             "egyptian_mau": "Kucing Egyptian Mau dengan bulu berbintik alami dan gerakan cepat.",
#             "chihuahua": "Anjing Chihuahua berukuran sangat kecil, berani, dan suka menempel dengan pemiliknya.",
#             "british_shorthair": "Kucing British Shorthair dengan tubuh bulat, bulu tebal, dan sifat tenang.",
#             "havanese": "Anjing Havanese kecil berbulu panjang dan lembut, sangat cocok sebagai hewan peliharaan keluarga.",
#             "japanese_chin": "Anjing kecil Japanese Chin dengan wajah datar dan ekspresi lucu.",
#             "keeshond": "Keeshond adalah anjing berbulu tebal, dikenal karena senyuman khas dan kepribadian ceria.",
#             "newfoundland": "Newfoundland adalah anjing besar, kuat, dan suka air, terkenal karena sifat lembutnya.",
#             "miniature_pinscher": "Miniature Pinscher adalah anjing kecil, gesit, dan penuh percaya diri.",
#             "pomeranian": "Anjing kecil berbulu tebal dan wajah seperti rubah, dikenal karena energi tinggi.",
#             "pug": "Anjing Pug dengan wajah datar, tubuh kecil, dan kepribadian menggemaskan.",
#             "persian": "Kucing Persia dengan wajah datar dan bulu panjang, sering dipelihara karena keanggunannya.",
#             "leonberger": "Anjing besar dan berbulu lebat, Leonberger dikenal karena kekuatan dan kesetiaannya.",
#             "maine_coon": "Kucing Maine Coon adalah salah satu ras terbesar, berbulu tebal dan sangat ramah.",
#             "saint_bernard": "Anjing Saint Bernard yang besar dan lembut, sering dikaitkan dengan misi penyelamatan di pegunungan.",
#             "ragdoll": "Kucing Ragdoll dikenal karena tubuhnya yang lemas saat digendong dan sifat yang sangat jinak.",
#             "russian_blue": "Kucing Russian Blue dengan bulu abu-abu kebiruan dan mata hijau cerah.",
#             "scottish_terrier": "Scottish Terrier atau Scottie dikenal karena tubuh kecil dan karakter keras kepala.",
#             "shiba_inu": "Anjing kecil asal Jepang, Shiba Inu, dikenal dengan wajah rubah dan kepribadian mandiri.",
#             "samoyed": "Anjing Samoyed berbulu putih lebat dan senyum khas, sangat ramah dan energik.",
#             "siamese": "Kucing Siamese berbadan ramping, bermata biru, dan sangat vokal.",
#             "yorkshire_terrier": "Yorkshire Terrier kecil dan elegan, sering dihias dengan pita di atas kepala.",
#             "staffordshire_bull_terrier": "Staffordshire Bull Terrier adalah anjing kuat namun penuh kasih, cocok untuk keluarga.",
#             "wheaten_terrier": "Soft Coated Wheaten Terrier memiliki bulu lembut seperti gandum dan kepribadian bersahabat.",
#             "sphynx": "Kucing Sphynx tidak berbulu dengan kulit keriput dan kepribadian penuh rasa ingin tahu.",
#             # Kelas khusus
#             "not_catxdog": "Gambar tidak terdeteksi sebagai anjing atau kucing. Kemungkinan adalah manusia, kartun, atau hewan lain.",
#             "garfield": "Gambar dikenali menyerupai karakter kartun Garfield. Mungkin ini gambar kartun atau fan art.",
#             "catdog": "Gambar terdeteksi mengandung dua jenis hewan (kucing dan anjing) dalam satu gambar atau objek hybrid seperti karakter CatDog."
#         }
    
#     @staticmethod
#     def download_model(model_path):
#         """Download model from Google Drive if not present."""
#         file_id = "1GDOwEq3pHwy1ftngOzCQCllXNtawsueI"
#         url = f"https://drive.google.com/uc?id={file_id}"
#         gdown.download(url, model_path, quiet=False)
    
#     def load_model(self):
#         """Load the TensorFlow model and class names."""
#         try:
#             # Download model if it doesn't exist
#             if not os.path.exists(self.MODEL_PATH):
#                 st.write("Model belum ditemukan. Mengunduh model dari Google Drive...")
#                 self.download_model(self.MODEL_PATH)
                
#             self.model = tf.keras.models.load_model(self.MODEL_PATH)
#             if os.path.exists('class_mapping.csv'):
#                 df = pd.read_csv('class_mapping.csv')
#                 self.class_names = df['class_name'].tolist()
#             else:
#                 self.class_names = list(self.class_descriptions.keys())
            
#             self.model_loaded = True
#             return True
#         except Exception as e:
#             st.error(f"Error loading model: {e}")
#             self.model_loaded = False
#             return False
    
#     def preprocess_image(self, image, img_size=(224, 224)):
#         """Preprocess image for model input."""
#         if isinstance(image, np.ndarray):
#             image = cv2.resize(image, img_size)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         else:
#             image = image.convert("RGB")  
#             image = image.resize(img_size)
#             image = np.array(image)
#         arr = np.expand_dims(image, axis=0)
#         return tf.keras.applications.resnet.preprocess_input(arr)
    
#     def predict(self, image_array):
#         """Predict breed from preprocessed image array."""
#         if not self.model_loaded:
#             if not self.load_model():
#                 return None
        
#         try:
#             preds = self.model.predict(image_array)
#             idxs = np.argsort(preds[0])[-3:][::-1]
#             results = []
#             for i in idxs:
#                 key = self.class_names[i]
#                 prob = float(preds[0][i]) * 100
#                 results.append({"key": key, "confidence": prob})
#             return results
#         except Exception as e:
#             st.error(f"Error during prediction: {e}")
#             return None
    
#     def format_prediction(self, breed_key, confidence):
#         """Format prediction output with title and description."""
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
#         self.setup_page_config()
#         self.apply_custom_css()
#         self.camera_status = "idle"  # Track camera status: idle, initializing, ready, error
    
#     @staticmethod
#     def setup_page_config():
#         """Configure Streamlit page settings."""
#         st.set_page_config(
#             page_title="Pet Breed Classifier",
#             page_icon="🐾",
#             layout="centered",
#             initial_sidebar_state="collapsed"
#         )
    
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
#         .camera-tips { background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
#         .input-section { margin-bottom: 20px; padding: 15px; border-radius: 10px; background-color: #f9f9f9; }
#         </style>
#         """, unsafe_allow_html=True)
    
#     def render_header(self):
#         """Render the application header."""
#         st.markdown('<p class="custom-header">🐾 Pet Breed Classifier</p>', unsafe_allow_html=True)
#         st.markdown('Upload foto anjing atau kucing untuk mengetahui breed-nya!')
        
#         # Add troubleshooting expander
#         with st.expander("ℹ️ Petunjuk & Bantuan"):
#             st.markdown("""
#             ### Cara Penggunaan
#             1. **Upload Gambar**: Pilih file gambar dari perangkat Anda dengan format JPG, JPEG, atau PNG.
#             2. **Gunakan Kamera**: Berikan izin akses kamera dan ambil foto langsung.
#             3. **Hasil Analisis**: Sistem akan menampilkan breed yang paling sesuai beserta tingkat kepercayaan.
            
#             ### Masalah Umum
#             - **Kamera Loading Lama**: 
#                 - Pastikan browser memiliki izin akses kamera
#                 - Coba refresh halaman (F5)
#                 - Gunakan browser terbaru (Chrome/Firefox)
#                 - Nonaktifkan ekstensi browser yang mungkin memblokir kamera
#             - **Hasil Tidak Akurat**: 
#                 - Pastikan foto memiliki pencahayaan yang baik
#                 - Hindari background yang terlalu ramai
#                 - Ambil foto dengan jelas menampilkan muka hewan
#             """)
    
#     def check_camera_compatibility(self):
#         """Display information about camera compatibility."""
#         # Using session state to avoid repetitive checks
#         if 'camera_checked' not in st.session_state:
#             st.session_state.camera_checked = True
            
#             # Add camera compatibility info
#             st.markdown("""
#             <div class='camera-tips'>
#                 <strong>Tips Kamera:</strong><br>
#                 - Izinkan akses kamera pada browser<br>
#                 - Pastikan perangkat Anda memiliki kamera<br>
#                 - Jika kamera loading lama, refresh halaman atau gunakan upload gambar
#             </div>
#             """, unsafe_allow_html=True)

#     def get_image_input(self):
#         """Get image input from upload or camera with improved user feedback."""
#         self.check_camera_compatibility()
        
#         # Create tabs for input options
#         tab1, tab2 = st.tabs(["📁 Upload Gambar", "📷 Gunakan Kamera"])
        
#         img = None
        
#         with tab1:
#             st.markdown("<div class='input-section'>", unsafe_allow_html=True)
#             uploaded = st.file_uploader("Pilih file gambar...", type=["jpg","jpeg","png"])
#             if uploaded:
#                 try:
#                     img = Image.open(uploaded)
#                     st.success("✅ Gambar berhasil diunggah!")
#                 except Exception as e:
#                     st.error(f"Gagal membuka gambar: {e}")
#             st.markdown("</div>", unsafe_allow_html=True)
        
#         with tab2:
#             st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            
#             # Camera status indicator
#             status_placeholder = st.empty()
            
#             # Provide guidance for camera usage
#             st.info("🔍 Pastikan wajah hewan terlihat jelas dan dalam pencahayaan yang baik")
            
#             # Camera placeholder for potential reloading
#             camera_placeholder = st.empty()
#             camera = camera_placeholder.camera_input("Ambil foto", key="pet_camera")
            
#             if camera is None:
#                 status_placeholder.warning("⏳ Menunggu kamera aktif atau akses diberikan...")
#                 self.camera_status = "initializing"
#             else:
#                 status_placeholder.success("✅ Kamera siap digunakan!")
#                 self.camera_status = "ready"
#                 try:
#                     img = Image.open(camera)
                    
#                     # Add option to retake
#                     if st.button("🔄 Ambil Ulang", key="retake"):
#                         st.experimental_rerun()
#                 except Exception as e:
#                     status_placeholder.error(f"⚠️ Gagal memproses foto: {e}")
#                     self.camera_status = "error"
            
#             st.markdown("</div>", unsafe_allow_html=True)
            
#             # Add troubleshooting help if camera is still initializing after 5 seconds
#             if self.camera_status == "initializing":
#                 st.warning("""
#                 **Kamera masih loading?**
#                 - Coba refresh halaman
#                 - Periksa izin kamera di browser
#                 - Coba gunakan upload gambar sebagai alternatif
#                 """)
        
#         return img
    
#     def get_image_input(self):
#         """Get image input from upload or camera with improved user feedback."""
#         # Create two columns for input options
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("### Upload Gambar")
#             uploaded = st.file_uploader("Pilih file...", type=["jpg","jpeg","png"])
        
#         with col2:
#             st.markdown("### Gunakan Kamera")
#             # Add information about camera access
#             st.info("👉 Jika kamera loading lama, coba refresh browser atau izinkan akses kamera pada pengaturan browser.")
#             camera_placeholder = st.empty()
#             camera = camera_placeholder.camera_input("Ambil foto", key="pet_camera")
            
#             # Add a button to reset camera if needed
#             if camera is not None:
#                 if st.button("Ambil Ulang", key="reset_camera"):
#                     camera_placeholder.empty()
#                     camera = camera_placeholder.camera_input("Ambil foto", key="pet_camera_reset")
        
#         # Process the image
#         img = None
#         if uploaded:
#             try:
#                 img = Image.open(uploaded)
#                 st.success("✅ Gambar berhasil diunggah!")
#             except Exception as e:
#                 st.error(f"Gagal membuka gambar: {e}")
#         elif camera:
#             try:
#                 img = Image.open(camera)
#                 st.success("✅ Foto berhasil diambil!")
#             except Exception as e:
#                 st.error(f"Gagal memproses foto: {e}")
                
#         return img
    
#     def display_results(self, predictions):
#         """Display prediction results."""
#         st.markdown("### Hasil Prediksi")
#         # Top1
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
        
#         # Others
#         st.markdown("### Kemungkinan Lainnya")
#         for p in predictions[1:]:
#             title, desc = self.classifier.format_prediction(p['key'], p['confidence'])
#             confidence = p['confidence']
#             color = "#2196F3" if p == predictions[1] else "#FF9800"
#             st.markdown(
#                 f"""<div class="result-card">
#                     <h4>{title}</h4>
#                     <p>{desc}</p>
#                     <div style="background-color: #e0e0e0; width:100%; border-radius:5px;">
#                         <div class="confidence-bar" style="width: {confidence:.1f}%; background-color: {color};">{confidence:.1f}%</div>
#                     </div>
#                 </div>""", 
#                 unsafe_allow_html=True
#             )
    
#     def render_footer(self):
#         """Render the application footer."""
#         st.markdown(
#             '<div class="footer">Pet Breed Classifier menggunakan ResNet101 dengan transfer learning</div>', 
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
#                 if self.classifier.model:
#                     predictions = self.classifier.predict(arr)
#                     self.display_results(predictions)
#                 else:
#                     st.error("Model tidak dapat dimuat. Pastikan file model ada di direktori yang sama.")
        
#         self.render_footer()


# def main():
#     """Application entry point."""
#     # Create classifier
#     classifier = BreedClassifier()
    
#     # Create and run UI
#     app = PetBreedClassifierUI(classifier)
#     app.run()


# if __name__ == "__main__":
#     main()