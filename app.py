import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
import gdown
import base64
import streamlit as st
import tensorflow as tf
import gdown
import os

class BreedClassifier:
    MODEL_PATH = "pet_breed_classifier_final.h5"

    @st.cache_resource
    def _load_model_with_cache(_self):
        """Load model with caching."""
        if not os.path.exists(_self.MODEL_PATH):
            st.write("Mengunduh model dari Google Drive...")
            file_id = "1GDOwEq3pHwy1ftngOzCQCllXNtawsueI"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, _self.MODEL_PATH, quiet=False)
        return tf.keras.models.load_model(_self.MODEL_PATH)

    def __init__(self):
        self.model = self._load_model_with_cache()
        self.class_names = None
        self.class_descriptions = self._build_class_descriptions()
        self.cat_breeds = {k for k, d in self.class_descriptions.items() if d.startswith("Kucing")}
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
            # Kelas khusus
            "not_catxdog": "Gambar tidak terdeteksi sebagai anjing atau kucing. Kemungkinan adalah manusia, kartun, atau hewan lain.",
            "garfield": "Gambar dikenali menyerupai karakter kartun Garfield. Mungkin ini gambar kartun atau fan art.",
            "catdog": "Gambar terdeteksi mengandung dua jenis hewan (kucing dan anjing) dalam satu gambar atau objek hybrid seperti karakter CatDog."
        }
    
    @staticmethod
    def download_model(model_path):
        """Download model from Google Drive if not present."""
        file_id = "1GDOwEq3pHwy1ftngOzCQCllXNtawsueI"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    
    def _load_model(self):
        """Load the TensorFlow model and class names."""
        try:
            # Download model if it doesn't exist
            if not os.path.exists(self.MODEL_PATH):
                st.write("Model belum ditemukan. Mengunduh model dari Google Drive...")
                self.download_model(self.MODEL_PATH)
                
            self.model = tf.keras.models.load_model(self.MODEL_PATH)
            if os.path.exists('class_mapping.csv'):
                df = pd.read_csv('class_mapping.csv')
                self.class_names = df['class_name'].tolist()
            else:
                self.class_names = list(self.class_descriptions.keys())
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    def preprocess_image(self, image, img_size=(224, 224)):
        """Preprocess image for model input."""
        if isinstance(image, np.ndarray):
            image = cv2.resize(image, img_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image.convert("RGB")  
            image = image.resize(img_size)
            image = np.array(image)
        arr = np.expand_dims(image, axis=0)
        return tf.keras.applications.resnet.preprocess_input(arr)
    
    def predict(self, image_array):
        """Predict breed from preprocessed image array."""
        preds = self.model.predict(image_array)
        idxs = np.argsort(preds[0])[-3:][::-1]
        results = []
        for i in idxs:
            key = self.class_names[i]
            prob = float(preds[0][i]) * 100
            results.append({"key": key, "confidence": prob})
        return results
    
    def format_prediction(self, breed_key, confidence):
        """Format prediction output with title and description."""
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
        self.setup_page_config()
        # Path gambar
        img_path = "Logo.jpg"
        img_base64 = self.get_base64_image(img_path)
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/jpg;base64,{img_base64}" width="200"/>
            </div>
            """,
            unsafe_allow_html=True
        )
        self.apply_custom_css()
    def get_base64_image(self, path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    
    @staticmethod
    def setup_page_config():
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="CatXDog",
            page_icon="🐾",
            layout="centered",
            initial_sidebar_state="collapsed"
        )
    
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
            img = Image.open(uploaded)
        elif camera:
            img = Image.open(camera)
        return img
    
    def display_results(self, predictions):
        """Display prediction results."""
        st.markdown("### Hasil Prediksi")
        # Top1
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
        
        # Others
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
                if self.classifier.model:
                    predictions = self.classifier.predict(arr)
                    self.display_results(predictions)
                else:
                    st.error("Model tidak dapat dimuat. Pastikan file model ada di direktori yang sama.")
        
        self.render_footer()


def main():
    classifier = BreedClassifier()
    app = PetBreedClassifierUI(classifier)
    app.run()

if __name__ == "__main__":
    main()