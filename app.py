import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
import io

class_descriptions = {
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


# Set page config - agar lebih mobile friendly
st.set_page_config(
    page_title="Pet Breed Classifier",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Fungsi untuk load model
@st.cache_resource
def load_model():
    """Load model H5 dan mapping kelas"""
    try:
        # Load model terlatih
        model = tf.keras.models.load_model('pet_breed_classifier_final.h5')
        
        # Load mapping kelas jika ada
        if os.path.exists('class_mapping.csv'):
            class_df = pd.read_csv('class_mapping.csv')
            class_names = class_df['class_name'].tolist()
        else:
            # Atau definisikan class names secara manual jika tidak ada file
            class_names = ["Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair", 
                          "Beagle", "Boxer", "Chihuahua", "German_Shepherd", "Labrador"]
            # Sesuaikan dengan kelas dataset Anda
        
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Fungsi untuk preprocessing gambar
def preprocess_image(image, img_size=(224, 224)):
    """Preprocess gambar untuk model ResNet101"""
    # Resize
    if isinstance(image, np.ndarray):
        # Jika input adalah array (dari cv2)
        image = cv2.resize(image, img_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    else:
        # Jika input adalah PIL Image
        image = image.resize(img_size)
        image = np.array(image)
    
    # Convert ke format yang dibutuhkan model
    image_array = np.expand_dims(image, axis=0)
    # Preprocessing ResNet101
    image_array = tf.keras.applications.resnet.preprocess_input(image_array)
    return image_array

# Fungsi untuk memprediksi breed
def predict_breed(model, image_array, class_names):
    """Prediksi breed dari gambar yang sudah dipreprocess"""
    # Prediksi
    predictions = model.predict(image_array)
    
    # Ambil top 3 prediksi
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_classes = [class_names[i] for i in top_3_idx]
    top_3_probs = [predictions[0][i] for i in top_3_idx]
    
    results = [{
        "breed": top_3_classes[i].replace("_", " "),
        "confidence": float(top_3_probs[i]) * 100  # konversi ke persen
    } for i in range(len(top_3_classes))]
    
    return results

# UI Utama
def main():
    # Load model dan class names
    model, class_names = load_model()
    
    # CSS untuk mobile-friendly UI
    st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        max-width: 100%;
    }
    .custom-header {
        font-size: 2rem;
        text-align: center;
        margin-bottom: 1rem;
        color: #4CAF50;
    }
    .upload-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 1rem;
    }
    .confidence-bar {
        height: 20px;
        border-radius: 5px;
    }
    .result-card {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        background-color: #f1f1f1;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        font-size: 0.8rem;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="custom-header">🐾 Pet Breed Classifier</p>', unsafe_allow_html=True)
    
    # Subheader
    st.markdown('Upload foto anjing atau kucing untuk mengetahui breed-nya!')
    
    # File uploader
    uploaded_file = st.file_uploader("Upload gambar...", type=["jpg", "jpeg", "png"])
    
    # Atau ambil foto dengan kamera (untuk mobile)
    camera_photo = st.camera_input("Atau ambil foto dengan kamera")
    
    # Pilih sumber gambar
    image = None
    
    if uploaded_file is not None:
        # Untuk file yang diupload
        image = Image.open(uploaded_file)
    elif camera_photo is not None:
        # Untuk foto dari kamera
        image = Image.open(camera_photo)
    
    # Proses prediksi jika ada gambar
    if image is not None:
        # Tampilkan gambar yang diupload/diambil
        st.image(image, caption="Foto hewan peliharaan", use_column_width=True)
        
        with st.spinner('Menganalisis gambar...'):
            # Preprocess gambar
            processed_image = preprocess_image(image)
            
            # Prediksi breed
            if model is not None:
                results = predict_breed(model, processed_image, class_names)
                
                # Tampilkan hasil prediksi
                st.markdown("### Hasil Prediksi")
                
                # Top 1 result dengan card khusus
                st.markdown(f"""
                <div class="result-card">
                    <h3>🏆 {results[0]['breed']}</h3>
                    <div style="background-color: #e0e0e0; width: 100%; border-radius: 5px;">
                        <div class="confidence-bar" style="width: {results[0]['confidence']}%; background-color: #4CAF50; text-align: center; color: white;">
                            {results[0]['confidence']:.1f}%
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Hasil prediksi tambahan
                st.markdown("### Kemungkinan Lainnya")
                
                # Prediksi lainnya
                for result in results[1:]:
                    st.markdown(f"""
                    <div class="result-card">
                        <h4>{result['breed']}</h4>
                        <div style="background-color: #e0e0e0; width: 100%; border-radius: 5px;">
                            <div class="confidence-bar" style="width: {result['confidence']}%; background-color: #2196F3; text-align: center; color: white;">
                                {result['confidence']:.1f}%
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Model tidak dapat dimuat. Pastikan file model ada di direktori yang sama.")
    
    # Footer
    st.markdown('<div class="footer">Pet Breed Classifier menggunakan ResNet101 dengan transfer learning</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()