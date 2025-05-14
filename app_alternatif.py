import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
import os
import gdown

import os
import gdown
import streamlit as st

MODEL_PATH = "pet_breed_classifier_final.h5" 

# Fungsi untuk mendownload model jika belum ada
def download_model():
    # Ganti dengan file ID model kamu yang ada di Google Drive
    file_id = "1GDOwEq3pHwy1ftngOzCQCllXNtawsueI"  # Ganti dengan ID file yang sesuai
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)

# Cek apakah model sudah ada, jika belum, download
if not os.path.exists(MODEL_PATH):
    st.write("Model belum ditemukan. Mengunduh model dari Google Drive...")
    download_model()


    
# Mapping kelas ke deskripsi
def build_class_descriptions():
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

# Build sets
class_descriptions = build_class_descriptions()
cat_breeds = {k for k, d in class_descriptions.items() if d.startswith("Kucing")}

# Format prediction output
def format_prediction(breed_key, confidence):
    name = breed_key.replace('_', ' ').title()
    desc = class_descriptions.get(breed_key, "Tidak ada deskripsi tersedia.")
    if breed_key in cat_breeds:
        title = f"Kucing dengan Ras {name}, Tingkat kepercayaan {confidence:.1f}%"
    elif breed_key in ["not_catxdog", "garfield", "catdog"]:
        title = name
    else:
        title = f"Anjing dengan Ras {name}, Tingkat kepercayaan {confidence:.1f}%"
    return title, desc

# Set page config
st.set_page_config(
    page_title="Pet Breed Classifier",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('pet_breed_classifier_final.h5')
        if os.path.exists('class_mapping.csv'):
            df = pd.read_csv('class_mapping.csv')
            class_names = df['class_name'].tolist()
        else:
            class_names = list(class_descriptions.keys())
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Preprocessing image
def preprocess_image(image, img_size=(224, 224)):
    if isinstance(image, np.ndarray):
        image = cv2.resize(image, img_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image.convert("RGB")  
        image = image.resize(img_size)
        image = np.array(image)
    arr = np.expand_dims(image, axis=0)
    return tf.keras.applications.resnet.preprocess_input(arr)


# Predict breed
def predict_breed(model, image_array, class_names):
    preds = model.predict(image_array)
    idxs = np.argsort(preds[0])[-3:][::-1]
    results = []
    for i in idxs:
        key = class_names[i]
        prob = float(preds[0][i]) * 100
        results.append({"key": key, "confidence": prob})
    return results

# Main UI
def main():
    model, class_names = load_model()

    # Custom CSS
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

    st.markdown('<p class="custom-header">🐾 Pet Breed Classifier</p>', unsafe_allow_html=True)
    st.markdown('Upload foto anjing atau kucing untuk mengetahui breed-nya!')

    uploaded = st.file_uploader("Upload gambar...", type=["jpg","jpeg","png"])
    camera = st.camera_input("Atau ambil foto dengan kamera")
    img = None
    if uploaded: img = Image.open(uploaded)
    elif camera: img = Image.open(camera)

    if img is not None:
        st.image(img, caption="Foto hewan peliharaan", use_column_width=True)
        with st.spinner('Menganalisis gambar...'):
            arr = preprocess_image(img)
            if model:
                preds = predict_breed(model, arr, class_names)
                st.markdown("### Hasil Prediksi")
                # Top1
                title, desc = format_prediction(preds[0]['key'], preds[0]['confidence'])
                st.markdown(f"<div class=\"result-card\">\n<h3>🏆 {title}</h3>\n<p>{desc}</p>\n<div style=\"background-color: #e0e0e0; width:100%; border-radius:5px;\">\n<div class=\"confidence-bar\" style=\"width: {preds[0]['confidence']:.1f}%\">{preds[0]['confidence']:.1f}%</div>\n</div>\n</div>", unsafe_allow_html=True)
                # Others
                st.markdown("### Kemungkinan Lainnya")
                for p in preds[1:]:
                    title, desc = format_prediction(p['key'], p['confidence'])
                    st.markdown(f"<div class=\"result-card\">\n<h4>{title}</h4>\n<p>{desc}</p>\n<div style=\"background-color: #e0e0e0; width:100%; border-radius:5px;\">\n<div class=\"confidence-bar\" style=\"width: {p['confidence']:.1f}%\">{p['confidence']:.1f}%</div>\n</div>\n</div>", unsafe_allow_html=True)
            else:
                st.error("Model tidak dapat dimuat. Pastikan file model ada di direktori yang sama.")

    st.markdown('<div class="footer">Pet Breed Classifier menggunakan ResNet101 dengan transfer learning</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
