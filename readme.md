[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)

## CatXDog 🐾

**CatXDog** adalah aplikasi web berbasis **Streamlit** & **TensorFlow** yang mampu mengidentifikasi ras kucing dan anjing hanya dengan mengunggah foto. Aplikasi ini memanfaatkan model **ResNet101** melalui teknik transfer learning untuk memberikan **tiga prediksi teratas**, beserta tingkat kepercayaan (confidence score) dan deskripsi singkat masing‑masing ras.

---

## ✨ Fitur Utama

- 🔄 **Unduh Otomatis Model**  
  Jika file `pet_breed_classifier_final.h5` tidak ditemukan di direktori, aplikasi akan mengunduhnya secara otomatis dari Google Drive.  
- 🖼️ **Preprocessing Gambar**  
  Menggunakan fungsi `tf.keras.applications.resnet.preprocess_input` untuk memastikan input sesuai kebutuhan model.  
- 🥇 **Top‑3 Prediksi**  
  Menampilkan tiga ras teratas dengan progress bar yang menunjukkan confidence score.  
- 🎨 **UI Responsif dan Konsisten**  
  Custom CSS untuk tampilan yang menarik dan mudah digunakan di berbagai perangkat.

---

## 📂 Struktur Direktori

```text
pet-breed-classifier/
│
├── app.py                         # Entry point Streamlit
├── pet_breed_classifier_final.h5  # Model hasil training
├── class_mapping.csv              # (Opsional) Pemetaan label ke nama kelas
├── requirements.txt               # Daftar dependensi Python
└── README.md                      # Dokumentasi proyek
```

---

## ⚙️ Instalasi

1. **Clone repository**  
   ```bash
   git clone https://github.com/username/pet-breed-classifier.git
   cd pet-breed-classifier
   ```

2. **(Opsional) Buat Virtual Environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows
   ```

3. **Instalasi Dependensi**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Jalankan Aplikasi**  
   ```bash
   streamlit run app.py
   ```  
   Buka browser pada: `http://localhost:8501`

---

## 🐶🐱 Daftar Kelas

### Kucing

- **abyssinian** ➔ Abyssinian  
- **bengal** ➔ Bengal  
- **bombay** ➔ Bombay  
- **birman** ➔ Birman  
- **egyptian_mau** ➔ Egyptian Mau  
- **persian** ➔ Persian  
- **british_shorthair** ➔ British Shorthair  
- **maine_coon** ➔ Maine Coon  
- **ragdoll** ➔ Ragdoll  
- **russian_blue** ➔ Russian Blue  
- **siamese** ➔ Siamese  
- **sphynx** ➔ Sphynx  

### Anjing

- **american_bulldog** ➔ American Bulldog  
- **american_pit_bull_terrier** ➔ American Pit Bull Terrier  
- **basset_hound** ➔ Basset Hound  
- **beagle** ➔ Beagle  
- **great_pyrenees** ➔ Great Pyrenees  
- **english_setter** ➔ English Setter  
- **german_shorthaired** ➔ German Shorthaired Pointer  
- **english_cocker_spaniel** ➔ English Cocker Spaniel  
- **boxer** ➔ Boxer  
- **chihuahua** ➔ Chihuahua  
- **havanese** ➔ Havanese  
- **japanese_chin** ➔ Japanese Chin  
- **keeshond** ➔ Keeshond  
- **newfoundland** ➔ Newfoundland  
- **miniature_pinscher** ➔ Miniature Pinscher  
- **pomeranian** ➔ Pomeranian  
- **pug** ➔ Pug  
- **leonberger** ➔ Leonberger  
- **saint_bernard** ➔ Saint Bernard  
- **scottish_terrier** ➔ Scottish Terrier  
- **shiba_inu** ➔ Shiba Inu  
- **samoyed** ➔ Samoyed  
- **staffordshire_bull_terrier** ➔ Staffordshire Bull Terrier  
- **wheaten_terrier** ➔ Wheaten Terrier  
- **yorkshire_terrier** ➔ Yorkshire Terrier  

### Kelas Khusus

- **not_catxdog** ➔ Bukan anjing atau kucing (manusia, kartun, hewan lain)  
- **garfield** ➔ Mirip karakter Garfield (fan art/kartun)  
- **catdog** ➔ Karakter hybrid “CatDog”  

---

## 📝 Cara Penggunaan

1. Unggah foto berformat `.jpg`, `.jpeg`, atau `.png`, atau aktifkan kamera langsung.  
2. Tunggu proses inferensi model.  
3. Lihat **Top‑3 Prediksi** beserta **confidence bar** dan deskripsi singkat setiap ras.

---

## 🔧 Konfigurasi Model

- Saat aplikasi dijalankan, `app.py` akan memeriksa keberadaan file `.h5`.  
- Untuk mengganti sumber model, ubah nilai `file_id` dalam fungsi `download_model()` di `app.py`.

---

## 🤝 Kontribusi

1. Fork repository ini.  
2. Buat branch baru untuk fitur Anda:  
   ```bash
   git checkout -b feature/XYZ
   ```  
3. Lakukan perubahan dan commit:  
   ```bash
   git commit -m "Add XYZ feature"
   ```  
4. Push ke branch Anda:  
   ```bash
   git push origin feature/XYZ
   ```  
5. Buka Pull Request ke branch `main`.

---

## 📄 Lisensi

Proyek ini dirilis di bawah lisensi **MIT License**.  
© 2025 CatXDog Team
