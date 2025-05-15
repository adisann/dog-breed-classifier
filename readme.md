## Deskripsi Proyek

**CatXDog** adalah aplikasi berbasis web yang dibangun dengan Streamlit dan TensorFlow untuk mengidentifikasi ras anjing atau kucing dari foto. Aplikasi ini memanfaatkan model ResNet101 yang telah dilatih (transfer learning) untuk mengklasifikasikan gambar ke dalam berbagai ras hewan peliharaan. Fitur utama:

* Deteksi otomatis dan pengunduhan model dari Google Drive.
* Preprocessing gambar dengan `tf.keras.applications.resnet.preprocess_input`.
* Menampilkan tiga prediksi teratas beserta tingkat kepercayaan dan deskripsi ras.
* Antarmuka yang responsif dan mudah digunakan melalui browser.

## Struktur Direktori

```plaintext
├── app.py                   # Skrip utama Streamlit
├── pet_breed_classifier_final.h5  # Model TensorFlow (H5)
├── class_mapping.csv        # (Opsional) Pemetaan nama kelas
├── requirements.txt         # Daftar dependensi Python
└── README.md                # Dokumentasi proyek
```

## Cara Instalasi

1. **Clone repositori**

   ```bash
   git clone https://github.com/username/pet-breed-classifier.git
   cd pet-breed-classifier
   ```

2. **Buat virtual environment (opsional tapi disarankan)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\\Scripts\\activate  # Windows
   ```

3. **Instal dependensi**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Cara Menjalankan Program

Jalankan Streamlit dari direktori proyek:

```bash
streamlit run app.py
```

Buka browser di `http://localhost:8501/` untuk mengakses antarmuka.

## Pengaturan Model

* Program akan mengecek keberadaan file `pet_breed_classifier_final.h5` di folder proyek.
* Jika belum ada, aplikasi akan otomatis mengunduh dari Google Drive berdasarkan *file ID* yang telah dikonfigurasi.
* Untuk mengganti sumber model, ubah nilai `file_id` di fungsi `download_model()`.

## Penggunaan

1. Unggah gambar hewan peliharaan (`jpg`, `jpeg`, `png`) melalui tombol "Upload".
2. Atau ambil foto langsung menggunakan "Camera Input".
3. Tunggu proses analisis hingga prediksi muncul.
4. Lihat hasil prediksi utama dan dua alternatif berikutnya beserta persentase kepercayaan.

## Informasi Tambahan

* **Cache**: Model dan file pemetaan (jika ada) di-cache untuk mempercepat loading.
* **Desain UI**: Kustom CSS disematkan untuk tampilan yang lebih menarik dan konsisten.
* **Ekstensi**: Anda dapat menambahkan ras tambahan dengan memperbarui `class_descriptions` dan `class_mapping.csv`.

## Lisensi

Lisensi MIT © 2025 \[Nama Anda]
