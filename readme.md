[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)

## CatXDog 🐾

**CatXDog** adalah aplikasi web berbasis **Streamlit** dan **TensorFlow** yang dirancang untuk mengidentifikasi ras kucing dan anjing hanya dengan mengunggah foto. Menggunakan model **ResNet101** melalui transfer learning, aplikasi ini menampilkan **tiga prediksi teratas**, disertai **tingkat kepercayaan** dan **deskripsi ringkas** setiap ras.

---

## ✨ Fitur Utama

* **Unduh Model Otomatis**
  Jika file `pet_breed_classifier_final.h5` tidak ditemukan, aplikasi akan mengunduhnya dari Google Drive secara otomatis.

* **Pra-proses Gambar**
  Mengaplikasikan `tf.keras.applications.resnet.preprocess_input` untuk menyesuaikan foto sebelum inferensi.

* **Tiga Prediksi Teratas**
  Menampilkan top-3 ras dengan progress bar yang menggambarkan confidence score.

* **Deskripsi Ras Lengkap**
  Menyajikan deskripsi singkat agar pengguna memahami karakteristik setiap ras.

* **Antarmuka Responsif**
  Custom CSS memastikan tampilan konsisten di berbagai perangkat.

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
3. **Instal Dependensi**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Jalankan Aplikasi**

   ```bash
   streamlit run app.py
   ```

   Buka browser pada `http://localhost:8501`.

---

## 🐶🐱 Daftar Kelas & Deskripsi Terbaru

| Kode                             | Nama Lengkap                         | Ukuran       | Asal             | Karakteristik Singkat                                                                                 |
| -------------------------------- | ------------------------------------ | ------------ | ---------------- | ----------------------------------------------------------------------------------------------------- |
| **american\_bulldog**            | American Bulldog                     | Besar        | Amerika Serikat  | Berotot dan kuat, setia, protektif, cocok untuk penjaga keluarga.                                     |
| **american\_pit\_bull\_terrier** | American Pit Bull Terrier            | Sedang–Besar | Amerika Serikat  | Atletis, enerjik, sangat loyal, membutuhkan sosialisasi awal dan pelatihan konsisten.                 |
| **basset\_hound**                | Basset Hound                         | Sedang       | Prancis          | Tubuh rendah, telinga panjang, penciuman tajam, ramah, namun bisa keras kepala.                       |
| **beagle**                       | Beagle                               | Sedang Kecil | Inggris          | Penciuman luar biasa, energik, ramah anak, memerlukan banyak aktivitas dan stimulasi mental.          |
| **boxer**                        | Boxer                                | Sedang–Besar | Jerman           | Penuh energi, atletis, ekspresif dengan moncong pesek, sangat penyayang dan pelindung keluarga.       |
| **great\_pyrenees**              | Great Pyrenees                       | Sangat Besar | Pegunungan Alpen | Bulu putih lebat, tenang namun waspada, cocok untuk menjaga ternak dan keluarga di iklim dingin.      |
| **chihuahua**                    | Chihuahua                            | Sangat Kecil | Meksiko          | Kepala berbentuk apel, mata besar, berani, waspada, setia, ideal sebagai pendamping di dalam ruangan. |
| **german\_shorthaired**          | German Shorthaired Pointer           | Sedang       | Jerman           | Atletis, serbaguna untuk berburu, cerdas, mudah dilatih, memerlukan banyak latihan fisik.             |
| **english\_setter**              | English Setter                       | Sedang–Besar | Inggris          | Anggun, berbulu panjang dengan bintik, ramah, lembut, cocok untuk keluarga aktif.                     |
| **english\_cocker\_spaniel**     | English Cocker Spaniel               | Sedang Kecil | Inggris          | Suka bermain, ramah, telinga panjang bergelombang, memerlukan grooming rutin.                         |
| **keeshond**                     | Keeshond                             | Sedang       | Belanda          | Bulu tebal “kupluk”, ekspresi seperti tersenyum, ramah, pintar, cocok sebagai anjing pendamping.      |
| **havanese**                     | Havanese                             | Kecil        | Kuba             | Bulu lembut panjang, ceria, penuh kasih, sosial, mudah beradaptasi di lingkungan keluarga.            |
| **japanese\_chin**               | Japanese Chin                        | Kecil        | Jepang           | Anggun, wajah seperti kucing, lutut halus, sangat setia, cocok tinggal di apartemen.                  |
| **miniature\_pinscher**          | Miniature Pinscher                   | Kecil        | Jerman           | “King of Toys”, gesit, berani, butuh pemilik tegas dan konsisten dalam pelatihan.                     |
| **newfoundland**                 | Newfoundland                         | Sangat Besar | Kanada           | Ahli renang, lembut, penyelamat air, berbulu tebal, cocok keluarga dengan aktivitas di air.           |
| **leonberger**                   | Leonberger                           | Sangat Besar | Jerman           | Mirip singa raksasa, sabar, lembut, penyayang anak-anak, perlu ruang luas dan grooming teratur.       |
| **pug**                          | Pug                                  | Kecil–Sedang | Tiongkok         | Wajah berkerut, mata besar, humoris, penyayang, rentan sensitif terhadap panas, cocok kos/kota.       |
| **pomeranian**                   | Pomeranian                           | Sangat Kecil | Jerman/Polandia  | Bulu tebal “double coat”, percaya diri, vokal, memerlukan perawatan bulu intensif.                    |
| **saint\_bernard**               | Saint Bernard                        | Sangat Besar | Pegunungan Alpen | Penyelamat gunung, sabar, lembut, menjaga keluarga, memerlukan ruang dan perawatan bulu ekstensif.    |
| **samoyed**                      | Samoyed                              | Sedang–Besar | Siberia          | Bulu putih tebal, “senyum Samoyed”, sangat sosial, aktif, cocok untuk iklim dingin.                   |
| **shiba\_inu**                   | Shiba Inu                            | Kecil–Sedang | Jepang           | Mandiri, bersih, waspada, setia, memerlukan sosialisasi untuk mengendalikan sifat teritorial.         |
| **staffordshire\_bull\_terrier** | Staffordshire Bull Terrier           | Sedang       | Inggris          | Kuat, setia, penyayang anak-anak, memerlukan latihan fisik dan pelatihan konsisten.                   |
| **wheaten\_terrier**             | Wheaten Terrier                      | Sedang       | Irlandia         | Bulu lembut gandum, energik, ramah, hypoallergenic, cocok untuk keluarga dengan alergi ringan.        |
| **scottish\_terrier**            | Scottish Terrier                     | Kecil        | Skotlandia       | Tangguh, mandiri, kumis khas, waspada, cocok untuk pemilik berpengalaman.                             |
| **yorkshire\_terrier**           | Yorkshire Terrier                    | Sangat Kecil | Inggris          | Bulu panjang halus, berani, enerjik, ideal sebagai anjing pendamping di dalam ruangan.                |
| **dalmatian**                    | Dalmatian                            | Sedang–Besar | Kroasia          | Bintik hitam pada bulu putih, atletis, aktif, cocok untuk olahraga anjing dan keluarga dinamis.       |
| **siberian\_husky**              | Siberian Husky                       | Sedang       | Siberia          | Stamina luar biasa, mata biru/karamel, independen, ramah, memerlukan aktivitas tinggi.                |
| **doberman**                     | Doberman Pinscher                    | Sedang–Besar | Jerman           | Elegan, protektif, cerdas, setia, memerlukan pemilik tegas dan pelatihan rutin.                       |
| **not\_catxdog**                 | Lain-lain (bukan anjing atau kucing) | —            | —                | Gambar non-hewani atau ambigu.                                                                        |
| **catdog**                       | Hybrid “CatDog”                      | —            | —                | Karakter fiktif campuran kucing dan anjing (editan atau ilustrasi).                                   |

---

## 📝 Cara Penggunaan

1. **Unggah Foto**
   Pilih file `.jpg`, `.jpeg`, atau `.png` dari perangkat, atau aktifkan kamera langsung.

2. **Tunggu Inferensi**
   Model akan memproses dan menghasilkan Top-3 prediksi.

3. **Lihat Hasil**
   Setiap prediksi disertai bar confidence dan deskripsi singkat ras.

---

## 🔧 Konfigurasi Model

* `app.py` otomatis memeriksa file model.
* Untuk mengganti sumber, ubah `file_id` di fungsi `download_model()`.

---

## 🤝 Kontribusi

1. Fork repository.
2. Buat branch baru (`feature/XYZ`).
3. Commit perubahan dan push.
4. Buka Pull Request ke `main`.

---

## 📄 Lisensi

Dirilis di bawah **MIT License**.
© 2025 CatXDog Team
