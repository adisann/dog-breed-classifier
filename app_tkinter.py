import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
import threading
import queue

class PetBreedClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Pet Breed Classifier")
        self.root.geometry("800x700")
        self.root.configure(bg="#f0f0f0")
        
        # Variabel untuk menyimpan gambar
        self.uploaded_image = None
        self.display_image = None
        
        # Queue untuk komunikasi antar thread
        self.queue = queue.Queue()
        
        # Load model dan class names
        self.model, self.class_names = self.load_model()
        
        # Tampilan aplikasi
        self.create_widgets()
        
        # Thread untuk memproses prediksi
        self.prediction_thread = None
        
        # Periodic check untuk hasil prediksi
        self.check_queue()
    
    def load_model(self):
        """Load model dan class names"""
        try:
            model = tf.keras.models.load_model('pet_breed_classifier_final.h5')
            if os.path.exists('class_mapping.csv'):
                df = pd.read_csv('class_mapping.csv')
                class_names = df['class_name'].tolist()
            else:
                class_names = list(self.build_class_descriptions().keys())
            return model, class_names
        except Exception as e:
            messagebox.showerror("Error", f"Error loading model: {e}")
            return None, None
    
    def build_class_descriptions(self):
        """Mapping kelas ke deskripsi"""
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
    
    def create_widgets(self):
        """Buat semua elemen GUI"""
        # Header
        header = tk.Frame(self.root, bg="#4CAF50", padx=10, pady=10)
        header.pack(fill=tk.X)
        
        tk.Label(
            header, 
            text="🐾 Pet Breed Classifier", 
            font=("Helvetica", 20, "bold"), 
            fg="white", 
            bg="#4CAF50"
        ).pack()
        
        # Konten utama
        main_frame = tk.Frame(self.root, bg="#f0f0f0", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Instruksi
        tk.Label(
            main_frame, 
            text="Upload foto anjing atau kucing untuk mengetahui breed-nya!", 
            font=("Helvetica", 12), 
            bg="#f0f0f0"
        ).pack(pady=10)
        
        # Tombol upload
        btn_frame = tk.Frame(main_frame, bg="#f0f0f0")
        btn_frame.pack(pady=10)
        
        upload_btn = ttk.Button(
            btn_frame, 
            text="Upload Gambar", 
            command=self.upload_image
        )
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        # Tombol analisis
        self.analyze_btn = ttk.Button(
            btn_frame, 
            text="Analisis", 
            command=self.analyze_image,
            state=tk.DISABLED
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Tombol buka kamera
        camera_btn = ttk.Button(
            btn_frame, 
            text="Buka Kamera", 
            command=self.open_camera
        )
        camera_btn.pack(side=tk.LEFT, padx=5)
        
        # Frame untuk gambar
        self.image_frame = tk.Frame(main_frame, bg="white", width=400, height=300)
        self.image_frame.pack(pady=10)
        self.image_frame.pack_propagate(False)
        
        self.image_label = tk.Label(self.image_frame, bg="white")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Label loading
        self.loading_label = tk.Label(main_frame, text="", bg="#f0f0f0")
        self.loading_label.pack(pady=5)
        
        # Frame untuk hasil
        self.result_frame = tk.Frame(main_frame, bg="#f0f0f0")
        self.result_frame.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, length=300, mode='indeterminate')
        
        # Footer
        footer = tk.Frame(self.root, bg="#f0f0f0", padx=10, pady=5)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        
        tk.Label(
            footer, 
            text="Pet Breed Classifier menggunakan ResNet101 dengan transfer learning", 
            font=("Helvetica", 8), 
            fg="#666", 
            bg="#f0f0f0"
        ).pack()
    
    def upload_image(self):
        """Menangani upload gambar"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                self.uploaded_image = Image.open(file_path)
                self.display_uploaded_image()
                self.analyze_btn.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open image: {e}")
    
    def open_camera(self):
        """Fungsi untuk membuka kamera menggunakan OpenCV"""
        try:
            # Buka jendela kamera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Cannot open camera")
                return
                
            # Buat jendela baru untuk kamera
            camera_window = tk.Toplevel(self.root)
            camera_window.title("Camera")
            camera_window.geometry("640x520")
            
            # Frame untuk tampilan kamera
            camera_frame = tk.Frame(camera_window, width=640, height=480)
            camera_frame.pack(pady=5)
            
            camera_label = tk.Label(camera_frame)
            camera_label.pack()
            
            # Tombol untuk mengambil gambar
            capture_btn = ttk.Button(
                camera_window, 
                text="Ambil Foto", 
                command=lambda: self.capture_image(cap, camera_window)
            )
            capture_btn.pack(pady=5)
            
            def update_frame():
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    camera_label.imgtk = imgtk
                    camera_label.configure(image=imgtk)
                    if camera_window.winfo_exists():
                        camera_window.after(10, update_frame)
                else:
                    cap.release()
                    camera_window.destroy()
            
            update_frame()
            
            # Ketika jendela ditutup
            def on_closing():
                cap.release()
                camera_window.destroy()
            
            camera_window.protocol("WM_DELETE_WINDOW", on_closing)
            
        except Exception as e:
            messagebox.showerror("Error", f"Camera error: {e}")
    
    def capture_image(self, cap, window):
        """Mengambil gambar dari kamera"""
        ret, frame = cap.read()
        if ret:
            # Konversi frame ke RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.uploaded_image = Image.fromarray(frame_rgb)
            
            # Tutup jendela kamera
            cap.release()
            window.destroy()
            
            # Tampilkan gambar yang diambil
            self.display_uploaded_image()
            self.analyze_btn.config(state=tk.NORMAL)
    
    def display_uploaded_image(self):
        """Menampilkan gambar yang diupload"""
        if self.uploaded_image:
            # Resize image agar fit di frame
            img_width, img_height = self.uploaded_image.size
            max_width = 380
            max_height = 280
            
            # Hitung rasio
            ratio = min(max_width/img_width, max_height/img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            resized_img = self.uploaded_image.resize((new_width, new_height), Image.LANCZOS)
            self.display_image = ImageTk.PhotoImage(resized_img)
            
            self.image_label.config(image=self.display_image)
            self.image_label.image = self.display_image  # Simpan referensi
    
    def analyze_image(self):
        """Memulai analisis gambar"""
        if self.uploaded_image and self.model:
            # Clear hasil sebelumnya
            for widget in self.result_frame.winfo_children():
                widget.destroy()
            
            # Tampilkan loading
            self.loading_label.config(text="Menganalisis gambar...")
            self.progress.pack(pady=10)
            self.progress.start(10)
            
            # Nonaktifkan tombol selama proses
            self.analyze_btn.config(state=tk.DISABLED)
            
            # Mulai thread untuk prediksi
            self.prediction_thread = threading.Thread(
                target=self.predict_in_thread, 
                args=(self.uploaded_image,)
            )
            self.prediction_thread.daemon = True
            self.prediction_thread.start()
        else:
            if not self.uploaded_image:
                messagebox.showinfo("Info", "Silakan upload gambar terlebih dahulu")
            else:
                messagebox.showerror("Error", "Model tidak dapat dimuat")
    
    def predict_in_thread(self, image):
        """Melakukan prediksi dalam thread terpisah"""
        try:
            # Preprocessing
            img_array = self.preprocess_image(image)
            
            # Prediksi
            preds = self.predict_breed(img_array)
            
            # Masukkan hasil ke queue
            self.queue.put(preds)
        except Exception as e:
            self.queue.put(f"Error: {e}")
    
    def check_queue(self):
        """Mengecek queue untuk hasil prediksi"""
        try:
            if not self.queue.empty():
                result = self.queue.get()
                if isinstance(result, str) and result.startswith("Error"):
                    messagebox.showerror("Error", result)
                else:
                    self.display_results(result)
                
                # Selesai loading
                self.loading_label.config(text="")
                self.progress.stop()
                self.progress.pack_forget()
                self.analyze_btn.config(state=tk.NORMAL)
            
            # Cek lagi nanti
            self.root.after(100, self.check_queue)
        except Exception as e:
            messagebox.showerror("Error", f"Queue error: {e}")
    
    def preprocess_image(self, image, img_size=(224, 224)):
        """Preprocessing image untuk model"""
        if isinstance(image, np.ndarray):
            image = cv2.resize(image, img_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image.resize(img_size)
            image = np.array(image)
        arr = np.expand_dims(image, axis=0)
        return tf.keras.applications.resnet.preprocess_input(arr)
    
    def predict_breed(self, image_array):
        """Prediksi breed dari gambar"""
        preds = self.model.predict(image_array)
        idxs = np.argsort(preds[0])[-3:][::-1]
        results = []
        for i in idxs:
            key = self.class_names[i]
            prob = float(preds[0][i]) * 100
            results.append({"key": key, "confidence": prob})
        return results
    
    def display_results(self, predictions):
        """Menampilkan hasil prediksi"""
        # Buat scrollable frame untuk hasil
        canvas = tk.Canvas(self.result_frame, bg="#f0f0f0")
        scrollbar = ttk.Scrollbar(self.result_frame, orient="vertical", command=canvas.yview)
        
        # Frame yang akan diletakkan di canvas
        results_container = tk.Frame(canvas, bg="#f0f0f0")
        
        # Konfigurasi canvas
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Buat window untuk frame di canvas
        canvas_window = canvas.create_window((0, 0), window=results_container, anchor="nw")
        
        # Update canvas saat frame berubah
        def _configure_canvas(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        results_container.bind("<Configure>", _configure_canvas)
        
        # Judul hasil
        tk.Label(
            results_container, 
            text="Hasil Prediksi", 
            font=("Helvetica", 14, "bold"), 
            bg="#f0f0f0"
        ).pack(pady=(10, 5))
        
        # Format dan tampilkan prediksi teratas
        title, desc = self.format_prediction(predictions[0]['key'], predictions[0]['confidence'])
        
        top_frame = tk.Frame(results_container, bg="#f1f1f1", padx=10, pady=10, relief=tk.RIDGE, borderwidth=1)
        top_frame.pack(fill=tk.X, pady=5, padx=5)
        
        tk.Label(
            top_frame, 
            text=f"🏆 {title}", 
            font=("Helvetica", 12, "bold"), 
            bg="#f1f1f1"
        ).pack(anchor=tk.W)
        
        tk.Label(
            top_frame, 
            text=desc, 
            font=("Helvetica", 10), 
            bg="#f1f1f1", 
            wraplength=500,
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=5)
        
        # Progress bar untuk confidence
        confidence_frame = tk.Frame(top_frame, bg="#f1f1f1")
        confidence_frame.pack(fill=tk.X, pady=5)
        
        confidence_bar = ttk.Progressbar(
            confidence_frame, 
            orient=tk.HORIZONTAL, 
            length=500, 
            mode='determinate', 
            value=predictions[0]['confidence']
        )
        confidence_bar.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        
        tk.Label(
            confidence_frame, 
            text=f"{predictions[0]['confidence']:.1f}%", 
            bg="#f1f1f1"
        ).pack(side=tk.RIGHT)
        
        # Judul kemungkinan lainnya
        tk.Label(
            results_container, 
            text="Kemungkinan Lainnya", 
            font=("Helvetica", 12, "bold"), 
            bg="#f0f0f0"
        ).pack(pady=(10, 5))
        
        # Format dan tampilkan prediksi lainnya
        for pred in predictions[1:]:
            title, desc = self.format_prediction(pred['key'], pred['confidence'])
            
            pred_frame = tk.Frame(results_container, bg="#f1f1f1", padx=10, pady=10, relief=tk.RIDGE, borderwidth=1)
            pred_frame.pack(fill=tk.X, pady=5, padx=5)
            
            tk.Label(
                pred_frame, 
                text=title, 
                font=("Helvetica", 11, "bold"), 
                bg="#f1f1f1"
            ).pack(anchor=tk.W)
            
            tk.Label(
                pred_frame, 
                text=desc, 
                font=("Helvetica", 10), 
                bg="#f1f1f1", 
                wraplength=500,
                justify=tk.LEFT
            ).pack(anchor=tk.W, pady=5)
            
            # Progress bar untuk confidence
            conf_frame = tk.Frame(pred_frame, bg="#f1f1f1")
            conf_frame.pack(fill=tk.X, pady=5)
            
            conf_bar = ttk.Progressbar(
                conf_frame, 
                orient=tk.HORIZONTAL, 
                length=500, 
                mode='determinate', 
                value=pred['confidence']
            )
            conf_bar.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
            
            tk.Label(
                conf_frame, 
                text=f"{pred['confidence']:.1f}%", 
                bg="#f1f1f1"
            ).pack(side=tk.RIGHT)
    
    def format_prediction(self, breed_key, confidence):
        """Format prediksi untuk ditampilkan"""
        class_descriptions = self.build_class_descriptions()
        cat_breeds = {k for k, d in class_descriptions.items() if d.startswith("Kucing")}
        
        name = breed_key.replace('_', ' ').title()
        desc = class_descriptions.get(breed_key, "Tidak ada deskripsi tersedia.")
        
        if breed_key in cat_breeds:
            title = f"Kucing dengan Ras {name}, Tingkat kepercayaan {confidence:.1f}%"
        elif breed_key in ["not_catxdog", "garfield", "catdog"]:
            title = name
        else:
            title = f"Anjing dengan Ras {name}, Tingkat kepercayaan {confidence:.1f}%"
        
        return title, desc

if __name__ == "__main__":
    root = tk.Tk()
    app = PetBreedClassifier(root)
    root.mainloop()