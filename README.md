# Iqra AI: Sistem Pengenalan Huruf Konsonan Arab

Iqra AI adalah proyek *Machine Learning* berbasis **Jaringan Saraf Tiruan (JST)** yang menggunakan arsitektur **Single-Layer Multi-Node Perceptron** untuk mengenali dan mengklasifikasikan 14 huruf konsonan Arab dalam format *isolated*.

Sistem ini dilengkapi dengan antarmuka interaktif berbasis Streamlit sehingga pengguna dapat melakukan pengujian model secara langsung dan visual.

---

## 📌 Fitur Utama

- Klasifikasi 14 huruf konsonan Arab
- Implementasi algoritma Perceptron multi-node
- Visualisasi proses training dan error
- Antarmuka interaktif menggunakan Streamlit
- Struktur proyek modular dan mudah dikembangkan

---

## 🧠 Arsitektur Model

### Struktur JST
- **Input Layer:** 35 neuron  
  Merepresentasikan grid piksel berukuran `7 × 5`

- **Output Layer:** 14 neuron  
  Merepresentasikan 14 kelas huruf Arab

### Konfigurasi Model
- **Arsitektur:** Single-Layer Multi-Node Perceptron
- **Fungsi Aktivasi:** Bipolar Threshold
- **Learning Rate ($\alpha$):** `1` *(default)*

---

## 🛠️ Teknologi yang Digunakan

| Teknologi | Fungsi |
|---|---|
| Python 3.10+ | Bahasa pemrograman utama |
| NumPy | Komputasi numerik & manipulasi matriks bobot |
| Matplotlib | Visualisasi grafik error & decision boundary |
| Streamlit | Antarmuka pengguna interaktif |

---

## 📂 Struktur Proyek

```bash
IqraAI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── train.py
│   ├── model.py
│   ├── utils.py
│   └── dataset.py
│
├── assets/
│   └── images/
│
└── data/
    └── dataset.csv
```

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Clone Repository

```bash
git clone https://github.com/USERNAME_ANDA/IqraAI.git
cd IqraAI
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. (Opsional) Jalankan Training Model

```bash
python src/train.py
```

---

### 4. Jalankan Aplikasi Streamlit

```bash
python -m streamlit run app.py
```

---

## 📊 Dataset

Dataset direpresentasikan dalam bentuk grid biner berukuran `7 × 5` yang menggambarkan pola masing-masing huruf Arab.

Setiap pola:
- Memiliki 35 fitur input
- Direpresentasikan menggunakan nilai bipolar (`-1` dan `1`)
- Dipetakan ke salah satu dari 14 kelas huruf

---

## 🎯 Tujuan Proyek

Proyek ini dibuat untuk:
- Mempelajari implementasi dasar JST/Perceptron
- Memahami proses training dan klasifikasi pola
- Mengembangkan sistem pengenalan karakter sederhana
- Menjadi media pembelajaran AI dan pengolahan citra dasar

---

## 📈 Pengembangan Selanjutnya

Beberapa pengembangan yang dapat dilakukan:
- Menambah jumlah dataset huruf Arab
- Mengimplementasikan Multi-Layer Perceptron (MLP)
- Menambahkan fitur upload gambar tulisan tangan
- Menggunakan CNN untuk akurasi yang lebih tinggi
- Deploy aplikasi ke Streamlit Cloud

---

## 👨‍💻 Author

Developed by **KELOMPOK 2 KECERDASAN BUATAN & PEMBELAJARAN MESIN TI-C**  
Project for Artificial Intelligence / Neural Network Learning

---

## 📄 License

This project is licensed under the MIT License.
