# PENGEMBANGAN SISTEM DETEKSI ISYARAT DARURAT BISINDO BERBASIS ARSITEKTUR ST-GCN DENGAN PENGUJIAN KETAHANAN TERHADAP OKLUSI PARSIAL

Sistem Deteksi Isyarat Darurat BISINDO (Bahasa Isyarat Indonesia) menggunakan arsitektur **Spatial-Temporal Graph Convolutional Network (ST-GCN)**. Sistem ini dirancang untuk mendeteksi isyarat darurat secara real-time dengan efisiensi tinggi dan biaya operasional Rp 0.

## Fitur Utama

- **Akurasi Tinggi**: Menggunakan ST-GCN 10-block untuk menangkap fitur spasial (skeleton) dan temporal (gerakan).
- **Real-time Inference**: Integrasi MediaPipe Holistic di browser dengan backend FastAPI (WebSocket).
- **Notifikasi Multi-Channel**: Mendukung SMS (via Textbee.dev), Panggilan Suara (via Twilio), dan Web Push Notification.
- **Aesthetic UI**: Antarmuka berbasis React/Vite dengan desain premium dark-mode dan visualisasi skeleton.
- **Robustness**: Diuji terhadap variasi pencahayaan dan oklusi menggunakan teknik data augmentation dan keypoint dropout.

## Struktur Proyek

```
bisindo-emergency/
├── ml/             # ML Pipeline (Graph, Model, Train, Evaluate)
├── backend/        # FastAPI Server, Inference, Notifications
├── frontend/       # React + Vite + MediaPipe UI
└── utils/          # Shared Normalization Logic
```

## Persiapan & Instalasi

### 1. Backend & ML
```bash
cd bisindo-emergency
pip install -r requirements.txt
```

### 2. Frontend
```bash
cd bisindo-emergency/frontend
npm install
```

## Penggunaan

### Langkah 1: Pelatihan Model
1. Letakkan video dataset di `bisindo-emergency/dataset/raw/`.
2. Ekstraksi keypoints: `python -m ml.extract_keypoints`
3. Split data: `python -m ml.dataset`
4. Augmentasi: `python -m ml.augment`
5. Training: `python -m ml.train`

### Langkah 2: Menjalankan Sistem
1. Jalankan Backend: `uvicorn backend.main:app --port 8000`
2. Jalankan Frontend: `cd frontend && npm run dev`
3. Buka browser di `http://localhost:5173`

## Evaluasi
Sistem menyediakan alat evaluasi otomatis:
- `python -m ml.evaluate`: Menghasilkan Confusion Matrix, F1-Score, dan grafik pelatihan.
- `python -m ml.evaluate_dropout`: Menguji redundansi skeleton terhadap kehilangan keypoint (DAR).

## Lisensi
Proyek ini dibuat untuk tujuan edukasi (Karya Tulis Ilmiah). Silakan gunakan secara bebas dengan mencantumkan kredit.

---
**Authors**: Gung Wah, Chelsea, Anin (Kelompok Anomali) - SMA Negeri 3 Denpasar