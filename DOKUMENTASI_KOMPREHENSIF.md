# 📘 Sistem Deteksi Isyarat Darurat BISINDO - Dokumentasi Komprehensif Proyek

**Terakhir Diperbarui:** 6 Februari 2026  
**Status Proyek:** Fase Perencanaan & Desain  
**Target Penyelesaian:** Maret 2026

---

## 📋 Daftar Isi

1. [Gambaran Umum Proyek](#gambaran-umum-proyek)
2. [Konteks Akademik](#konteks-akademik)
3. [Arsitektur Teknis](#arsitektur-teknis)
4. [Metodologi Penelitian](#metodologi-penelitian)
5. [Rencana Implementasi](#rencana-implementasi)
6. [Timeline & Milestone](#timeline--milestone)
7. [Keputusan Kunci & Rasionalisasi](#keputusan-kunci--rasionalisasi)
8. [Status Terkini](#status-terkini)

---

## 1. Gambaran Umum Proyek

### 1.1 Judul Proyek

**"Pengembangan Sistem Deteksi Isyarat Darurat BISINDO Berbasis Arsitektur ST-GCN dengan Ketahanan Terhadap Variasi Pencahayaan dan Oklusi Parsial"**

### 1.2 Pernyataan Masalah

- **Populasi:** 22,97 juta penyandang disabilitas pendengaran di Indonesia (BPS 2023)
- **Kesenjangan Komunikasi:** Masyarakat umum tidak memahami bahasa isyarat, menciptakan hambatan kritis dalam situasi darurat
- **Keterbatasan EWS:** Sistem Peringatan Dini saat ini berbasis audio, tidak dapat diakses oleh komunitas tuli
- **Konteks Bencana:** Indonesia mengalami 5.000+ gempa per tahun ditambah banjir rutin di berbagai daerah

### 1.3 Solusi

Sistem deteksi bahasa isyarat berbasis web secara real-time yang berfungsi sebagai **jembatan komunikasi** antara penyandang disabilitas tuli/bisu dengan masyarakat umum, khususnya untuk situasi darurat.

### 1.4 Fitur Utama

1. **Deteksi Real-time:** 10 kelas isyarat darurat (TOLONG, BAHAYA, KEBAKARAN, SAKIT, GEMPA, BANJIR, PENCURI, PINGSAN, KECELAKAAN, DARURAT)
2. **Notifikasi Otomatis:** Text-to-Speech (TTS) + Notifikasi SMS/Push ke kontak darurat
3. **Ketahanan:** Diuji terhadap variasi pencahayaan dan oklusi parsial
4. **Akses Mobile:** Aplikasi web yang dapat diakses dari smartphone
5. **Pre-training:** Diperkaya dengan dataset alfabet SIBI (26 kelas) untuk generalisasi lebih baik

---

## 2. Konteks Akademik

### 2.1 Jenis Penelitian

- **Kategori:** Eksakta (Ilmu Pengetahuan Alam/Teknik)
- **Metode:** Penelitian dan Pengembangan (R&D)
- **Model:** ADDIE (Analysis, Design, Development, Implementation, Evaluation)

### 2.2 Struktur Penelitian (Format Extended Abstract)

#### BAB I: PENDAHULUAN

- **A.1 Latar Belakang:** Statistik disabilitas, hambatan komunikasi, konteks darurat, kesenjangan teknologi
- **A.2 Tujuan Penelitian:** 4 tujuan (pengembangan sistem, ketahanan pencahayaan, ketahanan oklusi, integrasi notifikasi)
- **A.3 Manfaat Penelitian:** Praktis (komunikasi darurat) + Teoritis (benchmark BISINDO)

#### BAB III: METODE PENELITIAN

- **C.1 Waktu & Tempat:** Januari-Maret 2026, SMAN 3 Denpasar
- **C.2 Jenis Penelitian:** R&D dengan model ADDIE
- **C.3 Variabel Penelitian:**
  - Bebas: Kondisi pencahayaan, tingkat oklusi, jenis isyarat
  - Terikat: Akurasi, latensi, degradasi
  - Kontrol: Jarak, spesifikasi hardware, resolusi video
- **C.4 Rancangan Penelitian:** Diagram alir 5 fase (Analisis → Desain → Pengembangan → Implementasi → Evaluasi)
- **C.5 Alat & Bahan:** Hardware (laptop, webcam) + Software (Python, PyTorch, MediaPipe, FastAPI, React)
- **C.6 Metode Pengumpulan Data:**
  - Kuantitatif: Koordinat landmark, metrik akurasi, pengukuran lux, FPS
  - Kualitatif: Pengujian fungsional, observasi visual
- **C.7 Teknik Analisis Data:** Akurasi, Presisi, Recall, F1-Score, degradasi ketahanan

### 2.3 Persyaratan (Template Extended Abstract 2025)

- **Halaman Minimum:** 10 (termasuk lampiran)
- **Abstrak:** Maksimal 250 kata (Indonesia + Inggris)
- **Kata Kunci:** 3-5 kata
- **Referensi:** Gaya APA, prioritaskan jurnal 5 tahun terakhir
- **Lampiran:** Dokumentasi, data, instrumen, biodata

---

## 3. Arsitektur Teknis

### 3.1 Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (React + Vite)                       │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌─────────┐           │
│  │ Tampilan│ │ Tampilan │ │ Overlay   │ │ Panel   │           │
│  │ Kamera  │ │ Hasil    │ │ Skeleton  │ │ Alert   │           │
│  └────┬────┘ └──────────┘ └───────────┘ └─────────┘           │
└───────┼────────────────────────────────────────────────────────┘
        │ WebSocket (Frame real-time)
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   BACKEND (FastAPI + PyTorch)                    │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐                │
│  │ MediaPipe │ → │  ST-GCN   │ → │ Classifier│                │
│  │ Holistic  │   │  Encoder  │   │ (36 kelas)│                │
│  └───────────┘   └───────────┘   └─────┬─────┘                │
│                                        │                       │
│  ┌─────────────────────────────────────┴───────────────────┐  │
│  │ Output: TTS (pyttsx3) | SMS (Twilio) | Push (Telegram)  │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Arsitektur Model: ST-GCN

**Mengapa ST-GCN dibanding LSTM/TCN?**

| Aspek | LSTM | TCN | ST-GCN ⭐ |
|--------|------|-----|----------|
| Akurasi | 92-97% | 90-95% | **93-98%** |
| Kecepatan | Lambat | Cepat | Cepat |
| Pemodelan Spasial | ❌ | ❌ | ✅ Built-in |
| Kesesuaian Skeleton | Sedang | Sedang | **Sangat Baik** |
| Paralelisasi | ❌ Sekuensial | ✅ Paralel | ✅ Paralel |

**Pipeline ST-GCN:**

```
Input Video (B, T, H, W, 3)
    ↓
MediaPipe Holistic → 75 Keypoints (33 pose + 21×2 tangan)
    ↓
Konstruksi Graf (Adjacency Matrix)
    ↓
Blok ST-GCN (×10)
├─ Spatial Graph Conv (A × X × W)
├─ Temporal Conv (kernel=9)
├─ BatchNorm + ReLU + Dropout
└─ Residual Connection
    ↓
Global Average Pooling
    ↓
Classifier → 36 kelas (26 SIBI + 10 Darurat)
    ↓
Output: Kelas + Confidence + Trigger Alert
```

### 3.3 Strategi Dataset

**Multi-Task Learning:**

1. **Pre-training:** Alfabet SIBI (A-Z) = 26 kelas dari Kaggle
2. **Fine-tuning:** Isyarat Darurat = 10 kelas (rekaman custom)
3. **Total:** 36 kelas untuk training bersama

**Kelas Isyarat Darurat:**

1. TOLONG
2. BAHAYA
3. KEBAKARAN
4. SAKIT
5. GEMPA
6. BANJIR
7. PENCURI
8. PINGSAN
9. KECELAKAAN
10. DARURAT

### 3.4 Tech Stack

| Komponen | Teknologi | Versi |
|-----------|------------|---------|
| **Framework Model** | PyTorch | 2.0+ |
| **Ekstraksi Keypoint** | MediaPipe Holistic | 0.10+ |
| **Backend** | FastAPI | 0.100+ |
| **Komunikasi Real-time** | WebSocket (python-socketio) | 5.0+ |
| **TTS** | pyttsx3 | 2.90+ |
| **SMS** | Twilio API | 8.0+ |
| **Notifikasi Push** | Telegram Bot API | 20.0+ |
| **Frontend** | React 18 + Vite | 5.0+ |
| **Styling** | TailwindCSS | 3.0+ |
| **Database** | SQLite | 3.0+ |

---

## 4. Metodologi Penelitian

### 4.1 Variabel

**Variabel Bebas (Dimanipulasi):**

- Pencahayaan: Terang (>300 lux), Normal (100-300), Redup (50-100), Gelap (<50)
- Oklusi: Tanpa (0%), Ringan (<25%), Sedang (25-50%)
- Jenis Isyarat: 10 kelas darurat

**Variabel Terikat (Diukur):**

- Akurasi Deteksi (%)
- Latensi Inferensi (ms)
- Degradasi Akurasi (%)

**Variabel Kontrol (Konstan):**

- Jarak pengguna: 1-2 meter
- Hardware: Laptop + webcam yang sama
- Resolusi video: 720p

### 4.2 Pengumpulan Data

**Kuantitatif:**

- Koordinat landmark (x, y, z)
- Metrik performa (Akurasi, Presisi, Recall, F1)
- Data lingkungan (Pengukuran Lux)
- Metrik komputasi (Latensi, FPS)

**Kualitatif:**

- Pengujian fungsional (Pengiriman SMS, kejernihan TTS)
- Observasi visual (Stabilitas overlay skeleton)

### 4.3 Metrik Evaluasi

1. **Akurasi:** `(TP + TN) / (TP + TN + FP + FN) × 100%`
2. **Presisi:** `TP / (TP + FP)`
3. **Recall:** `TP / (TP + FN)`
4. **F1-Score:** `2 × (Presisi × Recall) / (Presisi + Recall)`
5. **Tingkat Degradasi:** `(Akurasi_normal - Akurasi_kondisi) / Akurasi_normal × 100%`

### 4.4 Pengujian Ketahanan

**Target Performa:**

| Kondisi | Rentang Lux | Target Akurasi |
|-----------|-----------|-----------------|
| Terang | >300 | ≥95% |
| Normal | 100-300 | ≥90% |
| Redup | 50-100 | ≥85% |
| Gelap | <50 | ≥75% |

| Tingkat Oklusi | Target Akurasi |
|-----------------|-----------------|
| 0% | ≥95% |
| <25% | ≥85% |
| 25-50% | ≥75% |

---

## 5. Rencana Implementasi

### 5.1 Struktur Proyek

```
BISINDO/
├── backend/              # Server FastAPI
│   ├── app/
│   │   ├── main.py
│   │   ├── api/          # Endpoints (prediksi, websocket, notifikasi)
│   │   ├── models/       # Implementasi ST-GCN
│   │   └── services/     # MediaPipe, TTS, SMS
│   └── checkpoints/
├── frontend/             # Aplikasi React
│   └── src/
│       ├── components/   # Camera, ResultPanel, SkeletonOverlay, AlertPanel
│       ├── hooks/        # useWebSocket
│       └── pages/        # Home, Settings
├── training/             # Training Model
│   ├── train_stgcn.py
│   ├── evaluate.py
│   └── configs/
├── data/
│   ├── sibi_alphabet/    # Dataset Kaggle
│   ├── emergency_signs/  # Rekaman custom
│   └── processed/
└── docs/
    ├── API.md
    └── DEMO.md
```

### 5.2 Fase Pengembangan (ADDIE)

**Fase 1: Analisis (Minggu 1-2)**

- Tinjauan literatur (ST-GCN, BISINDO, sistem darurat)
- Definisi 10 kelas isyarat darurat
- Identifikasi sumber dataset

**Fase 2: Desain (Minggu 2-3)**

- Desain arsitektur ST-GCN
- Mockup UI/UX
- Logika alur notifikasi

**Fase 3: Pengembangan (Minggu 3-8)**

- Pengumpulan data (5-10 subjek, berbagai sudut)
- Preprocessing MediaPipe
- Training ST-GCN (target: ≥90% akurasi)
- Pengembangan aplikasi web (FastAPI + React)
- Integrasi notifikasi (TTS, SMS, Telegram)

**Fase 4: Implementasi (Minggu 9-10)**

- Pengujian real-time di lingkungan terkontrol
- Simulasi skenario darurat

**Fase 5: Evaluasi (Minggu 11-12)**

- Pengujian ketahanan (pencahayaan + oklusi)
- Analisis performa
- Dokumentasi & penulisan paper

---

## 6. Timeline & Milestone

| Bulan | Minggu | Milestone |
|-------|------|-----------|
| **Januari 2026** | 1-2 | ✅ Tinjauan literatur, definisi scope |
| | 3-4 | Pengumpulan data dimulai |
| **Februari 2026** | 1-2 | Training model (pre-training SIBI) |
| | 3-4 | Fine-tuning pada isyarat darurat |
| **Maret 2026** | 1-2 | Pengembangan aplikasi web + integrasi |
| | 3-4 | Pengujian, analisis, penulisan paper |

**Deadline:** Maret 2026 (Pengumpulan Extended Abstract)

---

## 7. Keputusan Kunci & Rasionalisasi

### 7.1 Arsitektur: ST-GCN vs LSTM/TCN

**Keputusan:** ST-GCN  
**Rasionalisasi:**

- Dirancang khusus untuk pengenalan aksi berbasis skeleton
- Menggabungkan spasial (hubungan sendi) dan temporal (gerakan) dalam satu framework
- Lebih efisien dari LSTM (tidak ada masalah vanishing gradient)
- State-of-the-art untuk pengenalan gesture (5000+ sitasi)

### 7.2 Platform: Aplikasi Web

**Keputusan:** Inferensi berbasis server dengan frontend React  
**Rasionalisasi:**

- Dapat diakses mobile tanpa instalasi aplikasi
- Deployment dan update lebih mudah
- Sumber daya komputasi lebih baik di server
- Kompatibilitas lintas platform

### 7.3 Dataset: SIBI + Isyarat Darurat

**Keputusan:** Pendekatan transfer learning  
**Rasionalisasi:**

- Alfabet SIBI memberikan pemahaman bahasa isyarat umum
- Isyarat darurat mendapat manfaat dari representasi pre-trained
- Meningkatkan kemampuan generalisasi model

### 7.4 Scope: Fokus Darurat

**Keputusan:** 10 isyarat darurat (bukan kosakata BISINDO lengkap)  
**Rasionalisasi:**

- Mengatasi kebutuhan keselamatan kritis
- Dapat dikelola untuk timeline penelitian tingkat SMA
- Kriteria evaluasi yang jelas
- Dampak sosial tinggi

---

## 8. Status Terkini

### 8.1 Selesai

- ✅ Definisi scope proyek
- ✅ Pemilihan arsitektur (ST-GCN)
- ✅ Desain metodologi penelitian
- ✅ Bab 1 Pendahuluan (draft)
- ✅ Bab 3 Metode Penelitian (lengkap)
- ✅ Rencana implementasi
- ✅ Penetapan timeline

### 8.2 Sedang Berlangsung

- 🔄 Penyempurnaan tinjauan literatur
- 🔄 Identifikasi dataset (SIBI Kaggle)
- 🔄 Penulisan paper akademik

### 8.3 Tertunda

- ⏳ Implementasi ST-GCN
- ⏳ Pengumpulan data (isyarat darurat)
- ⏳ Training model
- ⏳ Pengembangan aplikasi web
- ⏳ Pengujian ketahanan
- ⏳ Pengumpulan paper final

---

## 9. Referensi & Sumber Daya

### 9.1 Paper Kunci

- Zhao & Chen (2023): ST-GCN untuk regresi dan ekstraksi fitur
- Lu et al. (2023): Pengenalan bahasa isyarat dengan sensor multimodal
- Yan et al. (2018): Paper ST-GCN original (pengenalan aksi)

### 9.2 Dataset

- Alfabet SIBI: Kaggle (26 kelas, A-Z)
- Isyarat Darurat: Rekaman custom (10 kelas)

### 9.3 Tools & Library

- MediaPipe: <https://google.github.io/mediapipe/>
- PyTorch: <https://pytorch.org/>
- FastAPI: <https://fastapi.tiangolo.com/>
- React: <https://react.dev/>

---

## 10. Kontak & Tim

**Institusi:** SMA Negeri 3 Denpasar  
**Jenis Proyek:** Karya Tulis Ilmiah  
**Kategori:** Eksakta (Ilmu Pengetahuan Alam/Teknik)  
**Timeline:** Januari - Maret 2026

---

**Versi Dokumen:** 1.0  
**Terakhir Diperbarui:** 6 Februari 2026  
**Status:** Dokumen Hidup (Diperbarui seiring kemajuan proyek)
