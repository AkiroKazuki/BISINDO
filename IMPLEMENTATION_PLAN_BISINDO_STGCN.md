# IMPLEMENTATION PLAN
## Sistem Deteksi Isyarat Darurat BISINDO Berbasis ST-GCN
**Kelompok Anomali — SMA Negeri 3 Denpasar 2026**

---

## GAMBARAN ARSITEKTUR KESELURUHAN

```
[Webcam] 
    ↓
[MediaPipe Holistic JS] — ekstraksi 75 keypoint real-time di browser
    ↓
[WebSocket] — streaming koordinat ke backend
    ↓
[FastAPI Backend]
    ├── Sliding Window Buffer (60 frame, stride 15)
    ├── Normalisasi koordinat
    └── ST-GCN Inference
            ↓
    [Confirmation Logic] — 4/5 inferensi terakhir + confidence > 0.85
            ↓
    [Notifikasi Pipeline]
    ├── Web Speech API (TTS di browser)
    ├── Twilio SMS
    ├── Twilio Voice (telepon otomatis)
    └── Web Push Notification (service worker)
```

Stack teknologi:
- **ML:** Python, PyTorch, MediaPipe Holistic (Python SDK untuk dataset collection)
- **Backend:** FastAPI, Uvicorn, WebSocket, Twilio
- **Frontend:** React + Vite, MediaPipe Holistic (JS SDK), Web Speech API, Web Push API

---

## BAGIAN 1 — ENVIRONMENT SETUP

Install semua dependency sebelum mulai apapun.

### Python Environment
```
torch>=2.0.0
torchvision
mediapipe>=0.10.0
numpy
scikit-learn
scipy
fastapi
uvicorn[standard]
websockets
python-multipart
twilio
matplotlib
seaborn
tqdm
pywebpush
```

### JavaScript (Frontend)
```
react
vite
@mediapipe/holistic
@mediapipe/camera_utils
@mediapipe/drawing_utils
```

### Struktur Direktori Proyek
```
bisindo-emergency/
├── dataset/
│   ├── raw/
│   │   ├── TOLONG/          # 120 video .mp4/.avi
│   │   ├── BAHAYA/          # 120 video
│   │   └── KEBAKARAN/       # 120 video
│   └── processed/
│       ├── TOLONG/          # file .npy hasil ekstraksi
│       ├── BAHAYA/
│       └── KEBAKARAN/
├── ml/
│   ├── extract_keypoints.py
│   ├── augment.py
│   ├── dataset.py
│   ├── graph.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_dropout.py
│   └── checkpoints/
├── backend/
│   ├── main.py
│   ├── inference.py
│   ├── notification.py
│   └── buffer.py
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── CameraView.jsx
│   │   │   ├── SkeletonOverlay.jsx
│   │   │   ├── DetectionDisplay.jsx
│   │   │   └── EmergencyContact.jsx
│   │   ├── hooks/
│   │   │   ├── useMediaPipe.js
│   │   │   └── useWebSocket.js
│   │   └── App.jsx
│   └── public/
│       └── sw.js            # service worker untuk push notification
└── requirements.txt
```

---

## BAGIAN 2 — DATASET COLLECTION

### 2.1 Protokol Perekaman

| Kriteria | Ketentuan |
|---|---|
| Durasi | 2–3 detik per gerakan |
| Resolusi | Minimal 720p |
| Frame rate | 30 fps |
| Framing | Kepala hingga pinggang terlihat penuh |
| Tangan | Tidak keluar frame saat isyarat dilakukan |
| Latar belakang | Bebas — variasi antar subjek justru meningkatkan generalisasi |

Variasi yang **wajib ada** antar 120 video per kelas:
- Minimal 5–10 subjek berbeda
- Variasi kecepatan gerakan (lambat, normal, cepat)
- Variasi jarak ke kamera: 50cm, 100cm, 150cm
- Variasi posisi horizontal dalam frame (kiri, tengah, kanan)
- Variasi pencahayaan (terang, normal) — tidak perlu terstandarisasi

### 2.2 Verifikasi Gesture BISINDO

Sebelum mulai rekam, verifikasi gesture ketiga kelas ke **Kamus Kosa Isyarat Pusbisindo**. Jika komunitas tuli merespons, lakukan verifikasi ulang dengan native signer. Jika ada perbedaan antara referensi Pusbisindo dan native signer, ikuti verifikasi native signer dan catat perbedaannya — ini bisa masuk sebagai temuan di bagian pembahasan.

---

## BAGIAN 3 — EKSTRAKSI KEYPOINT (`ml/extract_keypoints.py`)

### 3.1 Logika Ekstraksi

Alur per video:
1. Buka video menggunakan OpenCV
2. Jalankan MediaPipe Holistic pada setiap frame
3. Ekstrak koordinat `(x, y, z)` dari 75 landmark:
   - Pose landmarks: index 0–32 (33 titik)
   - Left hand landmarks: index 33–53 (21 titik)
   - Right hand landmarks: index 54–74 (21 titik)
4. Jika MediaPipe gagal mendeteksi di frame tertentu → isi dengan `np.zeros((75, 3))`
5. Stack semua frame menjadi array `(T, 75, 3)`
6. Normalisasi koordinat (lihat 3.2)
7. Padding atau trimming ke panjang tepat **60 frame** (lihat 3.3)
8. Simpan sebagai file `.npy`

Penamaan file output:
```
dataset/processed/TOLONG/TOLONG_001.npy
dataset/processed/TOLONG/TOLONG_002.npy
... dst
```

### 3.2 Normalisasi Koordinat

1. Ambil koordinat bahu kiri (pose landmark index 11) dan bahu kanan (pose landmark index 12)
2. Hitung titik tengah bahu: `shoulder_center = (left_shoulder + right_shoulder) / 2`
3. Hitung faktor skala: `scale = ||left_shoulder - right_shoulder||`
4. Kurangi semua 75 koordinat dengan `shoulder_center`
5. Bagi semua koordinat dengan `scale` — jika `scale == 0`, skip normalisasi untuk frame tersebut

### 3.3 Padding dan Trimming ke 60 Frame

- **Lebih dari 60 frame:** center crop temporal
  - `start = (T - 60) // 2`
  - `frames = frames[start : start + 60]`
- **Kurang dari 60 frame:** repeat last frame di bagian akhir sampai panjang 60
  - Jangan padding dengan nol — akan diinterpretasi sebagai oklusi

### 3.4 Verifikasi Output

- Semua file `.npy` harus berukuran `(60, 75, 3)`
- Nilai koordinat terpusat di sekitar 0 setelah normalisasi
- Tidak ada file yang seluruhnya nol
- Total file: 120 × 3 = **360 file .npy**

---

## BAGIAN 4 — AUGMENTASI DATA (`ml/augment.py`)

Augmentasi dilakukan pada array NumPy — **bukan pada video.** Setiap sampel mentah menghasilkan 4 versi augmentasi.

Total dataset setelah augmentasi: 360 × 5 = **1.800 sampel**

> **Penting:** Augmentasi **hanya** diterapkan ke train split. Validation dan test set selalu menggunakan data original.

### 4.1 Augmentasi 1 — Gaussian Noise
- Distribusi: `N(mean=0, std=0.01)`
- Terapkan ke seluruh tensor `(60, 75, 3)` sekaligus
- Mensimulasikan ketidakpresisian deteksi MediaPipe

### 4.2 Augmentasi 2 — Time Warping
- Faktor warp acak antara **0.8–1.2**
- Gunakan `scipy.interpolate.interp1d` pada dimensi temporal
- Interpolasi kembali ke panjang tepat 60 frame setelah warping
- Mensimulasikan variasi kecepatan gerakan antar pengguna

### 4.3 Augmentasi 3 — Spatial Scaling
- Faktor scale acak antara **0.9–1.1**
- Faktor yang **sama** diterapkan ke semua 60 frame dalam satu sekuens
- Mensimulasikan perbedaan jarak subjek ke kamera

### 4.4 Augmentasi 4 — Temporal Coordinate Jitter
- Buat noise `n_start ~ N(0, 0.005)` untuk frame pertama
- Buat noise `n_end ~ N(0, 0.005)` untuk frame terakhir
- Interpolasi linear dari `n_start` ke `n_end` sepanjang 60 frame
- Tambahkan ke koordinat
- Mensimulasikan getaran tangan yang natural dan konsisten

### 4.5 Yang TIDAK Boleh Dilakukan

| Teknik | Alasan Dilarang |
|---|---|
| Horizontal flip | Membalik kiri-kanan mengubah makna isyarat BISINDO |
| Rotasi > 15 derajat | Mengubah orientasi gerakan yang bermakna |
| Pembalikan urutan frame | Mengubah sekuens temporal gerakan |
| Random frame drop | Merusak panjang sekuens yang harus tepat 60 |

---

## BAGIAN 5 — DATASET LOADER (`ml/dataset.py`)

### 5.1 Split Dataset

Split ratio: **70% train / 15% validation / 15% test**

- Split dilakukan secara **stratified** — distribusi kelas seimbang di setiap split
- Simpan indeks split ke file JSON dengan `random_seed=42` supaya reproducible
- Label mapping: `TOLONG → 0`, `BAHAYA → 1`, `KEBAKARAN → 2`

Jumlah sampel per split:
- **Train:** 84 sampel original × 3 kelas × 5 (original + 4 aug) = **1.260 sampel**
- **Validation:** 18 sampel original × 3 kelas = **54 sampel**
- **Test:** 18 sampel original × 3 kelas = **54 sampel**

### 5.2 PyTorch Dataset Class

Harus mengembalikan tensor `(3, 60, 75)` per sampel — format `(channel, time, node)` yang dibutuhkan ST-GCN.

Reshape: dari `(60, 75, 3)` ke `(3, 60, 75)` menggunakan `np.transpose(arr, (2, 0, 1))`.

---

## BAGIAN 6 — GRAPH DEFINITION (`ml/graph.py`)

### 6.1 Daftar Edge

**Pose connections (33 node, index 0–32):**
Ikuti koneksi bawaan MediaPipe — hidung-mata, mata-telinga, bahu-siku, siku-pergelangan, pinggul-lutut, lutut-ankle, dan koneksi tubuh bagian atas lainnya.

**Left hand connections (21 node, index 33–53):**
Pangkal tangan ke setiap pangkal jari, lalu dari pangkal jari ke setiap ruas secara berurutan untuk semua 5 jari.

**Right hand connections (21 node, index 54–74):**
Identik dengan left hand, offset index +21.

**Cross-body connections (WAJIB ADA):**
- Pose left wrist (index 15) → Left hand wrist (index 33)
- Pose right wrist (index 16) → Right hand wrist (index 54)

Tanpa cross-body connections, informasi tidak akan mengalir antara representasi pose dan tangan.

### 6.2 Adjacency Matrix

1. Inisialisasi `A = np.zeros((75, 75))`
2. Untuk setiap edge `(i, j)`: set `A[i][j] = 1` dan `A[j][i] = 1` (undirected)
3. Tambahkan self-loop: `A += np.eye(75)`
4. Hitung degree matrix `D`: diagonal, `D[i][i] = sum(A[i])`
5. Symmetric normalization: `A_hat = D^(-1/2) @ A @ D^(-1/2)`
6. Konversi ke PyTorch tensor, register sebagai **buffer** (bukan parameter yang dioptimasi)

---

## BAGIAN 7 — MODEL ARCHITECTURE (`ml/model.py`)

### 7.1 Spatial Graph Convolution Layer

- Input: `(batch, C_in, T, V)` — batch, channel, time, vertex
- Operasi per timestep: `H_out = A_hat @ H_in @ W`
- Implementasi efisien:
  1. Reshape H_in → `(batch * T, V, C_in)`
  2. Batch matrix multiply dengan A_hat `(V, V)`
  3. Reshape output kembali
  4. Terapkan `nn.Conv2d(C_in, C_out, kernel_size=1)` sebagai weight matrix W
- Output: `(batch, C_out, T, V)`

### 7.2 Temporal Convolution Layer

Setelah spatial convolution:
- `nn.Conv2d(C_out, C_out, kernel_size=(9, 1), padding=(4, 0))`
- `kernel_size=(9, 1)`: 9 pada dimensi temporal, 1 pada dimensi node
- `padding=(4, 0)`: panjang temporal tidak berubah

### 7.3 ST-GCN Block

```
Input (batch, C_in, T, V)
    ↓
Spatial Graph Conv → (batch, C_out, T, V)
    ↓
BatchNorm2d(C_out) → ReLU
    ↓
Temporal Conv → (batch, C_out, T, V)
    ↓
BatchNorm2d(C_out)
    ↓
Residual:
  - Jika C_in == C_out → identity shortcut
  - Jika C_in != C_out → Conv2d(C_in, C_out, 1) + BatchNorm2d(C_out)
    ↓
Add residual + ReLU
    ↓
Output (batch, C_out, T, V)
```

### 7.4 Konfigurasi 10 Blok ST-GCN

| Blok | C_in | C_out |
|---|---|---|
| 1 | 3 | 64 |
| 2 | 64 | 64 |
| 3 | 64 | 64 |
| 4 | 64 | 128 |
| 5 | 128 | 128 |
| 6 | 128 | 128 |
| 7 | 128 | 128 |
| 8 | 128 | 256 |
| 9 | 256 | 256 |
| 10 | 256 | 256 |

### 7.5 Output Layer

```
Global Average Pooling (temporal + node) → (batch, 256)
    ↓
Dropout(p=0.5)
    ↓
Linear(256, 3)
    ↓
Output logits (batch, 3)
```

> **Penting:** Output adalah **logits**, bukan probabilitas. Jangan tambahkan Softmax di model — CrossEntropyLoss sudah menanganinya. Tambahkan Softmax **hanya saat inference** untuk mendapatkan confidence score.

---

## BAGIAN 8 — TRAINING (`ml/train.py`)

### 8.1 Konfigurasi

| Parameter | Nilai |
|---|---|
| Loss function | CrossEntropyLoss |
| Optimizer | Adam(lr=0.001, weight_decay=1e-4) |
| LR Scheduler | ReduceLROnPlateau(mode='min', factor=0.5, patience=10) |
| Batch size | 32 |
| Max epochs | 200 |
| Early stopping | patience=20, monitor val_loss |
| Checkpoint | Simpan model dengan val_accuracy tertinggi |

### 8.2 Training Loop

Per epoch:
1. **Train phase:** forward pass → hitung loss → backward → optimizer step
2. **Val phase:** tanpa gradient → hitung val_loss dan val_accuracy
3. **Scheduler step:** `scheduler.step(val_loss)`
4. **Early stopping:** jika val_loss tidak membaik 20 epoch berturut → stop
5. **Checkpoint:** jika val_accuracy > best → simpan model

### 8.3 Logging

Catat per epoch: `train_loss`, `val_loss`, `val_accuracy`, `current_lr` → simpan ke CSV untuk membuat training curve di bagian hasil.

### 8.4 Tanda Model Bermasalah

Jika melihat hal ini, **cek normalisasi koordinat terlebih dahulu** sebelum mengubah arsitektur:
- Loss tidak turun setelah 30 epoch pertama
- Val accuracy stuck di ~33% (random chance untuk 3 kelas)
- Loss explodes (NaN atau sangat besar)

---

## BAGIAN 9 — EVALUASI (`ml/evaluate.py` dan `ml/evaluate_dropout.py`)

### 9.1 Evaluasi Kondisi Normal

Load checkpoint terbaik, jalankan inferensi pada **test set** (54 sampel, original tanpa augmentasi):
- Hitung accuracy, precision, recall, F1-score — per kelas dan macro average
- Generate confusion matrix 3×3 sebagai heatmap

### 9.2 Evaluasi Keypoint Dropout (`evaluate_dropout.py`)

```
Untuk setiap dropout_rate dalam [0%, 10%, 25%, 50%]:
    Untuk setiap seed dalam [0, 1, 2, ..., 9]:
        1. Salin test set
        2. Pilih floor(dropout_rate × 75) keypoint secara acak (seeded)
        3. Set koordinat (x, y, z) keypoint tersebut → (0, 0, 0)
        4. Jalankan inferensi
        5. Catat accuracy
    
    Hitung mean dan std accuracy dari 10 seed
    Hitung DAR:
        DAR(d) = ((Accuracy_0% - Accuracy_d%) / Accuracy_0%) × 100%
```

### 9.3 Output yang Dibutuhkan untuk Karya Tulis

| Output | Deskripsi |
|---|---|
| Tabel 1 | Precision, recall, F1 per kelas pada kondisi normal |
| Tabel 2 | Accuracy (mean ± std) dan DAR per tingkat dropout |
| Gambar 1 | Confusion matrix kondisi normal (heatmap) |
| Gambar 2 | Grafik degradasi akurasi vs tingkat dropout (line chart dengan error bar) |
| Gambar 3 | Training curve: train_loss vs val_loss per epoch |

Simpan semua gambar sebagai PNG dengan `dpi=300`.

---

## BAGIAN 10 — BACKEND (`backend/`)

### 10.1 Sliding Window Buffer (`buffer.py`)

Gunakan `collections.deque(maxlen=60)`:
- Setiap frame masuk → append ke deque (frame terlama otomatis keluar)
- Jalankan inferensi setiap `frame_count % 15 == 0` DAN `len(buffer) == 60`
- Stride=15 → inferensi ~2 kali per detik pada 30fps

### 10.2 Inference Pipeline (`inference.py`)

```
Terima frame skeleton (75, 3)
    ↓
Normalisasi shoulder center (sama dengan saat training)
    ↓
Append ke sliding window buffer
    ↓
Jika stride terpenuhi dan buffer penuh:
    Stack buffer → tensor (60, 75, 3)
    Transpose → (3, 60, 75)
    Unsqueeze → (1, 3, 60, 75)
    Forward pass ST-GCN
    Softmax → confidence scores (3,)
    argmax → predicted class
    Return (class_name, confidence)
```

### 10.3 Confirmation Logic

```
predictions_buffer = deque(maxlen=5)

Setiap prediksi masuk:
    Append ke predictions_buffer
    
    Jika mode(predictions_buffer) muncul >= 4 kali
    DAN confidence prediksi terakhir > 0.85:
        → CONFIRMED
        → Emit ke frontend via WebSocket
        → Reset predictions_buffer
        → Aktifkan cooldown 10 detik
```

Cooldown mencegah trigger berulang untuk gesture yang sama.

### 10.4 FastAPI Endpoints (`main.py`)

**`WebSocket /ws`**
- Terima JSON dari frontend: `{ "keypoints": [[x,y,z], ...] }` (array 75×3)
- Proses melalui inference pipeline
- Kirim balik: `{ "class": "TOLONG", "confidence": 0.92, "is_confirmed": false }`

**`POST /notify`**
- Body: `{ "gesture": "TOLONG", "contact_number": "+628xxx", "user_name": "..." }`
- Trigger Twilio SMS + Voice
- Return: `{ "sms_status": "sent", "call_status": "initiated" }`

**`GET /health`**
- Return: `{ "status": "ok", "model_loaded": true }`

### 10.5 Notifikasi (`notification.py`)

**SMS via Twilio:**
```
"DARURAT: {user_name} membutuhkan bantuan segera.
Isyarat [{gesture}] terdeteksi pada [{timestamp}].
Hubungi segera."
```

**Voice Call via Twilio:**
- TwiML: `<Say language="id-ID">Peringatan darurat. {user_name} membutuhkan bantuan segera.</Say>`
- Rate limiting: trigger hanya jika gesture yang sama confirmed > 3 kali dalam 30 detik

**Push Notification:**
- Simpan push subscription saat user pertama membuka app
- Kirim menggunakan `pywebpush` ketika notifikasi triggered
- Service worker (`sw.js`) menangani display notifikasi di device

---

## BAGIAN 11 — FRONTEND (`frontend/`)

### 11.1 `useMediaPipe.js`

- Load MediaPipe Holistic saat mount
- Setup camera loop menggunakan `@mediapipe/camera_utils`
- Per frame: kirim ke MediaPipe → ekstrak hasil:
  - `results.poseLandmarks` → 33 pose landmarks
  - `results.leftHandLandmarks` → 21 left hand landmarks
  - `results.rightHandLandmarks` → 21 right hand landmarks
- Jika salah satu bagian null → isi dengan zeros
- Return flat array 75 keypoint `(75, 3)`

### 11.2 `useWebSocket.js`

- Connect ke `ws://localhost:8000/ws` saat mount
- Expose `sendFrame(keypoints)` — kirim koordinat ke backend
- Expose state `prediction` — hasil prediksi terbaru dari backend
- Handle reconnect otomatis jika koneksi terputus

### 11.3 `CameraView.jsx`

- Render `<video>` untuk webcam feed
- Render `<canvas>` overlay untuk skeleton
- Pass video ref ke `useMediaPipe`
- Setiap frame dari MediaPipe → `sendFrame(keypoints)`

### 11.4 `SkeletonOverlay.jsx`

- Gambar 75 keypoint sebagai lingkaran kecil di canvas
- Gambar edge sebagai garis penghubung
- Warna per bagian:
  - Pose: `rgba(0, 100, 255, 0.8)` — biru
  - Tangan kiri: `rgba(0, 200, 100, 0.8)` — hijau
  - Tangan kanan: `rgba(255, 50, 50, 0.8)` — merah
- Koordinat MediaPipe dalam rentang 0–1 → kalikan dengan lebar/tinggi canvas

### 11.5 `DetectionDisplay.jsx`

Status badge yang ditampilkan:

| Status | Kondisi | Warna |
|---|---|---|
| Mendeteksi... | Buffer belum penuh | Abu-abu |
| Terdeteksi: [KELAS] | Ada prediksi, belum confirmed | Kuning |
| DARURAT — Mengirim Notifikasi | Confirmed, notifikasi dikirim | Merah |
| Notifikasi Terkirim | Cooldown aktif | Hijau |

Tampilkan juga confidence bar (0–100%).

### 11.6 `EmergencyContact.jsx`

- Form input nama dan nomor telepon darurat
- Simpan ke `localStorage` — key: `emergency_contacts` (array)
- Tampilkan daftar kontak tersimpan dengan tombol hapus per kontak

### 11.7 Layout `App.jsx`

```
┌─────────────────────────────────────────┐
│    CameraView + SkeletonOverlay         │
│    (full width atas, ~60% tinggi)       │
├──────────────────┬──────────────────────┤
│ DetectionDisplay │ EmergencyContact     │
│ (kiri bawah)     │ (kanan bawah)       │
└──────────────────┴──────────────────────┘
```

---

## BAGIAN 12 — END-TO-END TESTING

Lakukan pengujian berikut secara berurutan setelah semua komponen selesai:

### Unit Tests
- Output shape model benar: input `(1, 3, 60, 75)` → output `(1, 3)`
- Augmentasi tidak mengubah shape tensor
- Normalisasi koordinat menghasilkan nilai terpusat di sekitar 0
- Adjacency matrix simetrik dan sudah ternormalisasi
- Sliding window buffer FIFO bekerja benar

### Integration Tests
- Buka browser → webcam aktif → skeleton muncul di overlay
- Lakukan isyarat TOLONG → dalam 3–5 detik status berubah ke confirmed
- SMS diterima di nomor darurat yang terdaftar
- Push notification muncul di device

### Robustness Demo (untuk dokumentasi penelitian)
- Tutup sebagian tangan dengan kertas → cek apakah sistem masih mendeteksi
- Lakukan isyarat dengan kecepatan berbeda → cek konsistensi prediksi
- Rekam hasil demo sebagai video untuk lampiran karya tulis

---

## RINGKASAN URUTAN PENGERJAAN

```
[1] Setup environment + struktur folder
    ↓
[2] Verifikasi gesture BISINDO (Pusbisindo / komunitas)
    ↓
[3] Rekam 120 video × 3 kelas dengan beragam subjek
    ↓
[4] Jalankan extract_keypoints.py → verifikasi 360 file .npy (60, 75, 3)
    ↓
[5] Jalankan augment.py → total 1.800 sampel
    ↓
[6] Implementasi graph.py → adjacency matrix (75, 75)
    ↓
[7] Implementasi model.py → ST-GCN 10 blok
    ↓
[8] Implementasi train.py → training + checkpoint
    ↓
[9] Implementasi evaluate.py → confusion matrix, F1, dll
    ↓
[10] Implementasi evaluate_dropout.py → DAR per tingkat dropout
     (catat semua angka → input untuk bab Hasil dan Pembahasan)
    ↓
[11] Implementasi backend FastAPI (main, inference, buffer, notification)
    ↓
[12] Implementasi frontend React (komponen + hooks)
    ↓
[13] Integrasi Twilio + push notification
    ↓
[14] End-to-end testing + rekam demo video
```

---

## CATATAN PENTING

> **Normalisasi:** Gunakan normalisasi shoulder center yang **identik** saat training dan saat inference real-time di backend. Inkonsistensi normalisasi adalah penyebab paling umum model yang bagus saat training tapi buruk saat real-time.

> **Augmentasi vs Split:** Augmentasi **hanya** diterapkan ke train set. Validation dan test set selalu data original.

> **Dropout Evaluation:** Jalankan hanya pada **checkpoint terbaik** setelah training selesai.

> **Twilio Trial:** Hanya bisa kirim ke nomor yang sudah diverifikasi di console Twilio. Verifikasi nomor tim sebelum testing.

> **Scope:** Ini proof-of-concept 3 kelas. Dokumentasikan semua keterbatasan secara eksplisit di bagian Simpulan dan Saran untuk memudahkan pengembangan selanjutnya.
