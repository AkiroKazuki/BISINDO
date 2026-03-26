# 📘 BISINDO Emergency Sign Detection - Comprehensive Project Documentation

**Last Updated:** February 6, 2026  
**Project Status:** Planning & Design Phase  
**Target Completion:** March 2026

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Academic Context](#academic-context)
3. [Technical Architecture](#technical-architecture)
4. [Research Methodology](#research-methodology)
5. [Implementation Plan](#implementation-plan)
6. [Timeline & Milestones](#timeline--milestones)
7. [Key Decisions & Rationale](#key-decisions--rationale)
8. [Current Status](#current-status)

---

## 1. Project Overview

### 1.1 Project Title

**"Pengembangan Sistem Deteksi Isyarat Darurat BISINDO Berbasis Arsitektur ST-GCN dengan Ketahanan Terhadap Variasi Pencahayaan dan Oklusi Parsial"**

### 1.2 Problem Statement

- **Population:** 22.97 million deaf/hard-of-hearing individuals in Indonesia (BPS 2023)
- **Communication Gap:** General public doesn't understand sign language, creating critical barriers in emergency situations
- **EWS Limitations:** Current Early Warning Systems are audio-based, inaccessible to deaf community
- **Disaster Context:** Indonesia experiences 5,000+ earthquakes annually plus frequent floods

### 1.3 Solution

A web-based real-time sign language detection system that acts as a **communication bridge** between deaf/mute individuals and the general public, specifically for emergency situations.

### 1.4 Core Features

1. **Real-time Detection:** 10 emergency sign classes (TOLONG, BAHAYA, KEBAKARAN, SAKIT, GEMPA, BANJIR, PENCURI, PINGSAN, KECELAKAAN, DARURAT)
2. **Automatic Notifications:** Text-to-Speech (TTS) + SMS/Push notifications to emergency contacts
3. **Robustness:** Tested against lighting variations and partial occlusion
4. **Mobile Accessible:** Web application accessible from smartphones
5. **Pre-training:** Enhanced with SIBI alphabet dataset (26 classes) for better generalization

---

## 2. Academic Context

### 2.1 Research Type

- **Category:** Eksakta (Exact Sciences/Engineering)
- **Method:** Research & Development (R&D)
- **Model:** ADDIE (Analysis, Design, Development, Implementation, Evaluation)

### 2.2 Research Structure (Extended Abstract Format)

#### BAB I: PENDAHULUAN

- **A.1 Latar Belakang:** Disability statistics, communication barriers, emergency context, technology gap
- **A.2 Tujuan Penelitian:** 4 objectives (system development, lighting robustness, occlusion robustness, notification integration)
- **A.3 Manfaat Penelitian:** Practical (emergency communication) + Theoretical (BISINDO benchmark)

#### BAB III: METODE PENELITIAN

- **C.1 Waktu & Tempat:** January-March 2026, SMAN 3 Denpasar
- **C.2 Jenis Penelitian:** R&D with ADDIE model
- **C.3 Variabel Penelitian:**
  - Independent: Lighting conditions, occlusion levels, sign types
  - Dependent: Accuracy, latency, degradation
  - Control: Distance, hardware specs, video resolution
- **C.4 Rancangan Penelitian:** 5-phase flowchart (Analysis → Design → Development → Implementation → Evaluation)
- **C.5 Alat & Bahan:** Hardware (laptop, webcam) + Software (Python, PyTorch, MediaPipe, FastAPI, React)
- **C.6 Metode Pengumpulan Data:**
  - Quantitative: Landmark coordinates, accuracy metrics, lux measurements, FPS
  - Qualitative: Functional testing, visual observation
- **C.7 Teknik Analisis Data:** Accuracy, Precision, Recall, F1-Score, Robustness degradation

### 2.3 Requirements (Template Extended Abstract 2025)

- **Minimum Pages:** 10 (including appendices)
- **Abstract:** Max 250 words (Indonesian + English)
- **Keywords:** 3-5 words
- **References:** APA Style, prioritize journals from last 5 years
- **Appendices:** Documentation, data, instruments, biodata

---

## 3. Technical Architecture

### 3.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (React + Vite)                       │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌─────────┐           │
│  │ Camera  │ │ Result   │ │ Skeleton  │ │ Alert   │           │
│  │ View    │ │ Display  │ │ Overlay   │ │ Panel   │           │
│  └────┬────┘ └──────────┘ └───────────┘ └─────────┘           │
└───────┼────────────────────────────────────────────────────────┘
        │ WebSocket (Real-time frames)
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                   BACKEND (FastAPI + PyTorch)                    │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐                │
│  │ MediaPipe │ → │  ST-GCN   │ → │ Classifier│                │
│  │ Holistic  │   │  Encoder  │   │ (36 class)│                │
│  └───────────┘   └───────────┘   └─────┬─────┘                │
│                                        │                       │
│  ┌─────────────────────────────────────┴───────────────────┐  │
│  │ Output: TTS (pyttsx3) | SMS (Twilio) | Push (Telegram)  │  │
│  └─────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Model Architecture: ST-GCN

**Why ST-GCN over LSTM/TCN?**

| Aspect | LSTM | TCN | ST-GCN ⭐ |
|--------|------|-----|----------|
| Accuracy | 92-97% | 90-95% | **93-98%** |
| Speed | Slow | Fast | Fast |
| Spatial Modeling | ❌ | ❌ | ✅ Built-in |
| Skeleton Suitability | Medium | Medium | **Excellent** |
| Parallelization | ❌ Sequential | ✅ Parallel | ✅ Parallel |

**ST-GCN Pipeline:**

```
Input Video (B, T, H, W, 3)
    ↓
MediaPipe Holistic → 75 Keypoints (33 pose + 21×2 hands)
    ↓
Graph Construction (Adjacency Matrix)
    ↓
ST-GCN Blocks (×10)
├─ Spatial Graph Conv (A × X × W)
├─ Temporal Conv (kernel=9)
├─ BatchNorm + ReLU + Dropout
└─ Residual Connection
    ↓
Global Average Pooling
    ↓
Classifier → 36 classes (26 SIBI + 10 Emergency)
    ↓
Output: Class + Confidence + Alert Trigger
```

### 3.3 Dataset Strategy

**Multi-Task Learning:**

1. **Pre-training:** SIBI Alphabet (A-Z) = 26 classes from Kaggle
2. **Fine-tuning:** Emergency Signs = 10 classes (custom recorded)
3. **Total:** 36 classes for joint training

**Emergency Sign Classes:**

1. TOLONG (Help)
2. BAHAYA (Danger)
3. KEBAKARAN (Fire)
4. SAKIT (Sick)
5. GEMPA (Earthquake)
6. BANJIR (Flood)
7. PENCURI (Thief)
8. PINGSAN (Unconscious)
9. KECELAKAAN (Accident)
10. DARURAT (Emergency)

### 3.4 Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Model Framework** | PyTorch | 2.0+ |
| **Keypoint Extraction** | MediaPipe Holistic | 0.10+ |
| **Backend** | FastAPI | 0.100+ |
| **Real-time Comm** | WebSocket (python-socketio) | 5.0+ |
| **TTS** | pyttsx3 | 2.90+ |
| **SMS** | Twilio API | 8.0+ |
| **Push Notifications** | Telegram Bot API | 20.0+ |
| **Frontend** | React 18 + Vite | 5.0+ |
| **Styling** | TailwindCSS | 3.0+ |
| **Database** | SQLite | 3.0+ |

---

## 4. Research Methodology

### 4.1 Variables

**Independent Variables (Manipulated):**

- Lighting: Bright (>300 lux), Normal (100-300), Dim (50-100), Dark (<50)
- Occlusion: None (0%), Light (<25%), Medium (25-50%)
- Sign Type: 10 emergency classes

**Dependent Variables (Measured):**

- Detection Accuracy (%)
- Inference Latency (ms)
- Accuracy Degradation (%)

**Control Variables (Constant):**

- User distance: 1-2 meters
- Hardware: Same laptop + webcam
- Video resolution: 720p

### 4.2 Data Collection

**Quantitative:**

- Landmark coordinates (x, y, z)
- Performance metrics (Accuracy, Precision, Recall, F1)
- Environmental data (Lux measurements)
- Computational metrics (Latency, FPS)

**Qualitative:**

- Functional testing (SMS delivery, TTS clarity)
- Visual observation (Skeleton overlay stability)

### 4.3 Evaluation Metrics

1. **Accuracy:** `(TP + TN) / (TP + TN + FP + FN) × 100%`
2. **Precision:** `TP / (TP + FP)`
3. **Recall:** `TP / (TP + FN)`
4. **F1-Score:** `2 × (Precision × Recall) / (Precision + Recall)`
5. **Degradation Rate:** `(Acc_normal - Acc_condition) / Acc_normal × 100%`

### 4.4 Robustness Testing

**Target Performance:**

| Condition | Lux Range | Target Accuracy |
|-----------|-----------|-----------------|
| Bright | >300 | ≥95% |
| Normal | 100-300 | ≥90% |
| Dim | 50-100 | ≥85% |
| Dark | <50 | ≥75% |

| Occlusion Level | Target Accuracy |
|-----------------|-----------------|
| 0% | ≥95% |
| <25% | ≥85% |
| 25-50% | ≥75% |

---

## 5. Implementation Plan

### 5.1 Project Structure

```
BISINDO/
├── backend/              # FastAPI Server
│   ├── app/
│   │   ├── main.py
│   │   ├── api/          # Endpoints (prediction, websocket, notifications)
│   │   ├── models/       # ST-GCN implementation
│   │   └── services/     # MediaPipe, TTS, SMS
│   └── checkpoints/
├── frontend/             # React App
│   └── src/
│       ├── components/   # Camera, ResultPanel, SkeletonOverlay, AlertPanel
│       ├── hooks/        # useWebSocket
│       └── pages/        # Home, Settings
├── training/             # Model Training
│   ├── train_stgcn.py
│   ├── evaluate.py
│   └── configs/
├── data/
│   ├── sibi_alphabet/    # Kaggle dataset
│   ├── emergency_signs/  # Custom recordings
│   └── processed/
└── docs/
    ├── API.md
    └── DEMO.md
```

### 5.2 Development Phases (ADDIE)

**Phase 1: Analysis (Week 1-2)**

- Literature review (ST-GCN, BISINDO, emergency systems)
- Define 10 emergency sign classes
- Identify dataset sources

**Phase 2: Design (Week 2-3)**

- ST-GCN architecture design
- UI/UX mockups
- Notification flow logic

**Phase 3: Development (Week 3-8)**

- Data collection (5-10 subjects, multiple angles)
- MediaPipe preprocessing
- ST-GCN training (target: ≥90% accuracy)
- Web app development (FastAPI + React)
- Notification integration (TTS, SMS, Telegram)

**Phase 4: Implementation (Week 9-10)**

- Real-time testing in controlled environment
- Emergency scenario simulation

**Phase 5: Evaluation (Week 11-12)**

- Robustness testing (lighting + occlusion)
- Performance analysis
- Documentation & paper writing

---

## 6. Timeline & Milestones

| Month | Week | Milestone |
|-------|------|-----------|
| **January 2026** | 1-2 | ✅ Literature review, scope definition |
| | 3-4 | Data collection begins |
| **February 2026** | 1-2 | Model training (SIBI pre-training) |
| | 3-4 | Fine-tuning on emergency signs |
| **March 2026** | 1-2 | Web app development + integration |
| | 3-4 | Testing, analysis, paper writing |

**Deadline:** March 2026 (Extended Abstract submission)

---

## 7. Key Decisions & Rationale

### 7.1 Architecture: ST-GCN vs LSTM/TCN

**Decision:** ST-GCN  
**Rationale:**

- Designed specifically for skeleton-based action recognition
- Combines spatial (joint relationships) and temporal (motion) in one framework
- More efficient than LSTM (no vanishing gradient issues)
- State-of-the-art for gesture recognition (5000+ citations)

### 7.2 Platform: Web Application

**Decision:** Server-based inference with React frontend  
**Rationale:**

- Mobile accessible without app installation
- Easier deployment and updates
- Better computational resources on server
- Cross-platform compatibility

### 7.3 Dataset: SIBI + Emergency Signs

**Decision:** Transfer learning approach  
**Rationale:**

- SIBI alphabet provides general sign language understanding
- Emergency signs benefit from pre-trained representations
- Increases model generalization capability

### 7.4 Scope: Emergency Focus

**Decision:** 10 emergency signs (not full BISINDO vocabulary)  
**Rationale:**

- Addresses critical safety need
- Manageable for SMA-level research timeline
- Clear evaluation criteria
- High social impact

---

## 8. Current Status

### 8.1 Completed

- ✅ Project scope definition
- ✅ Architecture selection (ST-GCN)
- ✅ Research methodology design
- ✅ Bab 1 Pendahuluan (draft)
- ✅ Bab 3 Metode Penelitian (complete)
- ✅ Implementation plan
- ✅ Timeline establishment

### 8.2 In Progress

- 🔄 Literature review refinement
- 🔄 Dataset identification (SIBI Kaggle)
- 🔄 Academic paper writing

### 8.3 Pending

- ⏳ ST-GCN implementation
- ⏳ Data collection (emergency signs)
- ⏳ Model training
- ⏳ Web application development
- ⏳ Robustness testing
- ⏳ Final paper submission

---

## 9. References & Resources

### 9.1 Key Papers

- Zhao & Chen (2023): ST-GCN for regression and feature extraction
- Lu et al. (2023): Sign language recognition with multimodal sensors
- Yan et al. (2018): Original ST-GCN paper (action recognition)

### 9.2 Datasets

- SIBI Alphabet: Kaggle (26 classes, A-Z)
- Emergency Signs: Custom recording (10 classes)

### 9.3 Tools & Libraries

- MediaPipe: <https://google.github.io/mediapipe/>
- PyTorch: <https://pytorch.org/>
- FastAPI: <https://fastapi.tiangolo.com/>
- React: <https://react.dev/>

---

## 10. Contact & Team

**Institution:** SMA Negeri 3 Denpasar  
**Project Type:** Karya Tulis Ilmiah (Scientific Paper)  
**Category:** Eksakta (Exact Sciences)  
**Timeline:** January - March 2026

---

**Document Version:** 1.0  
**Last Updated:** February 6, 2026  
**Status:** Living Document (Updated as project progresses)
