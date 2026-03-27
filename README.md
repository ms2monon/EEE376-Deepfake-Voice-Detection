# EEE376-Deepfake-Voice-Detection
Detection of AI-Generated Speech from Real Human Voice — EEE 376 DSP Project
# EEE 376 — Detection of AI-Generated Speech from Real Human Voice

**Course:** Digital Signal Processing (EEE 376), BUET  
**Section:** [Your Section] | **Group:** [Your Group Number]

**Team Members:**
| Student ID | Name |
|---|---|
| 2118027 | Monon Mohammad Sadim Sami |
| 2118028 | Upama Rani Roy |
| 2118029 | Nabil Faruque Rafin |
| 2118034 | Abir Ahammed Khan Swakhor |
| 2118041 | Suraiya Sujana Khan |

---

## Project Overview
A MATLAB-based system for detecting AI-generated (deepfake) voice
recordings from real human speech, using an 88-dimensional acoustic
feature vector and Random Forest / SVM classifiers trained on a
custom dataset of 60 BUET student speakers.

## Key Results
| Classifier | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Random Forest (200 trees) | 94.4% | 94.4% | 94.4% | 94.4% | 0.979 |
| SVM RBF Kernel | 88.9% | 96.7% | 80.6% | 87.9% | 0.974 |

- **LOSO Cross-Validation (RF):** 89.4% mean accuracy across 60 speakers

---

## Repository Structure
```
EEE376-Deepfake-Voice-Detection/
│
├── README.md
├── code/
│   ├── audioPreprocess.m
│   ├── Build_Dataset_Table_1.m
│   ├── Feature_Extraction_2.m
│   ├── EDA_Comparison_Analysis_3.m
│   ├── Classifier_4.m
│   ├── extractSpeakerIDs.m
│   ├── Deepfake_Detector_Live_Main_5.m
│   └── Deepfake_Detector_Live_Main_6_SVM.m
├── model/
│   ├── best_rf_model.mat
│   └── best_svm_model.mat
└── Dataset/
    ├── real/   ← 180 real voice recordings (60 speakers × 3 files)
    └── fake/   ← 180 AI-generated recordings (60 speakers × 3 files)
```

---

## Dataset Description
- **Total files:** 360 (perfectly balanced — 180 real, 180 fake)
- **Speakers:** 60 BUET students (36 male, 24 female)
- **Real audio:** Each speaker recorded 3 clips following a standardized
  script, on personal devices in natural indoor environments
- **Fake audio:** Generated using online AI voice cloning tools and
  text-to-speech synthesis of the same script
- **Format:** 16-bit WAV, mono, 16 kHz
- **Naming convention:**
  - Real: `[SpeakerName] script [1/2/3].wav`
  - Fake: `[SpeakerName] [1/2/3] (Fake).wav`

---

## Files Description

### MATLAB Source Code (`code/`)
| File | Purpose |
|---|---|
| `Build_Dataset_Table_1.m` | Step 1 — Scans dataset folder and builds file table |
| `Feature_Extraction_2.m` | Step 2 — Extracts 88-dimensional features from all audio |
| `EDA_Comparison_Analysis_3.m` | Step 3 — Exploratory data analysis and plots |
| `Classifier_4.m` | Step 4 — Trains and evaluates RF and SVM classifiers |
| `audioPreprocess.m` | Helper — Preprocessing pipeline (called automatically) |
| `extractSpeakerIDs.m` | Helper — Parses speaker IDs from filenames (called automatically) |
| `Deepfake_Detector_Live_Main_5.m` | Live detector using Random Forest |
| `Deepfake_Detector_Live_Main_6_SVM.m` | Live detector using SVM |

### Pre-trained Models (`model/`)
| File | Description |
|---|---|
| `best_rf_model.mat` | Trained Random Forest model — 94.4% accuracy, AUC 0.979 |
| `best_svm_model.mat` | Trained SVM RBF model — 88.9% accuracy, AUC 0.974 |

---

## Quick Start — Live Detector (No Retraining Needed)

1. Download this repository — click the green **Code** button → **Download ZIP**
   and extract it anywhere on your computer
2. Copy these 3 files into the same folder:
   - `code/Deepfake_Detector_Live_Main_5.m`
   - `code/audioPreprocess.m`
   - `model/best_rf_model.mat`
3. Open MATLAB and navigate to that folder
4. Type in the Command Window and press Enter:
```
Deepfake_Detector_Live_Main_5
```
5. When prompted:
   - Enter `1` — record 4 seconds from your microphone
   - Enter `2` — select an audio file (.wav .mp3 .m4a .flac)
   - Enter `0` — exit

---

## Full Pipeline — Retrain from Scratch

Only needed if you want to retrain on a new dataset.

1. Place audio files into `Dataset/real/` and `Dataset/fake/`
   following the naming convention above
2. Open `Build_Dataset_Table_1.m`, update `datasetRoot` to point
   to your `Dataset/` folder, then run it
3. Run `Feature_Extraction_2.m`
4. Run `EDA_Comparison_Analysis_3.m` (optional — for analysis plots)
5. Run `Classifier_4.m` to train and evaluate both models

---

## Requirements
- MATLAB R2022b or later
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox
