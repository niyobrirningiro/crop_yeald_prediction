# Case Study 9: Crop Yield Prediction

**Student Name:** Niyobyiringiro Izere  
**Objective:** Build a machine learning solution that helps organizations make data-driven agricultural decisions by predicting crop yield outcomes based on environmental and management features.

---

## ── Project Overview

This end-to-end data science system handles data preprocessing, explores relationships, trains and evaluates three distinct machine learning algorithms, and deploys the highest-performing model via a local Flask web application dashboard.

### Target Label
* `yield_tons_per_hectare`

### Required Features
* **Numerical:** `rainfall`, `temperature`, `irrigation` (binary 0/1)
* **Categorical:** `soil_type`, `fertilizer_use`, `season`

---

## ── Project Folder Structure

```text
crop-yield-prediction/
│
├── data/
│   ├── raw/
│   │   └── crop_yield_dataset.csv       # Original dataset
│   └── processed/
│       ├── X_train.csv                  # Processed training features
│       ├── X_test.csv                   # Processed testing features
│       ├── y_train.csv                  # Training targets
│       └── y_test.csv                   # Testing targets
│
├── models/
│   ├── preprocessor.pkl                 # Saved OneHot/Scaler pipeline
│   ├── evaluation_results.pkl           # Saved model comparison metrics
│   └── best_crop_model.pkl              # Top-performing serialized ML model
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py                    # Cleans, encodes, and splits data
│   ├── train.py                         # Trains 3 models and selects the best
│   └── app.py                           # Flask backend API & dashboard controller
│
├── templates/
│   └── index.html                       # Frontend HTML Interface
│
├── requirements.txt                     # System Python dependencies
└── README.md                            # Documentation (This file)