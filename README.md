# 🎙️ Voice Emotion Recognition System

## 🌍 Overview

This project focuses on building a **Voice Emotion Recognition** system using **audio signal processing** and **machine learning techniques**. It aims to classify human emotions such as *happy, sad, angry, fearful, neutral*, etc., by analyzing speech recordings.

---

## 📄 Dataset Description

The dataset used in this project contains `.wav` audio files labeled according to different emotional states. Each audio file is a recording of a human voice expressing a specific emotion. Example datasets include:

* **TESS** (Toronto Emotional Speech Set)
* **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
* **SAVEE** (Surrey Audio-Visual Expressed Emotion)

These datasets consist of files organized by emotion categories.

---

## 📊 Feature Extraction

The following audio features are extracted using the **`librosa`** library:

1. **MFCC (Mel-Frequency Cepstral Coefficients)**: Captures the timbre of the audio.
2. **Chroma Frequencies**: Reflects pitch class content.
3. **Mel Spectrogram**: Captures short-term power spectrum.
4. **RMS Energy**: Measures energy of the signal.
5. **Zero Crossing Rate**: Frequency of sign changes in signal.

These features are concatenated to form the final input feature vector for training.

---

## 🧱 Model Building

Several machine learning models are tested and evaluated, including:

* **Random Forest Classifier**
* **XGBoost Classifier**
* (SVM, KNN can also be incorporated optionally)

### Training Steps:

1. **Splitting the dataset** into training and testing sets
2. **Fitting the model** on the extracted features
3. **Evaluating accuracy** on test data

---

## 📊 Performance Metrics

The following metrics are used to evaluate the model:

* **Accuracy**
* **Confusion Matrix**
* **Precision, Recall, F1-score** (via classification report)

---

## 🔧 How to Run

### 1. Environment Setup

Install required libraries:

```bash
pip install numpy pandas librosa scikit-learn matplotlib seaborn xgboost soundfile
```

### 2. Notebook Execution

* Open the `voice_emotion.ipynb` file in Jupyter Notebook
* Run all cells sequentially
* The model will train and show evaluation results

---

## 🔄 Future Improvements

* Integration of **Deep Learning models** like CNN, RNN, LSTM for better accuracy
* Deployment of the system via **Streamlit or Flask Web App**
* Real-time emotion detection from **microphone input**

---

## 📅 Project Structure

```
voice_emotion_project/
├── voice_emotion.ipynb
├── audio_data/
│   └── *.wav
├── extracted_features.csv
├── model/
│   └── trained_model.pkl
└── README.md
```

---

## 🙋‍♂️ Contributors

* Developer: Savan Patel
* Contact: *[savanpatel0208@gmail.com](mailto:your.email@example.com)*

---

