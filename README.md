# 🧠 Deepfake Image Detection using CNN

Welcome to the Deepfake Capstone Project! This repository is focused on detecting deepfake images using Convolutional Neural Networks (CNNs). The goal is to build a robust image classification model that can distinguish between real and manipulated (deepfake) facial images.

---

## 📚 Project Overview

Deepfake media is rising rapidly and poses a serious threat to digital trust. This project:
- Uses real vs. fake human face images from a Kaggle dataset.
- Builds a CNN model from scratch.
- Balances classes to avoid bias.
- Trains and evaluates the model with accuracy and loss plots.
- Aims to deploy this as a web-based detector.

---

## 📁 Dataset

Dataset: [Real and Fake Face Detection](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)  
- `training_real/` - contains real face images  
- `training_fake/` - contains fake face images generated using GANs

> ⚠️ Make sure to download the dataset using Kaggle API and unzip it inside the working directory.

---

## ⚙️ Model Architecture

- Input: 64x64 RGB Images
- Layers:
  - Conv2D → MaxPooling → BatchNorm
  - Dense layers with Dropout
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metrics: Accuracy

---

## 📊 Results

| Metric     | Value |
|------------|-------|
| Accuracy   | ~53% (initial model) |
| Loss       | 0.693 (imbalanced data) |

> Work in progress — model performance is expected to improve after balancing and hyperparameter tuning.

---

## 🔧 How to Run

1. Open [`Deepfake_Capstone_2.ipynb`](./Deepfake_Capstone_2.ipynb) in Google Colab.
2. Mount Google Drive & set Kaggle API.
3. Download dataset via Kaggle.
4. Train the CNN model using the pre-processed data.
5. Save and evaluate the model.

---

## 🚀 Next Steps

- Implement class balancing techniques.
- Tune hyperparameters and try pretrained models (e.g., Xception, ResNet).
- Deploy using Flask or Streamlit for web-based inference.
- Add LIME explainability for predictions.

---

## 👨‍💻 Author

- **Velmurugan**  
  MCA Student | AI Enthusiast | 💬 Loves building ML projects from scratch

---

## 🌐 Connect

Feel free to raise issues or contribute to this project. Let's detect deepfakes together!

