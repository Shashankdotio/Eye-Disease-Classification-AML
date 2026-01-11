# AML Project — Eye Disease Classification

## 👥 Group C — Team Members  
- **Shashank Kamble (NetID: sk3369)**  
- **Saumya Poojari (NetID: sp2877)**  
- **Jieying Wang (NetID: jw2088)**  

---

## 📘 Project Overview  
This project focuses on the **classification of eye diseases** using a combination of **deep feature extraction (MobileNetV2)**, **PCA**, and **classical machine learning models**.  
The four disease classes are:  

- Cataract  
- Diabetic Retinopathy  
- Glaucoma  
- Normal  

Dataset Source (Kaggle):  
[gunavenkatdoddi/eye-diseases-classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)

---

## 🚀 Pipeline Summary  

### **1. Data Cleaning**  
- Removed unreadable/corrupted images  
- Extracted image dimensions  
- Verified class distribution  
- Encoded labels into numeric IDs  
- Created stratified Train/Validation/Test splits  
- Saved cleaned metadata as `eye_disease_cleaned.csv`

---

### **2. Feature Extraction (MobileNetV2)**  
Performed in the development notebook (`shashank_dev.ipynb`):

- Loaded MobileNetV2 pretrained on ImageNet  
- Removed classification head (`include_top=False`)  
- Used `pooling='avg'` to obtain **1280-dimensional embeddings**  
- Passed all images through the network  
- Saved extracted feature vectors as `.npy` files  

Code excerpt (documented only, not executed in main):
```python
base_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
base_model.trainable = False

def extract_features(path):
    img = image.load_img(path, target_size=(224,224))
    arr = preprocess_input(np.expand_dims(image.img_to_array(img),0))
    return base_model.predict(arr).flatten()
```

---

### **3. PCA Dimensionality Reduction**
- Standardized features using `StandardScaler`
- Applied PCA retaining **95% variance**
- Reduced feature size (typically ~220 components)
- Saved PCA-transformed train/val/test sets  

---

### **4. Machine Learning Models Applied**
Models were trained using PCA features:

| Model | Status |
|-------|--------|
| Logistic Regression | ✅ Implemented |
| SVM (RBF Kernel) | ✅ Implemented |
| Random Forest | ✅ Implemented |
| MLP Neural Network | ✅ Implemented |
| **XGBoost (New Method)** | ✅ Implemented |

---

## 📊 Results Summary  
- **SVM (RBF)** and **XGBoost** showed the highest accuracy and macro F1 scores  
- **MLP** performed strongly and consistently  
- **Random Forest** and **Logistic Regression** were solid baselines  
- Classical ML models work extremely well when combined with MobileNetV2 embeddings and PCA  

---

## 📁 Project Structure  
```
AML_Eye_Disease_Classification/
│
├── data/
│   └── eye_disease_cleaned.csv
│
├── features/
│   ├── train_pca.npy
│   ├── val_pca.npy
│   ├── test_pca.npy
│   ├── train_labels.npy
│   ├── val_labels.npy
│   └── test_labels.npy
│
├── notebooks/
│   ├── main.ipynb
│   └── shashank_dev.ipynb
│
├── README.md
└── requirements.txt
```

---

## 👥 Team Contributions

### **Jieying Wang (jw2088)**  
- Handled data cleaning, unreadable image removal, and dataset organization  
- Conducted initial dataset inspection and analysis  
- Contributed to the written report and project presentation  

### **Shashank Kamble (sk3369)**  
- Implemented MobileNetV2 feature extraction  
- Performed PCA dimensionality reduction  
- Built and evaluated all ML models (LR, SVM, RF, MLP, XGBoost)  
- Contributed to methodology and results documentation  

### **Saumya Poojari (sp2877)**  
- Assisted in model implementation and verification  
- Helped write methodology and explain modeling decisions  
- Contributed to report writing and final presentation material  

---

## 📦 Requirements  
```
numpy
pandas
scikit-learn
xgboost
tensorflow
matplotlib
tqdm
Pillow
```

---

## ▶️ How to Run

### **1. Clone the Repository**
```
git clone https://github.com/your-username/AML_Eye_Disease_Classification.git
cd AML_Eye_Disease_Classification
```

### **2. Install Dependencies**
```
pip install -r requirements.txt
```

### **3. Open Notebook**
Run:

```
notebooks/main.ipynb
```

### **4. Execute All Cells**
Feature extraction and PCA are precomputed.  
Models load PCA features and run instantly.

---

## 🏁 Final Notes  
This project demonstrates how combining **deep learning–based feature extraction** with **classical machine learning** can produce strong results in medical image classification.

We thank Prof. Alizadeh for their support !
