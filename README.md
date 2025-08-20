# 👤👓👕 Multi-Task CelebA Classifier (Gender, Glasses & Shirt Color Detection)

This project is a **multi-task deep learning system** that can:  
- Detect **gender** (Male / Female)  
- Detect **glasses** (With Glasses / No Glasses)  
- Detect **shirt color** (dominant shirt color using KMeans + LAB color space)  

using the **CelebA dataset**.

---

## 📂 Dataset
We use the **[CelebA dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)** available on Kaggle.  
- **CelebA Attributes CSV** (`list_attr_celeba.csv`) is used to fetch gender (`Male`) and glasses (`Eyeglasses`) labels.  
- Images are preprocessed and split into **train, validation, and test sets**.  

---

## ⚙️ Project Pipeline

1. **Data Preprocessing**  
   - Attributes cleaned (`-1 → 0`)  
   - Split into train/val/test (stratified by gender)  
   - Images resized to `224x224`, normalized  

2. **Model Architecture**  
   - **Backbone**: Pre-trained **ResNet-18** (ImageNet weights)  
   - **Multi-task heads**:  
     - FC layer for **Gender Classification**  
     - FC layer for **Glasses Classification**

3. **Training**  
   - Loss = `CrossEntropyLoss(Gender) + CrossEntropyLoss(Glasses)`  
   - Optimizer: **Adam (lr=1e-4)**  
   - Trained for 5 epochs (can be extended)  

4. **Evaluation**  
   - Metrics: **Accuracy + Confusion Matrix**  
   - Achieved:  
     - ✅ **Gender Accuracy**: ~98.6%  
     - ✅ **Glasses Accuracy**: ~99.6%  

5. **Shirt Color Detection**  
   - Crops shirt region (bottom-mid part of image)  
   - Runs **KMeans clustering** on pixels  
   - Converts cluster center to **LAB color space**  
   - Matches with predefined color centroids (`Red, Blue, Green, Black, White, etc.`)  

---

## 🚀 Results

- **High accuracy** for both Gender and Glasses detection.  
- **Shirt color prediction** works well with predefined LAB centroids.  
- Fast **inference time**:  
  - Gender & Glasses prediction: ~20–40 ms  
  - Shirt color detection: ~50–70 ms  

---

# 📦 Requirements

- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- numpy
- pandas
- OpenCV
- tqdm
