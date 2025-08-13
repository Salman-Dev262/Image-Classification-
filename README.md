# CIFAR-10 Image Classification using CNN  

## 📌 Overview  
This project implements an **Image Classification model** using **Convolutional Neural Networks (CNNs)** to classify images from the **CIFAR-10 dataset**, which contains 60,000 low-resolution (32×32) RGB images in **10 distinct categories**:  
`Airplane`, `Automobile`, `Bird`, `Cat`, `Deer`, `Dog`, `Frog`, `Horse`, `Ship`, and `Truck`.  

Using **Python** with **TensorFlow/Keras**, a custom CNN architecture was built to automatically extract spatial features from images and predict their class. The model was trained on **50,000 images**, validated on **10,000 images**, and tested on **10,000 unseen images**, achieving **~82% test accuracy** after hyperparameter tuning.  

---

## 📂 Dataset Description  
- **Source:** CIFAR-10 (available via `keras.datasets`)  
- **Size:** 60,000 images (50,000 train, 10,000 test)  
- **Format:** 32×32 pixel RGB images  
- **Classes:** 10 balanced categories  

| Label ID | Class Name  | Train Samples | Test Samples |
|----------|-------------|---------------|--------------|
| 0        | Airplane    | 5,000         | 1,000        |
| 1        | Automobile  | 5,000         | 1,000        |
| 2        | Bird        | 5,000         | 1,000        |
| 3        | Cat         | 5,000         | 1,000        |
| 4        | Deer        | 5,000         | 1,000        |
| 5        | Dog         | 5,000         | 1,000        |
| 6        | Frog        | 5,000         | 1,000        |
| 7        | Horse       | 5,000         | 1,000        |
| 8        | Ship        | 5,000         | 1,000        |
| 9        | Truck       | 5,000         | 1,000        |

---

## 🔄 Data Preprocessing  
- **Normalization:** Pixel values scaled from [0, 255] → [0, 1]  
- **One-Hot Encoding:** Class labels converted for categorical crossentropy loss  
- **Train-Validation Split:** 80% train / 20% validation from training set  
- **Visualization:** Sample images displayed for verification of correct labels  

---

## 🏗 Model Architecture  
The CNN architecture consists of:  
- **Input Layer** → 32×32×3 images  
- **Convolutional Layers** → Extract low and high-level features  
- **MaxPooling Layers** → Reduce spatial dimensions  
- **Batch Normalization** → Stabilize learning  
- **Dropout Layers** → Prevent overfitting  
- **Dense Layers** → Combine features for classification  
- **Output Layer (Softmax)** → 10 units for CIFAR-10 classes  

---

## ⚙ Training Strategy  
- **Optimizer:** Adam (learning rate = 0.001 with scheduling)  
- **Loss Function:** Categorical Crossentropy  
- **Epochs:** 25–50  
- **Batch Size:** 32 or 64  
- **Data Augmentation:**  
  - Horizontal flipping  
  - Random shifts  
  - Minor rotations (±15°)  
- **Callbacks:**  
  - EarlyStopping (patience = 5)  
  - ReduceLROnPlateau  
  - ModelCheckpoint  

---

## 📊 Results & Evaluation  
- **Training Accuracy:** ~85%  
- **Validation Accuracy:** ~84%  
- **Test Accuracy:** ~82%  
- **Loss (Test):** 0.4–0.6  
- **Confusion Matrix:** Best performance on classes like Airplane and Frog; some confusion between visually similar classes like Cat and Dog  

📈 **Performance Plots:** Accuracy & loss curves over training epochs included in `/plots` directory.  

---

## ✅ Successes  
- CNN architecture learned robust features  
- Data augmentation improved generalization  
- Callback functions optimized training efficiency  


```

git clone https://github.com/Salman-Dev262/image_Classifier.git
cd image_Classifier
pip install -r requirements.txt
python train.py

```
## 🚀 Future Work  
- Implement deeper architectures (ResNet, VGG, DenseNet)  
- Apply transfer learning with pre-trained models  
- Perform hyperparameter tuning  
- Deploy via Flask/Streamlit web app  
- Extend to CIFAR-100 or custom datasets  

---

## 🛠 Technologies Used  
- **Python 3.10+**  
- **TensorFlow/Keras**  
- **NumPy, Pandas**  
- **Matplotlib, Seaborn**  

---

## 📜 References  
- [Keras Documentation](https://keras.io/)  
- [TensorFlow Official Site](https://www.tensorflow.org/)  
- [Kaggle CIFAR-10 Dataset](https://www.kaggle.com/c/cifar-10)  
- François Chollet, *Deep Learning with Python*  

---

## 📄 License  
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  
