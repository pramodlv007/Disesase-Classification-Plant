
# Potato Disease Classification Using CNN

### **Project Overview**
This project focuses on identifying potato diseases, specifically early and late blight, using a Convolutional Neural Network (CNN). With an impressive **98% accuracy**, the model leverages deep learning techniques to classify disease conditions effectively.

---

### **Steps and Technologies Used**

#### **1. Importing Dependencies**
Imported necessary libraries like TensorFlow, Keras, Matplotlib, and NumPy to streamline development:
- TensorFlow/Keras: For model building and training.
- Matplotlib: For visualizing images and results.
- IPython: To display enhanced outputs.

---

#### **2. Dataset Preparation**
- **Source**: Dataset obtained from `PlantVillage` directory.
- **Image Processing**:
  - Resized all images to a fixed size of **256x256 pixels**.
  - Set batch size as **32** and ensured RGB color channel usage.
- **Splitting**: Divided the dataset into:
  - **Training (80%)**
  - **Validation (10%)**
  - **Testing (10%)**

Used `get_dataset_partitions_tf` for splitting and ensured shuffling for unbiased learning.

---

#### **3. Data Augmentation**
To improve model robustness, data augmentation was applied:
- **Random Flipping**: Horizontal and vertical transformations.
- **Random Rotation**: Rotated images to simulate real-world conditions.

---

#### **4. Preprocessing**
- Applied **Rescaling (1./255)** to normalize pixel values between 0 and 1.
- Cached, shuffled, and prefetched the dataset using `AUTOTUNE` for efficient data pipeline processing.

---

#### **5. Model Architecture**
Constructed a sequential CNN model with:
- **Input Layer**: Defined input shape as `(256, 256, 3)`.
- **Convolutional Layers**:
  - Multiple `Conv2D` layers for feature extraction (32, 64 filters).
  - Used **ReLU** activation for non-linearity.
- **MaxPooling**: Reducing spatial dimensions after each convolution.
- **Dense Layers**:
  - Flattened the feature map.
  - Added fully connected layers for classification.
- **Output Layer**: Used `softmax` activation for multi-class prediction.

---

#### **6. Compilation**
Configured the model with:
- **Optimizer**: Adam optimizer for adaptive learning rates.
- **Loss Function**: SparseCategoricalCrossentropy.
- **Metrics**: Accuracy to measure performance.

---

#### **7. Training**
Trained the model over **50 epochs**, achieving:
- Training Accuracy: **98%**
- Validation Accuracy: Robust generalization.

---

#### **8. Visualization**
Plotted training and validation metrics using Matplotlib:
- Accuracy curves to track model learning.
- Loss curves for monitoring convergence.

---

#### **9. Evaluation**
Evaluated model performance on the test dataset, confirming high accuracy.

---

#### **10. Predictions**
Implemented a function to make predictions:
- Identifies the disease condition with confidence percentages.
- Visualized predictions alongside actual labels.

---

#### **11. Deployment**
Saved the trained model as `model1.0.keras` for reuse and deployment. Future plans could include integration into a web or mobile application for farmers.

---

### **Key Takeaways**
- Leveraged **data augmentation** and preprocessing techniques to enhance generalization.
- Robust architecture ensures accurate disease identification in potatoes.
- With potential deployment, this project could transform agricultural diagnostics.


