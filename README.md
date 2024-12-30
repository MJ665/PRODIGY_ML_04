
---

# Hand Gesture Recognition using CNN

## Project Overview
This project develops a **Convolutional Neural Network (CNN)** model to recognize and classify hand gestures from images. The model is trained on the **LeapGestRecog** dataset and can be used for gesture-based human-computer interaction systems.

---

## Dataset
The dataset used is the **LeapGestRecog Dataset**, which contains various hand gestures captured under controlled conditions.

- **Source**: [LeapGestRecog Dataset on Kaggle](https://www.kaggle.com/gti-upm/leapgestrecog)
- The dataset consists of multiple gesture categories performed by different subjects, stored in organized folder structures.

### Steps to Use the Dataset:
1. Install and configure `kagglehub` to download datasets directly.
2. The dataset will be automatically downloaded and extracted to the specified path during script execution.

---

## Project Workflow

### 1. **Download and Extract the Dataset**
Using the `kagglehub` library, the dataset is downloaded and extracted to the local directory.

### 2. **Data Loading and Preprocessing**
- **Images**: Loaded and resized to `(64, 64)` pixels with 3 color channels.
- **Labels**: Each gesture folder is assigned a unique label, and the labels are one-hot encoded.
- **Normalization**: Image pixel values are scaled to the range `[0, 1]`.

### 3. **Data Augmentation**
To improve model generalization, training images are augmented using techniques like:
- Rotation
- Width and height shifts
- Shearing
- Zooming
- Horizontal flipping

### 4. **Model Architecture**
A CNN is designed with the following layers:
- Convolutional layers (with Batch Normalization and ReLU activation)
- Max-Pooling layers
- Dropout layers (to prevent overfitting)
- Fully connected dense layers
- Softmax output layer (for classification)

### 5. **Model Compilation and Training**
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy
- The model is trained for 25 epochs with a batch size of 32.

### 6. **Model Evaluation**
The model is evaluated using:
- **Test Accuracy**
- **Classification Report**
- **Confusion Matrix**

### 7. **Visualization**
- Loss and accuracy trends during training are visualized.
- Confusion matrix heatmap shows the model's performance across gesture categories.

---

## Dependencies

Install the required libraries using:
```bash
pip install numpy matplotlib tensorflow scikit-learn seaborn kagglehub
```

---

## Running the Code

1. Ensure you have the **LeapGestRecog** dataset downloaded or available for automatic download.
2. Run the script:
   ```bash
   python hand_gesture_recognition.py
   ```
3. The script will:
   - Download and preprocess the dataset.
   - Train the CNN on gesture images.
   - Evaluate the model and generate a classification report.
   - Visualize the confusion matrix.

---

## Results

### Metrics
- **Test Accuracy**: Achieved after training.
- **Classification Report**: Precision, recall, and F1-score for each gesture.

Example Output:
```
Test Accuracy: 0.95
Classification Report:
              precision    recall  f1-score   support

    Gesture1       0.95      0.96      0.95       200
    Gesture2       0.94      0.93      0.94       200
    ...
```

### Confusion Matrix
A confusion matrix heatmap shows the prediction results for each gesture.

---

## Notes and Future Work
1. **Improvement Ideas**:
   - Increase training epochs or fine-tune the architecture.
   - Use a pre-trained model like MobileNet or EfficientNet for feature extraction.
   - Experiment with additional data augmentation techniques.
2. **Real-Time Implementation**:
   - Extend the model to recognize gestures from live video streams using OpenCV.

