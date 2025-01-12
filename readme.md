# PneumoGAN: A Variant of Generative Adversarial Network for Pneumonia Detection

## Project Overview

**Project Title**: PneumoGAN: A Variant of Generative Adversarial Network for Pneumonia Detection  
**Level**: Advanced  
**Dataset**: Chest X-ray dataset from NIH Clinical Center (https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

This project focuses on leveraging a novel variant of Generative Adversarial Networks (GANs), known as PneumoGAN, to improve biomedical imaging accuracy in detecting pneumonia. PneumoGAN aims to enhance pneumonia detection accuracy, achieving an impressive 94.99% detection rate, demonstrating the advancements of GANs in the field of medical image analysis.

## Objectives

1. **Data Preprocessing and Augmentation**: Clean and augment the Chest X-ray dataset to make it suitable for training the model.
2. **PneumoGAN Model Development**: Develop the PneumoGAN model for pneumonia detection using GAN-based techniques.
3. **Model Training and Evaluation**: Train the model, evaluate its performance, and visualize the results.
4. **Synthetic Data Generation**: Use GANs to generate synthetic data, improving the training process and model performance.
5. **Image Classification**: Integrate deep learning techniques to classify images as pneumonia-positive or pneumonia-negative.

## Project Structure

### 1. Dataset and Preprocessing

- **Dataset Source**: The project uses the Chest X-ray dataset sourced from the NIH Clinical Center, which contains both pneumonia-positive and pneumonia-negative cases.
- **Preprocessing**: The dataset is preprocessed to resize the images, normalize pixel values, and augment data to create a robust model.
  
```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load dataset
train_data_gen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, 
                                    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, 
                                    horizontal_flip=True, fill_mode='nearest')

# Data augmentation for training set
train_data = train_data_gen.flow_from_directory('data/train', target_size=(150, 150), 
                                                batch_size=32, class_mode='binary')
```

### 2. Model Development

- **PneumoGAN**: The core of the project is the PneumoGAN model, a specialized GAN designed for medical image classification tasks.
- **Model Architecture**: It consists of a Generator and a Discriminator. The Generator creates synthetic images, and the Discriminator distinguishes between real and fake images.
- **Training the Model**: The model is trained on a dataset of chest X-ray images for detecting pneumonia.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, LeakyReLU

# Example of Discriminator architecture
discriminator = Sequential([
    Conv2D(64, (3, 3), input_shape=(150, 150, 3)),
    LeakyReLU(alpha=0.2),
    Flatten(),
    Dense(1, activation='sigmoid')
])
```

### 3. Results & Evaluation

- **Accuracy**: PneumoGAN achieved an accuracy of 94.99% on the task of pneumonia detection.
- **Synthetic Data Generation**: The GAN model generated synthetic chest X-ray images to augment the dataset and improve model robustness.
- **Evaluation Metrics**: Various performance metrics, including precision, recall, and F1 score, are evaluated for pneumonia detection.

```python
# Example of model evaluation
test_loss, test_acc = model.evaluate(test_data)
print(f"Test accuracy: {test_acc * 100}%")
```

### 4. File Structure

- **pneumogan_model.ipynb**: Jupyter Notebook containing the full implementation of the PneumoGAN model, including data preprocessing, model architecture, training, and evaluation.
- **data/**: Contains preprocessed chest X-ray datasets (replace with your dataset).
- **results/**: Contains generated synthetic images and evaluation metrics.

#### Getting Started
##### 1. Prerequisites
- Python 3.x
- Jupyter Notebook
- Required libraries (install via pip):

```bash
pip install tensorflow keras numpy matplotlib scikit-learn opencv-python
```
##### 2. Usage    
- Clone the repository:
```bash
git clone https://github.com/vaishakb251/MyProjects.git
```
- Navigate to the PneumoGAN directory:
```bash
cd MyProjects/PneumoGAN
```
- Open the notebook:
```bash
jupyter notebook pneumogan_model.ipynb
```
- Replace the data/ directory with your own chest X-ray dataset if necessary.
- Follow the steps in the notebook to train and evaluate the PneumoGAN model.

### Acknowledgements

- Dataset: The Chest X-ray dataset is sourced from the NIH Clinical Center.
- This project highlights the potential of GANs to improve medical diagnostics by providing a deep learning-based solution for pneumonia detection.
