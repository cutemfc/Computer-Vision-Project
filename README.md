# Computer-Vision-Project
# 🎑Computer-Vision-Project (Convolutional Neural Network)
### Data of this project is download from CIFAR-10 dataset


### 🌍 Goal:
This project aims to identify 10 class objects using convolutional Neural Network modelling. 

### 🌍Descriptions:
This project focuses on multi-class image classification using the CIFAR-10 dataset, which contains 60,000 color images (32x32 pixels) spanning 10 mutually exclusive classes. The dataset is split into 10,000 training images and 10,000 test images for experimental consistency. A custom classification head was created and combined with a pre-trained ResNet50 model (CNN architecture) to leverage the power of transfer learning. The ResNet50 base model was initially frozen, and later selectively fine-tuned for performance enhancement.

### 🌍 Experiments Conducted:
1.Increased Dropout Layers to reduce overfitting and improve generalization.
 
2.Tuned Regularization Weights (e.g., L2 penalties) to optimize learning and reduce complexity.

3.Fine-Tuned Model Layers by unfreezing parts of the ResNet50 base model and training additional layers to improve feature learning.




### 🌍Skills:


#### Skills (CNN & ResNet50 Focused):
1.Image classification using CNN and ResNet50 (transfer learning)

2.Model building and training with Keras (Conv2D, MaxPooling2D, Dense)

3.Fine-tuning and layer freezing strategies for pre-trained models

4.Data augmentation and preprocessing (ImageDataGenerator, preprocess_input)

5.Model evaluation and performance visualization (confusion matrix, accuracy/loss curves)

### 🌍Insight:
1.Fine-tuning ResNet50 boosts accuracy – By carefully unfreezing the last 100 layers and using a low learning rate, the model achieved a significant accuracy improvement from 59.5% to 70.8%.

2.Hyperparameter tuning reduced overfitting, with dropout layers (0.5, 0.3) and L2 regularization tweaks enhancing model stability.

3.Traffic-related objects (automobile, ship, truck) were easier to classify (F1 > 0.8), while animals (cat, deer, dog) had lower accuracy (F1 < 0.7).




### 🌍Presentation
[Presentation](https://youtu.be/zcPRyP_dtSE)
