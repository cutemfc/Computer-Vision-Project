# Computer-Vision-Project
# 🎇Computer-Vision-Project (Convolutional Neural Network)
### Data of this project is download from CIFAR-10 dataset


### 🌍 Goal:
This project aims to identify 10 class objects using convolutional Neural Network modelling. 

### 🌍Descriptions:
This project focuses on multi-class image classification using the CIFAR-10 dataset, which contains 60,000 color images (32x32 pixels) spanning 10 mutually exclusive classes. The dataset is split into 10,000 training images and 10,000 test images for experimental consistency. A custom classification head was created and combined with a pre-trained ResNet50 model (CNN architecture) to leverage the power of transfer learning. The ResNet50 base model was initially frozen, and later selectively fine-tuned for performance enhancement.

### 🧪 Experiments Conducted:
Increased Dropout Layers to reduce overfitting and improve generalization.
 
Tuned Regularization Weights (e.g., L2 penalties) to optimize learning and reduce complexity.

Fine-Tuned Model Layers by unfreezing parts of the ResNet50 base model and training additional layers to improve feature learning.




### 🌍Skills:


Skills (CNN & ResNet50 Focused):
1.Image classification using CNN and ResNet50 (transfer learning)

2.Model building and training with Keras (Conv2D, MaxPooling2D, Dense)

3.Fine-tuning and layer freezing strategies for pre-trained models

4.Data augmentation and preprocessing (ImageDataGenerator, preprocess_input)

5.Model evaluation and performance visualization (confusion matrix, accuracy/loss curves)

### Insight:
1.Exploratory data analysis revealed that holidays and the perishability of items significantly impact unit sales.

2.The naive model served as a baseline, while XGBoost achieved the best performance with a mean absolute percentage error (MAPE) of 9.49% and an R² score of 0.59.

3.Hyperparameter tuning further improved the predictive accuracy of the XGBoost model.

4.A Streamlit app was developed to provide an interactive interface for forecasting future sales and anticipating customer demand.


### Presentation
[Presentation](https://youtu.be/zcPRyP_dtSE)
