## Project Title
Aerial Object Classification & Detection
Skills take away From This Project
Deep Learning


Computer Vision


Image Classification & Object Detection


Python


PyTorch


Data Preprocessing & Augmentation


YOLOv8 (Object Detection)


Model Evaluation


Streamlit Deployment


Domain
Aerial Surveillance, Wildlife Monitoring, Security & Defense Applications

📌 Problem Statement
This project aims to develop a deep learning-based solution that can classify aerial images into two categories — Bird or Drone — and optionally perform object detection to locate and label these objects in real-world scenes.
The solution will help in security surveillance, wildlife protection, and airspace safety where accurate identification between drones and birds is critical. The project involves building a Custom CNN classification model, leveraging transfer learning, and implementing YOLOv8 for real-time object detection. The final solution will be deployed using Streamlit for interactive use.
📌 Real-Time Business Use Cases
Wildlife Protection


Detect birds near wind farms or airports to prevent accidents.


Security & Defense Surveillance


Identify drones in restricted airspace for timely alerts.


Airport Bird-Strike Prevention


Monitor runway zones for bird activity.


Environmental Research


Track bird populations using aerial footage without misclassification.

📌 Project Workflow
1. Understand the Dataset
Inspect dataset folder structure


Check number of images per class


Identify class imbalance


Visualize sample images

2. Data Preprocessing
Normalize pixel values to [0, 1]


Resize images to a fixed size (224×224 for classification)

3. Data Augmentation
Apply transformations: rotation, flipping, zoom, brightness, cropping

4. Model Building (Classification)
Custom CNN: Conv layers, pooling, dropout, batch normalization, dense output layer


Transfer Learning: Load models like ResNet50, MobileNet, EfficientNetB0 and fine-tune

5. Model Training
Train both models


Use EarlyStopping & ModelCheckpoint


Track metrics: Accuracy, Precision, Recall, F1-score

6. Model Evaluation
Evaluate test results with confusion matrix & classification report


Plot accuracy/loss graphs



7. Model Comparison
Compare accuracy, training time, and generalization performance
Save the best performing model for Streamlit deployment

📌 Object Detection with YOLOv8
Steps:
Install YOLOv8.


Prepare dataset (images and YOLOv8-format .txt labels — already done).


Create a data.yaml configuration file for YOLOv8.


Train the YOLOv8 model.


Validate the trained model.


Run inference on test or new images.

📌 Streamlit Deployment
Create a simple UI with image upload


Display prediction (Bird / Drone) & confidence score


(Optional) Show YOLOv8 detection results with bounding boxes

📌 Project Deliverables
Trained models (Custom CNN, Transfer Learning, YOLOv8)


Streamlit app for classification/detection


Scripts & notebooks for preprocessing, training, evaluation


Model comparison report


GitHub repository with documentation


Well-structured, commented code


## 🛠 Technical Tags
Computer Vision, Deep Learning, Image Classification, Object Detection, CNN, YOLOv8, Transfer Learning, Data Augmentation, Model Evaluation, Streamlit Deployment, Aerial Surveillance AI, PyTorch📌 Datasets

📌 Dataset Relationship
> **IMPORTANT**: The classification and object detection datasets share **identical source images** (byte-for-byte). The classification dataset (3,319 images) is the detection dataset (3,400 images) with 81 empty-label background images excluded, reorganized into `bird/` and `drone/` class folders. This must be accounted for in evaluation to prevent data leakage.

📌 Classification Dataset
 Source: classification_dataset
 Task: Image Classification (Binary: Bird / Drone)
 Data Type: RGB Images
 Format: .jpg 
Structure
TRAIN set:
 - bird: 1414 images
 - drone: 1248 images
VALID set:
 - bird: 217 images
 - drone: 225 images
TEST set:
 - bird: 121 images
 - drone: 94 images

📌 Object Detection Dataset (YOLOv8 Format)
Source : object_detection_Dataset
The dataset contains 3400 images with corresponding YOLOv8-format annotations (.txt files).
Of these, 81 images have empty label files (background/hard negative images with no objects).
Each annotation file contains bounding boxes in the format:

 	<class_id> <x_center> <y_center> <width> <height>


Data split: Train (2,728), Validation (448), Test (224).

Bounding Box Class Distribution:
 - Bird (class 0): 1,406 bounding boxes
 - Drone (class 1): 135 bounding boxes
 - Ratio: 10.4:1 (Bird:Drone) — severe bbox-level imbalance
 - Note: Image-level counts are balanced (~1,414 bird images vs ~1,248 drone images)



