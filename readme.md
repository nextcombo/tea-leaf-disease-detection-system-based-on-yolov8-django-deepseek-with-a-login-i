# Tea Leaf Disease Detection System Based on YOLOv8+Django+DeepSeek with a Login Interface in Python Source Code + ONNX Model + Beautiful Web Interface

Effect Display: Login interface main interface based on YOLOv8 target detection system, supports image, video and real-time camera detection, integrated DeepSee.

### Effect Display

Login interface

![](01-41d5d655.jpg)

Main Interface

![](02-79c4d159.jpg)

A YOLOv8-based object detection system, supporting image, video, and real-time camera detection, integrated with DeepSeek AI analysis functions.

### Functional Characteristics

-

Image target detection

-

Video file detection

-

Real-time camera detection

-

DeepSeek AI Intelligent Analysis

-

Adjustable detection parameters (confidence, IoU)

-

Beautiful user interface

### [Test Environment]

windows10

anaconda3+python3.10

torch2.3.1

ultralytics8.4.7

Django==5.2.11

### Model can detect 8 categories

['Black rot of tea', 'Brown blight of tea', 'Leafrust of tea', 'Red Spider infested tea leaf', 'Tea Mosquito bug infested leaf', 'Tea leaf', 'White spot of tea', 'disease']

### Data Set Introduction

Dataset format: YOLO format (txt file without split path, only contains jpg images and corresponding yolo format txt files)

Number of images (number of jpg files): 9591

Number of annotations (number of txt files): 9591

Number of training set: 7260

Number of Validation Sets: 784

Number of test sets: 1547

Annotation category count: 8

Label category names (Note that the order of categories in the YOLO format does not correspond to this, but is based on the classes.txt file in the labels folder): ['Black rot of tea', 'Brown blight of tea', 'Leaf rust of tea', 'Red Spider infested tea leaf', 'Tea Mosquito bug infested leaf', 'Tea leaf', 'White spot of tea', 'disease']

Number of boxes labeled in each category:

Black rot of tea (tea black rot) frame count = 128

Brown blight of tea (Tea brown rot) frame count = 8

Leaf rust of tea (tea rust) frame count = 3645

Red Spider infested tea leaf (red spider infested tea leaves) frame count = 1022

Tea Mosquito bug infested leaf (Number of frames for tea leaf damage by mosquitoes = 7254

Tea leaf (healthy tea) frame count = 427

White spot of tea (tea white spot disease) frame count = 324

Disease (Unclassified) Frame Count = 4

Total frame count = 12812

Using labelImg:

Annotation rules: Draw a rectangle around the category.

Important Notice: No

Special : This dataset does not guarantee the accuracy of the trained models or weight files.

![](03-a885a96e.jpg)

Add image annotations, not exceeding 140 characters (optional)

![](04-27413674.jpg)

Add image annotations, not exceeding 140 characters (optional)

Annotation example:

![](05-5ec0ab81.jpg)

Add image annotations, not exceeding 140 characters (optional)

### Training Model Information

Number of training set images = 2487

Number of validation set images = 666

Training map = 97.7%

Training Accuracy = 95.9%

Training recall rate (Recall) = 97.0%

Validation set accuracy:

class=MAP50(%)

all=98

Black rot of tea=100

Leaf rust of tea=96

Red Spider infested tea leaf=98

Tea Mosquito bug infested leaf=98

Tea leaf=100

White spot of tea=96

### Install Dependencies

```
pip install -r requirements.txt
```

### Database Migration

```
python manage.py makemigrations
python manage.py migrate
```

### Running Projects

```
python manage.py runserver
```

Visit http://127.0.0.1:8000

### Default login information

-

Username: admin

-

Password: admin

### Main File Structure of the Project

```
yolov8-django-deepseek-det-system/
├── manage.py                 # Django Project Management Script
├── requirements.txt  # Project dependency package list
├── README.md  # Project Description Document
├── index.html  # Main page (as a backup or example page)
├── start_lan.bat  # A one-click batch script for starting the system
├── yolo_detection/ # Django project configuration directory
├── __init__.py
├── settings.py # Project configuration file (database, API key, etc.)
├── urls.py # Project's total route configuration
└── wsgi.py # WSGI deployment configuration
├── detection/ Detection function application module
├── __init__.py
├── apps.py # Application configuration
├── forms.py # Form definition (such as file upload form)
├── models.py
├── urls.py # Detection of application route configuration
└── views.py # Core view functions (YOLO detection logic, file processing, etc.)
├── templates/ # Web page template directory
├── index.html # Main operation interface (upload, detection, result display)
└── login.html # Login Page
├── media/ # Media file storage directory
└── uploads/ # Directory for storing uploaded files by users
├── test_img/ # Test image directory
└── *.jpg # Detection sample images
├── weights/  # Template directory
├── yolov8n.onnx # Model file
└── results.png # Training log image
├── class_names # Model category file
└── model_description.txt # Brief introduction to the training and process map0.5 parameters
```

### Technology Stack

-

Django 4.2+

-

YOLOv8 (Ultralytics)

-

OpenCV

-

DeepSeek API

-

Chart.js

-

HTML5/CSS3/JavaScript

### Notes

-

Ensure that the YOLOv8n.onnx model file is in the project's weights directory.

-

DeepSeek API key needs to be configured in settings.py

-

Recommend using Python 3.8+

### Common Evaluation Parameters Introduction

In the task of object detection, evaluating the performance of a model is crucial. The terms you mentioned are commonly used metrics for assessing the performance of a model. Here are detailed explanations of these terms:

Class: This usually refers to the category of objects that the model is designed to detect. For example, a model may be trained to detect different categories of objects such as vehicles, pedestrians, or animals.

Images: Represents the number of images in the validation set. The validation set is used to evaluate model performance and is separate from the training set to ensure fairness in evaluation results. Instances: Total number of target objects in all images. This includes the sum of all category objects, for example, if the validation set contains 100 images with an average of 5 target objects per image, then Instances would be 500.

P (Precision): Precision is the ratio of true positives among the instances predicted as positive by the model. The formula is: Precision = TP / (TP + FP), where TP represents True Positives and FP represents False Positives.

R (Recall): Recall is the proportion of all true positives correctly predicted by the model. The formula is: Recall = TP / (TP + FN), where FN represents false negatives.

mAP50: Mean Average Precision at IoU (Intersection over Union) threshold of 0.5, which represents the average precision when the IoU (Intersection over Union) is set to 0.5. The IoU metric measures the degree of overlap between predicted bounding boxes and ground truth bounding boxes. mAP is a comprehensive metric that considers both precision and recall, used to evaluate the performance of models across different recall levels. At an IoU of 0.5, if the overlap between the predicted bounding box and the ground truth bounding box reaches or exceeds 50%, it is considered correct.

mAP50-95: Refers to the average precision of a model within the range of IoU from 0.5 to 0.95 (with a step of 0.05). This is a more stringent evaluation criterion, demanding higher overlap between predicted bounding boxes and true ones. In object detection tasks, a higher IoU threshold implies that the model needs to be more accurate in target object localization. The computation of mAP50-95 takes into account multiple IoU thresholds ranging from lenient to strict, thus providing a comprehensive assessment of the model's performance. These metrics collectively form an important framework for evaluating the performance of object detection models. By comparing the performance of different models on these metrics, one can determine which model may be more effective in practical applications.

### Common Issues

In target detection training, a low Mean Average Precision (MAP) may be due to one of the following reasons:

Reason two: Small Objects: If the dataset contains a large number of small objects, it is likely to result in a low map value. This is because the features of small objects are not distinct, making it difficult for the model to learn them.

Reason 3: Mismatched model tuning parameters: For instance, if the learning rate is adjusted too high, it may lead to an overly rapid learning capability, resulting in disordered adjustments of model parameters.

Reason 4: Overfitting (This is now a rare occurrence in models): If the model performs exceptionally well on training data but poorly on validation or test data, it may be due to overfitting. This is often caused by an excessive number of model parameters relative to the amount of training data available, leading the model to learn noise or specific patterns from the training data and fail to generalize to new data. Nowadays, target detection models are well-equipped to handle this situation, with very few instances of overfitting occurring.

Reason 5: Different scenes: The validation set has high accuracy, but the test set does not, which may be due to inconsistencies between the training model's scene images and those in the test map. To address these issues, the following measures can be taken to improve MAP:

(1) Optimize model structure: Select the appropriate model based on the characteristics of the task and dataset, and try using different network architectures and components to improve model performance.

(2) Enhance Data Preprocessing: Proper preprocessing and enhancement of data, such as data normalization, missing value imputation, data expansion, etc., to enhance the generalization capability of the model.

(3) Adjusting the Loss Function: Try using different loss functions or combining multiple loss functions to optimize model performance.

(4) Optimize training strategies: Adjust hyperparameters such as learning rate, batch size, and number of training iterations, and employ optimization algorithms like learning rate decay and momentum to enhance model training effectiveness.

(5) Use pre-trained models: Transfer learning using pre-trained models on large datasets can accelerate model convergence and improve performance. (6) Increase the number of datasets

### Provided documents

Python source code

YOLOv8n.onnx model (without providing PyTorch model)

Training map, P, R curve graph (in weights\results.png)

Test images (under test_img folder)

Please note that the dataset is provided in the dataset_address.xlsx file.


## Payment

Here is a pay link on Stripe ( https://buy.stripe.com/3cs8yP7sY87d0vu9AB ). Please contact me lonlonago@foxmail.com after funding $89, and I will send you a complete data files , thank you!

![111.png](111.png)
