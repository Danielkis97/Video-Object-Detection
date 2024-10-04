![Python](https://img.shields.io/badge/Python-v3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.4.1-orange?logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-v4.7.0.72-green?logo=opencv)
![TorchMetrics](https://img.shields.io/badge/TorchMetrics-v1.4.2-purple?logo=python)
![Matplotlib](https://img.shields.io/badge/Matplotlib-v3.6.2-red?logo=matplotlib)
![tqdm](https://img.shields.io/badge/tqdm-v4.66.3-yellow?logo=python)
![Issues](https://img.shields.io/github/issues/danielkis97/video-object-detection)

# Video Object Detection Project

## Introduction
The **Video Object Detection Project** is designed specifically for detecting **persons** in video sequences. The current implementation is focused only on person detection, and the models are trained and calibrated to recognize and track human figures. This project integrates multiple models, including YOLOv8, Faster R-CNN, and SSD, providing flexibility and robustness in detecting persons within video frames. It offers a user-friendly graphical interface, efficient video processing capabilities, detailed evaluation metrics, and intuitive visualizations of detection results.

## Example Video Output
Here is an example of the output generated by the model "Faster R-CNN", showcasing **person detection**:

![Example GIF](https://github.com/Danielkis97/Video-Object-Detection/blob/main/assets/MOT17-04_FasterR-CNN_output_GIF.gif?raw=true)

## Architecture
The project is structured into several modular components to ensure scalability, maintainability, and ease of integration. The core components include:

- **Person Detection Focus:** The current version of the project is designed to detect persons specifically, with future updates planned to add additional object categories.
- **Graphical User Interface (GUI):** Allows users to select video sequences and choose the desired detection models.
- **Main Application:** Orchestrates the workflow by handling video processing, model integration, evaluation, and visualization.
- **Video Processing Module:** Utilizes OpenCV for reading and writing video frames.
- **Model Integration Module:** Integrates various object detection models such as YOLOv8, Faster R-CNN, and SSD.
- **Evaluation Module:** Computes performance metrics using TorchMetrics and manages data with Pandas.
- **Visualization Module:** Generates plots of metrics using Matplotlib and displays progress bars with tqdm.
- **Output Directory:** Stores annotated videos, metrics tables, and visualization images for further analysis.



## Project Architecture

Here is the high-level architecture of the project, represented by a PlantUML diagram:

![Enhanced Video Object Detection Project Architecture](https://github.com/Danielkis97/Video-Object-Detection/blob/main/assets/PlantUML.png?raw=true)



## Features
- **Multi-Model Support:** Seamlessly integrates YOLOv8, Faster R-CNN, and SSD for versatile object detection.
- **User-Friendly GUI:** Intuitive interface for selecting video sequences and models.
- **Efficient Video Processing:** Fast frame extraction and annotation using OpenCV.
- **Detailed Evaluation Metrics:** Computes mAP, Precision, Recall, F1 Score, IoU, and more.
- **Comprehensive Visualizations:** Generates tables and confusion matrices to visualize performance metrics.
- **Automatic Calibration:** Option to calibrate confidence thresholds using linear or binary search methods.
- **Progress Monitoring:** Real-time progress bars to track processing status.

## Models

### YOLOv8 Model
The YOLOv8 model (`yolo.py`) leverages the **Ultralytics YOLOv8** framework for real-time object detection. It is optimized for speed and accuracy, making it suitable for applications requiring rapid inference.

### Faster R-CNN Model
The Faster R-CNN model (`custom_faster_rcnn.py`) is implemented using **PyTorch**. This model excels in detecting objects with high precision by employing a region proposal network to identify potential object locations before classification.

### SSD Model
The SSD300 model (`ssd.py`) utilizes the **Single Shot MultiBox Detector (SSD)** architecture with a VGG16 backbone. SSD balances speed and accuracy, making it effective for detecting objects in various scales within video frames.

## Installation

> [!IMPORTANT]
> Follow these instructions to set up and run the video object detection project.

### Prerequisites
- **Python 3.10**
- **Git**

### Clone the Repository
```bash
git clone https://github.com/Danielkis97/video-object-detection.git
cd video-object-detection
```

### Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies
Install the required Python packages using pip.

```bash
pip install -r requirements.txt
```

### Additional Setup

#### 1. **Ultralytics YOLOv8:**
Ensure you have access to the YOLOv8 weights. You can download pre-trained weights from the Ultralytics repository.

#### 2. **Faster R-CNN and SSD:**
These models utilize pre-trained weights available through PyTorch's model zoo. Ensure you have an active internet connection for the initial download.

#### 3. **MOT17 Dataset:**
Download the [MOT17 Dataset](https://motchallenge.net/data/MOT17/) and extract it. Place the dataset in the `data` folder located in the project's root directory.

```bash
mkdir -p data/MOT17
# Place the extracted MOT17 dataset files here
```
Ensure the directory structure is as follows:

```
videodetectionproject/
│
├── custom_models/
│   └── __init__.py
│   └── custom_faster_rcnn.py
│   └── ssd.py
│   └── yolo.py
│   └── yolov8s.pt
├── utils/
│   └── __init__.py
│   └── video_processing.py
│   └── visualization.py
├── data/
│   └── MOT17/
│       ├── train/
│       └── test/
│
├── requirements.txt
├── main.py
...
```


#### Steps to Use the Application

1. **Select Video Sequences:**
    - Click on the "Add Sequences" button to browse and select directories containing videos.
    - Supported format is mp4 .
2. **Choose Detection Models:**
    - Select one or more models (YOLOv8, Faster R-CNN, SSD) by checking the corresponding boxes.
3. **Configure Parameters:**
    - Enable "Automatic Calibration" to let the application determine optimal confidence thresholds.
    - Choose the calibration method: Linear Search (standard, more accurate) or Binary Search (faster).
    - If automatic calibration is disabled, manually set the confidence threshold.
4. **Start Processing:**
    - Click on the "Start Processing" button to begin object detection on the selected video sequences.
    - Monitor progress through the progress bar and status display.
5. **Stop Processing:**
    - Click on the "Stop Processing" button to terminate the process at any time.
6. **View Results:**
    - Upon completion, annotated videos, metrics tables, and confusion matrix images will be saved in the output directory.
    - Evaluation results are displayed within the application under the "Evaluation Results" section.

## Evaluation
The application computes a range of performance metrics to evaluate the effectiveness of the object detection models.

### Metrics Computed
- **Mean Average Precision (mAP):** Measures the accuracy of the model in predicting bounding boxes and classifying objects.
- **Precision:** The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall:** The ratio of correctly predicted positive observations to all actual positives.
- **F1 Score:** The weighted average of Precision and Recall.
- **Intersection over Union (IoU):** Measures the overlap between the predicted bounding box and the ground truth.
- **Processing Time:** Total time taken to process the video sequences.
- **Average Confidence:** The average confidence score of the detections.

### Evaluation Process
1. **Calibration (Optional):** If enabled, the application calibrates the confidence threshold to optimize mAP.
2. **Detection:** Each frame is processed to detect objects using the selected models.
3. **Metrics Calculation:** Performance metrics are computed based on the detections and ground truth annotations.
4. **Visualization:** Metrics are visualized through tables and confusion matrices for easy interpretation.

## Visualization
The project provides comprehensive visualizations to aid in understanding the detection performance.

### Metrics Table
A table summarizing all the computed metrics is generated and saved as an image. This table includes mAP, Precision, Recall, F1 Score, IoU, Processing Time, Average Confidence, TP, FP, and FN.

### Confusion Matrix
A confusion matrix image displays the counts of True Positives (TP), False Positives (FP), and False Negatives (FN), providing insight into the model's detection capabilities.

### Annotated Videos
Processed videos with bounding boxes and labels drawn around detected objects are saved for visual inspection.

## Results and Metrics

## Results and Metrics

In this section, the results of the evaluation using different calibration methods will be presented (Automatic and Manual). These methods were applied on three videos from the **MOT17 Dataset**, specifically **Videos 02, 04, and 05**, and the models evaluated include Faster R-CNN, SSD300, and YOLOv8.

### 1. Binary Search Automatic Calibration

#### Metrics Table:
![Binary Search Metrics](https://github.com/Danielkis97/Video-Object-Detection/blob/main/assets/automatic%20calibration%20binary%20search%20metrics%20table.png?raw=true)

---

#### Confusion Matrix:
![Binary Search Confusion Matrix](https://github.com/Danielkis97/Video-Object-Detection/blob/main/assets/automatic%20calibration%20binary%20search%20confusion%20matrix.png?raw=true)

---

### 2. Linear Search Automatic Calibration

#### Metrics Table:
![Linear Search Metrics](https://github.com/Danielkis97/Video-Object-Detection/blob/main/assets/automatic%20calibration%20linear%20search%20metrics%20table.png?raw=true)

---

#### Confusion Matrix:
![Linear Search Confusion Matrix](https://github.com/Danielkis97/Video-Object-Detection/blob/main/assets/automatic%20calibration%20linear%20search%20confusion%20matrix.png?raw=true)

---

### 3. Manual Calibration with 0.1 as Threshold

#### Metrics Table:
![Manual Calibration Metrics 0.1](https://github.com/Danielkis97/Video-Object-Detection/blob/main/assets/Manual%20calibration%200.1%20threshold%20metrics%20table.png?raw=true)

---

#### Confusion Matrix:
![Manual Calibration Confusion Matrix 0.1](https://github.com/Danielkis97/Video-Object-Detection/blob/main/assets/manual%20calibration%200.1%20threshold%20confusion%20matrix.png?raw=true)

---

### 4. Manual Calibration with 0.95 as Threshold

#### Metrics Table:
![Manual Calibration Metrics 0.95](https://github.com/Danielkis97/Video-Object-Detection/blob/main/assets/Manual%20calibration%200.95%20treshold%20metrics%20table.png?raw=true)

---

#### Confusion Matrix:
![Manual Calibration Confusion Matrix 0.95](https://github.com/Danielkis97/Video-Object-Detection/blob/main/assets/manual%20calibration%200.95%20threshold%20confusion%20matrix.png?raw=true)

---

These tables and confusion matrices provide a detailed overview of the performance of different object detection models under various calibration methods.

## Possible Bugs and Solutions

### 1. Stopping Sequences Issue

**Bug:**
If the **Stop Sequence** button is used during processing and the program is not restarted, it can cause errors in subsequent runs.

**Solution:**
After stopping a sequence or finishing a sequence process, ensure that the program is **restarted** before running a new sequence. This prevents any lingering state issues.

---

### 2. Video Format Issues

**Bug:**
If unsupported video formats are used, the application may fail to process the videos.

**Solution:**
Make sure to use only **MP4** format, as this is the only format currently supported.

---

## License
This project is licensed under the MIT License.

## Acknowledgments
- **Ultralytics YOLOv8:** For providing an efficient and accurate object detection framework.
- **PyTorch:** For its robust deep learning library that facilitates model training and inference.
- **OpenCV:** For powerful video processing capabilities.
- **TorchMetrics:** For comprehensive evaluation metrics.
- **Matplotlib & tqdm:** For effective data visualization and progress tracking.
- **MOT17 Dataset:** For providing ground truth annotations used in evaluation.

## Development Environment

The code for this project was developed using **PyCharm**, which offers a powerful IDE for Python development.

Happy Detecting! 🚀
