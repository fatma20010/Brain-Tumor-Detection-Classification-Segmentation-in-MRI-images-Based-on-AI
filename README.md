# Tumor Analysis and Reporting System

## Brief Description
The Tumor Analysis and Reporting System is an advanced tool for brain tumor analysis, leveraging machine learning to detect, classify, and segment tumors from MRI scans (NiFTI files). It predicts patient survival, generates detailed reports, and includes a chatbot for interactive user assistance, making it a valuable resource for medical professionals.

## Features
- **Tumor Detection**: Identifies the presence of tumors using a pre-trained ResNet50 model.
- **Tumor Classification**: Classifies tumors into types (e.g., Glioma, Meningioma, Pituitary) using a DenseNet model.
- **Tumor Segmentation**: Segments tumor regions in multi-sequence MRI data (T1, T1ce, T2, FLAIR) using an Improved UNet3D model.
- **Feature Extraction**: Extracts detailed tumor features (e.g., volume, intensity, major axis length) for Glioma cases.
- **Survival Prediction**: Predicts patient survival days based on extracted features using an XGBoost model.
- **Patient Reporting**: Generates concise patient reports using the Together AI LLM and converts them to audio with text-to-speech.
- **Chatbot Integration**: Includes an interactive chatbot to assist users with queries, provide guidance, and explain results.
- **Visualization**: Displays segmentation results with green overlays and saves them as images.

## Results
Below is an example of the segmentation result, showing the original MRI slice (left) and the segmented tumor with a green overlay (right):

![Segmentation Result](images/segmentation_result.png "Segmentation Result with Tumor Overlay")

## Requirements
- Python 3.8+
- Required libraries:
  - `tensorflow`
  - `torch`
  - `torchvision`
  - `numpy`
  - `nibabel`
  - `matplotlib`
  - `opencv-python`
  - `scikit-image`
  - `scikit-learn`
  - `joblib`
  - `gtts`
  - `playsound`
  - `together` (for Together AI API)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tumor-analysis-system.git
   cd tumor-analysis-system
