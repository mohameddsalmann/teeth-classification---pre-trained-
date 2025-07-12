# Dental Image Classification Project

## Overview
This project implements a deep learning model to classify dental images into 7 categories using a MobileNetV2 architecture with a custom Focal Loss function. The solution includes data preprocessing, training on Kaggle, and a Streamlit-based GUI for local inference. The model is trained on a dataset hosted on Kaggle and saved for local use.

## Features
- **Training**: Utilizes TensorFlow 2.15.0 with GPU acceleration on Kaggle.
- **Augmentation**: Applies random flips, rotations, zooms, and color adjustments to enhance dataset robustness.
- **GUI**: Streamlit interface for single and multiple image predictions with class probabilities.
- **Custom Loss**: Implements a Focal Loss to handle class imbalance.

## Requirements
### Kaggle Environment
- TensorFlow 2.15.0
- tensorflow-addons 0.23.0
- matplotlib, seaborn, sklearn, numpy
- Access to the dataset at `/kaggle/input/teeth/`

### Local Environment (Windows)
- Python 3.9+
- TensorFlow 2.15.0
- Streamlit
- Pillow
- numpy
- pickle

## Installation
### Kaggle Setup
1. Create a new Kaggle notebook.
2. Add the dataset `/kaggle/input/teeth/` to your notebook.
3. Install dependencies manually in a code cell:
   ```python
   !pip uninstall -y tensorflow tensorflow-addons -q
   !pip install --no-cache-dir tensorflow==2.15.0 tensorflow-addons==0.23.0 -q
   ```
4. Restart the kernel after installation.

### Local Setup
1. Clone or download the project files.
2. Install required packages:
   ```bash
   pip install tensorflow==2.15.0 streamlit pillow numpy
   ```
3. Place the saved model and class names files in `C:\Users\asus\Downloads\teeth classfication1\`.

## Usage
### Training on Kaggle
1. Run the following sections in order:
   - **Section 1**: Preprocessing to load and augment the dataset.
   - **Section 2**: Visualization to plot class distribution and augmentation examples.
   - **Section 3**: Training with 500 initial epochs and 75 fine-tuning epochs, saving the model.
2. Download `teeth_classification_model.keras` and `teeth_classification_class_names.pkl` from `/kaggle/working/`.

### Running the GUI Locally
1. Save the script as `app.py`.
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Upload images via the interface:
   - **Single Image Prediction**: Upload one `.jpg`, `.jpeg`, or `.png` file.
   - **Test Multiple Images**: Upload multiple files (up to 200MB each) for batch prediction.

## Files
- **`app.py`**: Streamlit GUI for image classification.
- **`teeth_classification_model.keras`**: Trained model file.
- **`teeth_classification_class_names.pkl`**: Pickled dictionary of class names.

## Troubleshooting
- **Kaggle Kernel Crashes**: Reduce `batch_size` to 16 in Section 1.
- **Model Loading Error**: Ensure TensorFlow 2.15.0 is used locally and resave the model on Kaggle if deserialization fails.
- **Multiple Upload Failure**: Check for typos or invalid image files.
- **GPU Not Detected**: Enable GPU accelerator in Kaggle settings.

## License
This project is for educational purposes. Modify and distribute as needed, but credit the original implementation.

## Contact
For issues or questions, refer to the development logs or contact the project maintainer.
