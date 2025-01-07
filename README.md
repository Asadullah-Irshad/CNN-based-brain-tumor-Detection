# CNN-based-brain-tumor-Detection

This project aims to detect brain tumors using different methods, including Convolutional Neural Networks (CNN) and VGG models, trained on MRI scanned images for accurate classification of tumor and non-tumor images. The model leverages deep learning techniques to analyze complex patterns in medical images and assist healthcare professionals in the early diagnosis of brain tumors.

## Required Libraries

This project requires the following Python libraries:

- `os`: For interacting with the operating system and file handling.
- `numpy`: For numerical computing and array manipulation.
- `matplotlib`: For data visualization (graphs and plots).
- `seaborn`: For statistical data visualization.
- `PIL`: For image processing (opening, converting, resizing images).
- `tensorflow`: For building and training deep learning models.
- `tensorflow_addons`: For additional TensorFlow functionalities.
- `keras`: For neural network functionalities.
- `scikit-learn`: For evaluation metrics such as classification report and confusion matrix.

To install these libraries, you can use the following:

```bash
pip install numpy matplotlib seaborn pillow tensorflow tensorflow-addons scikit-learn
## How to Run the Notebook

You can run the notebook in the following ways:

### Option 1: Run Locally with Jupyter Notebook

To run the notebook locally, follow these steps:

1. Ensure you have **Jupyter Notebook** installed. You can install it via pip if needed:
   ```bash
   pip install notebook
2. Install the required libraries:

bash
pip install numpy matplotlib seaborn pillow tensorflow tensorflow-addons scikit-learn

3. Download the notebook and dataset to your local machine.

4. Open a terminal or command prompt and run: jupyter notebook

5. In the browser window that opens, navigate to the brain_tumor_detection.ipynb file and open it.

6. Run the cells in the notebook to start the analysis.

### Option 2: Run on Google Colab

You can also run this notebook on Google Colab without needing to install anything locally.

1. Open the notebook in Google Colab by clicking the link below:

   [Open in Google Colab](<Your-Link>)

2. Upload the dataset (if prompted) to Colab.

3. Run each cell in the notebook to start the analysis.

### Project Structure

- `brain_tumor_detection.ipynb`: Jupyter notebook containing the model training and evaluation code.
- `dataset/`: Folder containing the MRI image dataset.
- `models/`: Folder to save trained models (optional).

### Model Architecture

This project uses two main deep learning models for brain tumor detection:

### Convolutional Neural Network (CNN):
A custom CNN architecture is used to detect tumor patterns in MRI images.

### VGG:
A pre-trained VGG model is fine-tuned for the task of brain tumor classification.

### Evaluation Metrics
The model's performance is evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
