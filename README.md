# 🧬 Skin Cancer Detection App (Benign vs Malignant vs Normal)
This repository contains a deep learning-based web application to detect **skin cancer** from skin lesion images. It classifies input images into **Benign**, **Malignant**, or **Normal (Non-cancerous)** using a fine-tuned EfficientNetB0 model. The app provides predictions, descriptions, symptoms, and a severity graph.
## 🚀 Features
* Deep learning model trained on dermoscopic images
* Classification into 3 skin conditions: Benign, Malignant, and None (Normal)
* Interactive web app using Streamlit
* Upload patient details and skin image
* Dynamic prediction results with condition overview and treatment suggestions
* Confusion matrix, training logs, and metrics visualizations
---
## 📁 Project Structure
```
.
├── app.py                    # Streamlit frontend application
├── skincancer10.py          # Model training script
├── models/
│   ├── best_model.h5        # Saved trained model
│   └── class_indices.json   # Class label mappings
├── results/
│   ├── confusion_matrix.png
│   └── classification_report.txt
├── logs/
│   └── training_log.csv
├── training.log             # Training console log
├── requirements.txt         # Python dependencies
└── README.md                # This file
```
---
## 🧪 Model Training
**`skincancer10.py`** handles the training pipeline:
* Uses `EfficientNetB0` as the base model (pretrained on ImageNet)
* Applies data augmentation and normalization using `ImageDataGenerator`
* Includes early stopping, checkpointing, and logging
* Saves the best model and evaluation metrics
### 🔧 Configuration
Update the dataset path in `CONFIG` inside `skincancer10.py`:
```python
CONFIG = {
    'dataset_path': r"your/dataset/path",
    ...
}
```
To train the model:
```bash
python skincancer10.py
```
Make sure your dataset is structured as:
```
data/
├── Benign/
├── Malignant/
└── None/
```
---
## 🖥️ Running the Web App
The web application is built with **Streamlit**.
### 💡 Features of `app.py`:
* Upload an image and patient details
* Get prediction results with condition name and description
* View symptoms and treatment options
* Visualize severity using matplotlib graphs
### 📦 Installation
Create a virtual environment and install dependencies:
```bash
pip install -r requirements.txt
```
Run the app locally:
```bash
streamlit run app.py
```
---
## 📊 Sample Outputs
* 🧾 Prediction: **Malignant**
* 💬 Description: A malignant skin lesion that may require immediate attention.
* 🧠 Graph: Shows severity and progression level.
* ✅ Additional outputs: Confusion matrix, classification report (after training)
---
## 📌 Requirements
You can generate a `requirements.txt` using:
```bash
pip freeze > requirements.txt
```
Typical dependencies include:
```
streamlit
tensorflow
numpy
matplotlib
seaborn
scikit-learn
```
---
## 📍 Acknowledgements
* [Malignant Vs Bengin](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign) - The dataset used for training.
* TensorFlow/Keras for model development.
* Streamlit for UI interface.
---
## 🧠 Future Improvements
* Deploy the app using Hugging Face Spaces or Render
* Include real-time camera input
* Add user feedback collection for improving model
---
## 🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
---
## 📜 License
This project is open source and available under the [MIT License](LICENSE).
