# **Visual Taxonomy - Attribute Prediction for E-commerce**

A deep learning-based project to classify products into categories and predict their attributes using EfficientNet. Developed as part of the Meesho Hackathon, this project demonstrates high-performance modeling and efficient handling of large-scale datasets.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Project Structure](#project-structure)
5. [Setup and Installation](#setup-and-installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Contributors](#contributors)
9. [License](#license)

---

## **Overview**
The **Visual Taxonomy** project focuses on predicting product attributes for e-commerce platforms using advanced computer vision techniques. The project highlights include:
- EfficientNet-based deep learning model for image classification.
- Preprocessing pipelines for handling large datasets.
- Automated training and inference scripts for streamlined execution.

---

## **Dataset**
The dataset is publicly available on Kaggle:
- [Meesho Datasets](https://www.kaggle.com/datasets/samarth060803/meesho-datasets/data)

### Folder Structure:
```
Meesho_Hack/
├── data/                    # Contains dataset files
│   ├── images/              # Images used for training and testing
│   │   ├── train_images/    # Training images
│   │   ├── test_images/     # Testing images
│   ├── processed/           # Processed data for training/validation
│   │   ├── train_split.csv  # Training split metadata
│   │   ├── valid_split.csv  # Validation split metadata
│   │   ├── prediction.csv   # Predictions for the dataset
│   ├── raw/                 # Raw metadata files
│   │   ├── train.csv        # Raw training metadata
│   │   ├── test.csv         # Raw testing metadata
│   ├── train_processed.csv  # Fully processed training data
│   ├── val_processed.csv    # Fully processed validation data
├── encoders/                # Encoded labels
│   ├── label_encoders.joblib  # Saved label encoder
│   ├── train_encoded.csv    # Encoded training data
│   ├── val_encoded.csv      # Encoded validation data
├── models/                  # Trained models and logs
│   ├── logs/                # Training logs
│   ├── efficientnet_model_checkpoint.h5  # Intermediate model checkpoint
│   ├── efficientnet_model_final.h5       # Final trained model
├── scripts/                 # Supporting Python scripts
│   ├── prepare_data.py      # Prepares and splits datasets
│   ├── encode_labels.py     # Handles label encoding
│   ├── category_attributes_mapping.py  # Maps product categories to attributes
│   ├── train.py             # Main training script
│   ├── inference.py         # Script for running inference
├── submission/              # Submission-related files
│   ├── sample_submission.csv  # Example submission format
│   ├── submission_1.csv     # First submission results
│   ├── submission_2.csv     # Second submission results
│   ├── submission.csv       # Final submission file
├── README.md                # Project documentation
├── requirements.txt         # Dependencies for the project
├── run_training.sh          # Shell script for automating training

```
---

## **Model Architecture**
The project leverages **EfficientNet** for image classification:
- **EfficientNet-B7**: Pretrained on ImageNet and fine-tuned for the dataset.
- Model Checkpoints:
  - `efficientnet_model_checkpoint.h5`: Intermediate checkpoint during training.
  - `efficientnet_model_final.h5`: Final trained model.

---

## **Project Structure**
```
Meesho_Hack/
├── data/                    # Dataset files
├── encoders/                # Encoded labels
├── models/                  # Trained models and logs
├── scripts/                 # Helper scripts for data processing
├── submission/              # Submission CSV files
├── train.py                 # Main training script
├── inference.py             # Inference script
├── requirements.txt         # Dependencies
├── run_training.sh          # Shell script for automation
├── README.md                # Project documentation
```
---

## **Setup and Installation**

### Prerequisites
- Python 3.10 (recommended with Conda environment)
- Git
- Virtual Environment (Conda)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/SamarthShinde/Visual-Taxonomy-.git
   cd Visual-Taxonomy-
   ```

2.	Set up a Conda environment:
```bash
conda create --name mesho_env python=3.10
conda activate mesho_env
```

3.	Install dependencies:
```bash
pip install -r requirements.txt
```

4.	Download the dataset from Kaggle and place it in the data/images folder.

Usage

1. Data Preparation

Run the script to preprocess the dataset and split it into training and validation sets:
```
python scripts/prepare_data.py
```
2. Train the Model

Train the model using:
```
python train.py
```
Alternatively, automate training with the shell script:
```
bash run_training.sh
```
3. Run Inference

Make predictions on the test dataset:
```
python inference.py
```
Results
```
	•	F1 Score: 0.35138
	•	Model Accuracy: 92.78%
	•	Epochs: 100
	•	Training Details:
	•	GPU Setup: Tesla V100 32GB (4 GPUs)
	•	Training Time: ~3 days
	•	Recommendation: Use high-end GPUs for optimal training performance.
```
Contributors
```
	•	[Samarth Shinde](https://github.com/SamarthShinde)

GitHub Profile
Email: Samarth.shinde505@gmail.com
```
License

This project is licensed under the MIT License. See the LICENSE file for details.

Additional Notes
	•	For improved training performance, ensure access to high-end GPUs.
	•	For any issues or contributions, feel free to open an issue or submit a pull request.

---

