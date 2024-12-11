# CMU_IDL_51

## Overview
This repository contains the implementation for a multi-attribute text style transfer project. The project focuses on leveraging transformer-based architectures to rewrite text across stylistic dimensions such as sentiment, tone, and formality while preserving semantic integrity. It includes two key components:

1. **Data Preprocessing**: Preparing datasets for transformer-based text style transfer.
2. **Model Training**: Training custom and pre-trained transformer models to perform text style transfer and evaluating their performance.

---

## Repository Structure
- `Preprocessing/`
  - Scripts and instructions for preprocessing datasets (e.g., Yelp, Amazon) to prepare data for training models.
- `Training Code/`
  - `CustomModel.py`: Script for training the custom transformer model.
  - `T5_Finetuning.py`: Script for fine-tuning the T5 model.
- `README.md`
  - The main readme file.

---

## Setup Instructions
Follow the steps below to set up the project and reproduce the results.

### **1. Clone the Repository**
```bash
git clone https://github.com/your-repo-link.git
cd CMU_IDL_51
```

### **2. Install Dependencies**
- Make sure you have Python 3.8 or higher installed.
- Install the required packages:
```bash
pip install -r requirements.txt
```

### **3. Data Preprocessing**

#### **Step 1: Process the Data**
Move into the `Preprocessing/Data/` directory and follow the instructions provided in the README file there. This will include steps such as:
- Downloading and extracting the datasets (e.g., Yelp, Amazon reviews).
- Cleaning, normalizing, and tokenizing the text data.
- Annotating data with relevant style attributes (e.g., sentiment, tone).

```bash
cd Preprocessing/Data
cat README.md  # Follow the instructions here
```

#### **Step 2: Prepare Transformer Data**
Next, move to `Preprocessing/Transformer/` to prepare the datasets for training transformer models. Follow the README provided there to:
- Generate subword tokenized data using BPE (Byte Pair Encoding).
- Split datasets into training, validation, and test sets.
- Save the processed data paths for model training.

```bash
cd Preprocessing/Transformer
cat README.md  # Follow the instructions here
```

### **4. Model Training**
Move to the `Training Code/` directory to train the model. Use the preprocessed dataset paths created in the preprocessing step. The directory contains the following scripts:

- **CustomModel.py**: Train the custom transformer model.
  ```bash
  python CustomModel.py 
  ```

- **T5_Finetuning.py**: Fine-tune the T5 model.
  ```bash
  python T5_Finetuning.py 
  ```


---

## Results
The project includes:
- **Custom Transformer Model**
  - Reconstruction Loss: ~6.2
  - Adversarial Loss: ~1.39
  - Cross-Alignment Loss: ~0.90
- **Fine-Tuned T5 Model**
  - Total Loss: ~0.98

---


## Acknowledgments
This work is inspired by existing research in text style transfer, including models like T5 and custom transformers. Special thanks to the CMU-IDL course for guidance and resources.


