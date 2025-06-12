# iNaturalist-ComputerVision

A deep learning project for image classification on a subset of the [iNaturalist dataset](https://www.inaturalist.org/), developed as part of the *Deep Learning for Computer Vision* course.

## Project Overview

This project tackles the problem of multi-class classification of animal and plant species from images. The focus is on building and comparing different CNN architectures, both from scratch and using transfer learning with VGG16, on a curated subset of the iNaturalist dataset.

Key goals:
- Preprocess and subsample the iNaturalist dataset for efficient training
- Build and tune convolutional neural networks
- Apply regularization techniques (dropout, data augmentation)
- Leverage transfer learning with VGG16 (feature extraction and fine-tuning)
- Evaluate models using accuracy, loss, AUC, and Precision-Recall metrics

## Repository Structure

iNaturalist-ComputerVision/
- iNaturalist_DLCV.ipynb # Main Jupyter notebook with code and experiments
- iNaturalist_DLCV_report.pdf # Full technical report with detailed analysis and results
- iNaturalist_DLCV_slides.pptx # Presentation slides summarizing the project
- requirements.txt # List of Python dependencies

## Dataset

A subset of the iNaturalist dataset was used:
- Only RGB images (discarded CMYK and L modes)
- Resized to 200x200 for faster computation
- 100 most frequent categories (from 1003) selected to reduce class imbalance
- Total: ~58,000 images split into train (60%), validation (20%), and test (20%)

## Models

### 1. **From Scratch CNNs**
- Simple 2-layer CNN: ~600k parameters, early overfitting, ~11% accuracy
- Deeper CNNs: up to 5M+ parameters, improved performance (up to 17% accuracy)
- Regularization with dropout and data augmentation improved accuracy to **24%**

### 2. **Transfer Learning with VGG16**
- **Feature Extraction**: ~33% accuracy but prone to overfitting
- **Feature Extraction + Data Augmentation**: ~26% accuracy with better generalization
- **Fine-Tuning (last 4 conv layers + classifier)**: Best model with **37% validation accuracy**, balanced performance across all metrics

## Evaluation Metrics

Evaluated on multiple performance metrics:
- **Top-1 Accuracy**
- **Top-5 Accuracy**
- **Cross-Entropy Loss**
- **AUC (Area Under Curve)**
- **Precision-Recall curves** (macro and per-class)

The **fine-tuned VGG16** model outperformed all others and showed the best generalization capacity.

## Results

| Model                              | Accuracy | AUC Score | Observations                              |
|-----------------------------------|----------|-----------|-------------------------------------------|
| Scratch (unregularized)           | ~11%     | Low       | Overfitted quickly                        |
| Scratch + Regularization          | ~24%     | Medium    | Good generalization, solid baseline       |
| VGG16 Feature Extraction          | ~33%     | Low       | High accuracy, but poor generalization    |
| VGG16 + Data Augmentation         | ~26%     | Medium    | Better generalization, lower loss         |
| VGG16 Fine-Tuned (Best)           | **37%**  | **High**  | Best overall model                        |


## Getting Started

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/iNaturalist-ComputerVision.git
   cd iNaturalist-ComputerVision
   ```
3. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the notebook: 
Open `iNaturalist_DLCV.ipynb` in Jupyter and execute the cells.

## Acknowledgments

Dataset: iNaturalist

Pretrained model: VGG16

Course: Deep Learning for Computer Vision – Bocconi University
