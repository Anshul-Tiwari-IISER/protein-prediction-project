# Predicting the Subcellular Localization of Ubiquitinated Proteins

This project implements a two-stage machine learning pipeline to first identify ubiquitinated proteins and then predict their subcellular location within a human cell.

## Project Overview

Ubiquitination is a critical cellular process that plays a key role in protein degradation, signaling, and trafficking. Understanding where ubiquitinated proteins are located within the cell is essential for studying many biological pathways and diseases. This project builds a computational tool to automate this analysis on a large scale.

The pipeline consists of two main stages:

1.  **Stage 1: Ubiquitination Prediction:** A pre-trained LSTM model is used as a "gatekeeper" to predict whether a given protein sequence contains ubiquitination sites.
2.  **Stage 2: Localization Prediction:** If a protein is predicted to be ubiquitinated, a custom-built, high-performance deep learning model predicts its subcellular location from among five classes: Nucleus, Cytoplasm, Mitochondrion, Golgi Apparatus, or Endoplasmic Reticulum.

## Model Performance

The final localization model was trained on a large-scale dataset of over 15,000 protein sequences. By upgrading the architecture to a **Bidirectional LSTM with an Attention mechanism**, the model achieved a **weighted F1-score of 0.67** on an independent test set, demonstrating high predictive accuracy.

### Final Classification Report
| Location      | Precision | Recall | F1-Score |
|---------------|-----------|--------|----------|
| Endoplasmic   | 0.66      | 0.76   | 0.70     |
| Golgi         | 0.71      | 0.67   | 0.69     |
| Cytoplasm     | 0.54      | 0.43   | 0.48     |
| Mitochondrion | 0.88      | 0.87   | 0.87     |
| Nucleus       | 0.59      | 0.66   | 0.62     |
| **Weighted Avg** | **0.67** | **0.68** | **0.67** |

## How to Run

### Prerequisites
- Python 3.10+
- PyTorch (with CUDA for GPU support)
- pandas, scikit-learn, BioPython

### Running the Pipeline
To run the full prediction pipeline on a sample protein, clone this repository and execute the following command in your terminal, replacing `"YOUR_SEQUENCE_HERE"` with your protein sequence:

```bash
python run_pipeline.py "YOUR_SEQUENCE_HERE"

## How to Run

### Prerequisites
- Python 3.10+
- PyTorch (with CUDA for GPU support)
- pandas, scikit-learn, BioPython
