# Predicting the Subcellular Localization of Ubiquitinated Proteins

This project implements a two-stage machine learning pipeline to first identify ubiquitinated proteins and then predict their subcellular location within a human cell.

## Pipeline Overview

The pipeline consists of two main stages:

1.  **Stage 1: Ubiquitination Prediction:** A pre-trained LSTM model is used to predict whether a given protein sequence contains ubiquitination sites.
2.  **Stage 2: Localization Prediction:** If a protein is predicted to be ubiquitinated, a custom deep learning model predicts its subcellular location (Nucleus, Cytoplasm, Mitochondrion, Golgi Apparatus, or Endoplasmic Reticulum).

## Model Performance

The final localization model, built using a **Bidirectional LSTM with an Attention mechanism**, achieved a **weighted F1-score of 0.67** on an independent test set.

## How to Run

To run the full prediction pipeline on a sample protein, execute the following command in the terminal:

```bash
python run_pipeline.py