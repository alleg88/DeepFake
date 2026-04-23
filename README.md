# Deepfake Detection Project

This repository contains a notebook-based deepfake detection workflow built with PyTorch and a ResNet18 backbone. The main notebook trains a binary classifier on extracted face frames and evaluates it on a second dataset.

## Project Files

- `Face Detection.ipynb`: main training and evaluation notebook
- `metrics.resnet.csv`
- `metrics.efficientnet.csv`
- `metrics_evaluation.celeb.csv`

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Setup

Extract `data.zip` into the repository root before running the notebook.

After extraction, the folder structure should look like this:

```text
face.done/
  Face Detection.ipynb
  data.zip
  data/
    Faces/
      Real/
      Fake/
    CelebDF_Faces/
      Real/
      Fake/
```

The notebook is already configured to read from:

- `data/Faces`
- `data/CelebDF_Faces`

## Running The Notebook

1. Extract `data.zip` into this folder.
2. Install the dependencies from `requirements.txt`.
3. Open `Face Detection.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
4. Run the cells in order.

The notebook performs:

- dataset loading and validation
- training on the extracted face-frame dataset
- validation metrics during training
- cross-dataset evaluation on Celeb-DF face frames
