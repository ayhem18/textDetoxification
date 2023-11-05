# Detoxification

## Description

This repository includes my experiments with text several text detoxification approaches. 

The repository is structured as follows:  

- `data/` - containing all the data needed across the repository: Run the `src/data_preparation/make_data.py`
- `src/` - folder with source code
- `models/` - folder with the minimalistic verison of the models
- `notebooks/` - folder with several examples of code using in the experiments

## Data
The repository uses 3 different datasets: 
1. ParaMNT dataset
2. [The Jigsaw toxicity dataset](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data?select=train.csv.zip)
3. [The paradetox dataset](https://github.com/s-nlp/paradetox)


## Training
Each model is associated with a `train` and `predict` scripts.

## Evaluation
Assuming the we have the generated comments <ABS_PATH_TO_PREDS> in a file named  and the correct / reference comments in a file <ABS_PATH_TO_REFERENCES>, run the folling command

bash eval.sh <ABS_PATH_TO_REFERENCES> <ABS_PATH_TO_PREDS>