# Sentimental-Analysis-w-DL-using-BERT

Using Pytorch implement pre-trained base uncased Sentimental Aanalysis on twitter text corpus 
- Goolge Colab: Cuda GPU (K80)
- Epochs: 10
- Batch: 32

## EDA (Exploratory Data Analysis) & Preprocessing
Preprocess data from: ./Data/smile-annotations-final.csv
with Pandas & Numpy

## Training / Validation split
Split data into 85% training & 15% validation by each class

## Loading tokenizer and Encoding data
From HuggingFace library, import BertTokenizer to tokenize structured text then imported into TensorDataset

## Setting up BERT pre-trained Model
From HuggingFace library, import pretrained bert-base-uncased model with BertForSequenceClassification 

## Creating Data Loaders
From Torch import DataLoader, RandomSampler. Dump tokenized text tokens into dataloader 

## Setting up Optimizer and Scheduler
- learning rate = 1e-5
- epsilon = 1e-8
- Optimizer: AdamW
- Scheduler: get_linear_schedule_with_warmup

## Defining our performance Metrics
Define F1 score and Accuracy per class for evaluation

## Creating our training loop


## Loading & Evaluating our Model

