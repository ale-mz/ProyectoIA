# Copyright Oscar Fernández, Jorge Madríz y Kenneth Villalobos
# Code adapted from: https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/
# Code adapted from: https://www.kaggle.com/code/pritishmishra/fine-tune-bert-for-text-classification?scriptVersionId=116951029
# Special thanks to A. Badilla

# ----------------------------- Imports ---------------------------------------
import pytorch_lightning

import pandas

import torch.nn

from torchmetrics import MetricCollection
from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelPrecision, MultilabelF1Score


# ----------------------------- Constants -------------------------------------
# *************** Arguments constants *************************
TUNNING = True
def initialize_mode(arg):
    global TUNNING
    if arg == 0:
      TUNNING = False
    else:
      TUNNING = True

# *************** General constants ***************************
RNG_SEED = 0

# Apply deterministic behavior for tuning
if TUNNING:
  print("")
  RNG_SEED = 42
  pytorch_lightning.seed_everything(RNG_SEED)
else:
   print("\nSeeds set to random values")

# *************** Dataset naming constants ********************
DATASET = "datasets/Dataset_noticias.csv"
TRAIN_DATASET = "datasets/Train_Noticias.csv"
TEST_DATASET = "datasets/Test_Noticias.csv"

# *************** Tokenization config constants ***************
TEST_SIZE = 0.2
MAX_TOKEN_COUNT = 512

# *************** Dataset manipulation constants ***************
# dataframe from the dataset
DATA = pandas.read_csv(DATASET)

# Name of the noticia cell
NOTICIA = DATA.columns.tolist()[5]

# Name of the classes from the dataframe
CLASS_NAMES = DATA.columns.tolist()[6:]
CLASS_NUM = len(CLASS_NAMES)

# *************** Model config constants ***********************
MUTLILINGUAL = "bert-base-multilingual-cased"
BETO_MLDOC = "dccuchile/bert-base-spanish-wwm-cased-finetuned-mldoc"
BETO_PAWSX = "dccuchile/bert-base-spanish-wwm-cased-finetuned-pawsx"
BETO_XNLI = "dccuchile/bert-base-spanish-wwm-cased-finetuned-xnli"

MODEL_NAME = MUTLILINGUAL

METRICS = MetricCollection(
        {
            "Macro Avg-Accuracy": MultilabelAccuracy(
              num_labels=CLASS_NUM, average="macro"),
            "Micro Avg Accuracy": MultilabelAccuracy(
              num_labels=CLASS_NUM, average="micro"),
            "Recall": MultilabelRecall(num_labels=CLASS_NUM),
            "Precision": MultilabelPrecision(num_labels=CLASS_NUM),
            "F1 Score": MultilabelF1Score(num_labels=CLASS_NUM),
        }
)
LOSS_FUNCTION = torch.nn.BCELoss()
BATCH_SIZE = 8

if TUNNING:
   N_EPOCHS = 1
else:
   N_EPOCHS = 20

LEARNING_RATE = 2e-5

# The activation function and the optimization algorithm should
# be here, but their constructions depends on the model data
# so they cannot be included here