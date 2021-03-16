# ##CONFIGURATE VARIABLES FOR MY MODEL


# #from transformers import AutoTokenizer, AutoModelForMaskedLM
# import transformers

# max_sentencelen = 40
# train_size = 1629
# valid_size = 543
# test_size = 543
# epochs = 7


# tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

# #model = transformers.AutoModelForMaskedLM.from_pretrained("bert-base-uncased)

# model = transformers.BertTokenizer.from_pretrained("bert-base-uncased")


import transformers

MAX_LEN = 220

TRAIN_BATCH_SIZE = 29
VALID_BATCH_SIZE = 6

EPOCHS = 10

MODEL_PATH = "NNIA2020/21.bin"

TRAINING_FILE = "..\houda\datasetsv.tsv"
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
#model = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
model = transformers.BertModel.from_pretrained("bert-base-uncased")
