# import transformers
from transformers import BertTokenizer, BertForSequenceClassification
# from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import os

NUM_LABELS = 3
OUTPUT_ATTENTIONS = False
OUTPUT_HIDDEN_STATES = True

DIR_ROOT = os.getcwd()
TRAIN_DATA_PATH = "train.csv" #os.path.join(DIR_ROOT,"..\\train.csv")
TEST_DATA_PATH = "test.csv" #os.path.join(DIR_ROOT,"..\\test.csv")
SUBMISSION_PATH = "sample_submission.csv" #os.path.join("..\\sample_submission.csv")

BERT_PATH = os.path.join(DIR_ROOT,"..\\Bert-base-uncased")

BERT_DOWNLOAD_PATH = 'bert-base-uncased'

FEATURES = ['title1_en','title2_en']
TARGET = ['label']
READ = True

# config = AutoConfig.from_pretrained('bert-base-uncased',num_labels = NUM_LABELS, output_attentions = OUTPUT_ATTENTIONS, output_hidden_states = OUTPUT_HIDDEN_STATES)
# MODEL = AutoModelForSequenceClassification.from_config(config)

# TOKENIZER = AutoTokenizer.from_pretrained("bert-base-uncased")

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
MODEL = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels = NUM_LABELS, output_attentions = OUTPUT_ATTENTIONS, output_hidden_states = OUTPUT_HIDDEN_STATES)
#transformers.BertTokenizer.from_pretrained(BERT_PATH)
#MODEL = AutoModelForSequenceClassification.from_config(config=TOKENIZER)
#transformers.BertForSequenceClassification.from_pretrained(BERT_PATH,num_labels = NUM_LABELS, output_attentions = OUTPUT_ATTENTIONS, output_hidden_states = OUTPUT_HIDDEN_STATES)
MODEL_PATH = "final_svm_model.pkl" #os.path.join(DIR_ROOT,"../Saved_Model/final_svm_model.pkl")

MAX_LEN = 64
BATCH_SIZE = 1