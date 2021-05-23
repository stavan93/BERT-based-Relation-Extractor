!pip install pandas
!pip install numpy
!pip install transformers
!pip install torch torchvision
!pip install sentencepiece

import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer, RobertaForSequenceClassification, AdamW, BertForSequenceClassification
from transformers import DistilBertForTokenClassification, Trainer, TrainingArguments
import BERT_Entity_Marker, BERT_Tag_Sentence, Roberta_Entity_Marker, Roberta_Tag_Sentence


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    file = input()

    print("Select a model to train:")
    print("1.) BERT")
    print("2.) RoBERTa")

    model = input()

    if model == 1:
        print("Select a method:")
        print("Entity Marker A")
        print("Entity Marker B")
        print("Entity Marker C")
        print("Tag Sentence")

        method = input()

        if method == 1:
            BERT_Entity_Marker.entity_marker_a(file)
        elif method == 2:
            BERT_Entity_Marker.entity_marker_b(file)
        elif method == 3:
            BERT_Entity_Marker.entity_marker_c(file)
        else:
            BERT_Tag_Sentence.tag_sentence(file)

    elif model == 2:
        print("Select a method:")
        print("Entity Marker A")
        print("Entity Marker B")
        print("Entity Marker C")
        print("Tag Sentence")

        method = input()

        if method == 1:
            Roberta_Entity_Marker.entity_marker_a(file)
        elif method == 2:
            Roberta_Entity_Marker.entity_marker_b(file)
        elif method == 3:
            Roberta_Entity_Marker.entity_marker_c(file)
        else:
            Roberta_Tag_Sentence.tag_sentence(file)

    else:
        print("Please select 1 or 2.")
