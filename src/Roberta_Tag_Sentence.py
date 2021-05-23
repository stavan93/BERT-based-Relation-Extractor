import Read
import json
import torch
import numpy as np
import Evaluate
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer, RobertaForSequenceClassification, AdamW, \
    BertForSequenceClassification
from transformers import DistilBertForTokenClassification, Trainer, TrainingArguments

def tag_sentence(file):
    with open(file) as f:
        data = json.load(f)

    text_list = []
    tag_list = []

    for d in data:
        text_list.append([d["id"], d["text"].strip("\n")])
        for ent in d["entities"]:
            tag_list.append([d["id"], d["text"].replace(ent["arg1"] + " ", "ENTITY1 ").replace(" " + ent["arg2"] + " ",
                                                                                               " ENTITY2 ").strip(
                "\n")])

    text, tags = Read.read_data(text_list, tag_list)
    labels = Read.read_labels(data)

    train_texts, test_texts, train_tags, test_tags, train_labels, test_labels = train_test_split(text, tags,
                                                                                                 labels, test_size=.2)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_ids, train_attn = Read.tokenizeData(tokenizer, train_texts, train_tags)
    test_ids, test_attn = Read.tokenizeData(tokenizer, test_texts, test_tags)

    trainFeatures = (train_ids, train_attn, train_labels)
    testFeatures = (test_ids, test_attn)

    trainDataLoader, testDataLoader = Read.buildDataLoaders(8, trainFeatures, testFeatures)

    model = RobertaForSequenceClassification.from_pretrained('roberta-base').to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    Read.train(3, 3, model, optimizer, trainDataLoader)

    pred_labels = []
    for text, tags in zip(test_texts, test_tags):
        pred_labels.append(Read.predict(tokenizer, model, text, tags))

    Evaluate.evaluate(pred_labels, test_labels)