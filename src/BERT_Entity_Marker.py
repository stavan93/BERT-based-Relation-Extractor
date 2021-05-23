import Read
import json
import torch
import Evaluate
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import BertTokenizer, RobertaTokenizer, RobertaForSequenceClassification, AdamW, \
    BertForSequenceClassification
from transformers import DistilBertForTokenClassification, Trainer, TrainingArguments


def entity_marker_a(file):
    with open(file) as f:
        data = json.load(f)

    text_list = []
    tag_list = []

    for d in data:
        for ent in d["entities"]:
            tag_list.append(
                [d["id"],
                 d["text"].replace(ent["arg1"] + " ", "[e]" + ent["arg1"] + "[\e]").replace(" " + ent["arg2"] + " ",
                                                                                            "[e]" + ent[
                                                                                                "arg2"] + "[\e]").strip(
                     "\n")])

    text = Read.read_data(tag_list)
    labels = Read.read_labels(data)

    train_texts, test_texts, train_labels, test_labels = train_test_split(text, labels, test_size=.2)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    train_ids, train_attn = Read.tokenizeData(tokenizer, train_texts)
    test_ids, test_attn = Read.tokenizeData(tokenizer, test_texts)

    trainFeatures = (train_ids, train_attn, train_labels)
    testFeatures = (test_ids, test_attn)

    trainDataLoader, testDataLoader = Read.buildDataLoaders(8, trainFeatures, testFeatures)

    model = BertForSequenceClassification.from_pretrained('bert-base-cased').to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    Read.train(3, 3, model, optimizer, trainDataLoader)

    pred_labels = []
    for text in test_texts:
        pred_labels.append(Read.predict(tokenizer, model, text))

    Evaluate.evaluate(pred_labels, test_labels)


def entity_marker_b(file):
    with open(file) as f:
        data = json.load(f)

    text_list = []
    tag_list = []

    for d in data:
        for ent in d["entities"]:
            tag_list.append(
                [d["id"],
                 d["text"].replace(ent["arg1"] + " ", "[e] " + ent["arg1"] + " [\e]").replace(" " + ent["arg2"] + " ",
                                                                                            "[e] " + ent[
                                                                                                "arg2"] + " [\e]").strip(
                     "\n")])

    text = Read.read_data(tag_list)
    labels = Read.read_labels(data)

    train_texts, test_texts, train_labels, test_labels = train_test_split(text, labels, test_size=.2)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    train_ids, train_attn = Read.tokenizeData(tokenizer, train_texts)
    test_ids, test_attn = Read.tokenizeData(tokenizer, test_texts)

    trainFeatures = (train_ids, train_attn, train_labels)
    testFeatures = (test_ids, test_attn)

    trainDataLoader, testDataLoader = Read.buildDataLoaders(8, trainFeatures, testFeatures)

    model = BertForSequenceClassification.from_pretrained('bert-base-cased').to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    Read.train(3, 3, model, optimizer, trainDataLoader)

    pred_labels = []
    for text in test_texts:
        pred_labels.append(Read.predict(tokenizer, model, text))

    Evaluate.evaluate(pred_labels, test_labels)


def entity_marker_c(file):
    with open(file) as f:
        data = json.load(f)

    text_list = []
    tag_list = []

    for d in data:
        for ent in d["entities"]:
            tag_list.append(
                [d["id"],
                 d["text"].replace(ent["arg1"] + " ", "ENTITY1" + ent["arg1"]).replace(" " + ent["arg2"] + " ",
                                                                                            "ENTITY2" + ent[
                                                                                                "arg2"]).strip(
                     "\n")])

    text = Read.read_data(tag_list)
    labels = Read.read_labels(data)

    train_texts, test_texts, train_labels, test_labels = train_test_split(text, labels, test_size=.2)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    train_ids, train_attn = Read.tokenizeData(tokenizer, train_texts)
    test_ids, test_attn = Read.tokenizeData(tokenizer, test_texts)

    trainFeatures = (train_ids, train_attn, train_labels)
    testFeatures = (test_ids, test_attn)

    trainDataLoader, testDataLoader = Read.buildDataLoaders(8, trainFeatures, testFeatures)

    model = BertForSequenceClassification.from_pretrained('bert-base-cased').to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    Read.train(3, 3, model, optimizer, trainDataLoader)

    pred_labels = []
    for text in test_texts:
        pred_labels.append(Read.predict(tokenizer, model, text))

    Evaluate.evaluate(pred_labels, test_labels)