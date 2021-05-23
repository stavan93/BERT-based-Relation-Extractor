def read_data(tags):
    context_text = []
    context_tags = []

    for tagg in tags:
        context_text.append(tagg[1])
    return context_text


def read_data(texts, tags):
    context_text = []
    context_tags = []

    for textt in texts:
        for tagg in tags:
            if int(textt[0]) == int(tagg[0]):
                context_text.append(textt[1])
                context_tags.append(tagg[1])
            else:
                continue
    return context_text, context_tags


def read_labels(dataa):
    labels = []
    count = 0
    for dat in dataa:
        for ent in dat["entities"]:
            labels.append(int(ent["label"]))
            count += 1
    print(count)
    return labels


def tokenizeData(tokenizer, text, max_length=256):
    input_ids = []
    attention_masks = []

    for tx in text:
        tokenizedData = tokenizer.encode_plus(tx, max_length=max_length,
                                              padding='max_length', truncation="longest_first")
        tokenizedQP = tokenizedData["input_ids"]
        attentionMask = tokenizedData["attention_mask"]

        input_ids.append(tokenizedQP)
        attention_masks.append(attentionMask)

    return np.array(input_ids), np.array(attention_masks)


def buildDataLoaders(batchSize, trainFeatures, testFeatures):
    trainTensors = [torch.tensor(feature, dtype=torch.long) for feature in trainFeatures]
    testTensors = [torch.tensor(feature, dtype=torch.long) for feature in testFeatures]

    trainDataset = TensorDataset(*trainTensors)
    testDataset = TensorDataset(*testTensors)

    trainSampler = RandomSampler(trainDataset)
    testSampler = SequentialSampler(testDataset)

    trainDataloader = DataLoader(trainDataset, sampler=trainSampler, batch_size=batchSize)
    testDataloader = DataLoader(testDataset, sampler=testSampler, batch_size=batchSize)

    return trainDataloader, testDataloader


def train(numEpochs, gradSteps, model, optimizer, trainDataLoader):
    trainLossHistory = []

    for _ in tqdm(range(numEpochs), desc="Training Epoch's"):

        # Train the model for fine-tuning
        epochTrainLoss = 0  # Cumulative loss
        model.train()
        model.zero_grad()

        for step, batch in enumerate(trainDataLoader):
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            label = batch[2].to(device)
            outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks, labels=label)

            loss = outputs[0]
            loss = loss / gradSteps
            epochTrainLoss += loss.item()
            loss.backward()

            if (step + 1) % gradSteps == 0:  # Gradient accumulation is over
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clipping gradients
                optimizer.step()
                model.zero_grad()

        epochTrainLoss = epochTrainLoss / len(trainDataLoader)
        trainLossHistory.append(epochTrainLoss)


def predict(tokenizer, model, text, tag, max_length=256):
    sequence = tokenizer.encode_plus(tag, text, max_length=max_length,
                                     padding='max_length', truncation="longest_first"
                                     , return_tensors="pt")['input_ids'].to(device)

    logits = model(sequence)[0]
    probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
    if probabilities[1] > 0.5:
        return 1
    return 0

def predict(tokenizer, model, text, max_length=256):
    sequence = tokenizer.encode_plus(text, max_length=max_length,
                                     padding='max_length', truncation="longest_first"
                                     , return_tensors="pt")['input_ids'].to(device)

    logits = model(sequence)[0]
    probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
    if probabilities[1] > 0.5:
        return 1
    return 0
