from sklearn.metrics import accuracy_score, f1_score
import statistics
import numpy as np


def evaluate(pred_labels, test_labels):
    pred_labels1 = np.array_split(pred_labels, 5)
    test_labels1 = np.array_split(test_labels, 5)
    accuracy = []
    f1 = []
    for test, pred in zip(test_labels1, pred_labels1):
        accuracy.append(accuracy_score(test, pred))
        print(f1_score(test, pred, average="weighted"))
        f1.append(f1_score(test, pred, average="weighted"))

    print("Accuracy: " + str(sum(accuracy) / len(accuracy)))
    print("Standard Deviation" + str(statistics.stdev(accuracy)))

    print("F1 Score: " + sum(f1) / len(f1))
    print("Standard Deviation: " + statistics.stdev(f1))
