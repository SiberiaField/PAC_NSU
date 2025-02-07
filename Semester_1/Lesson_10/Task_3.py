import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from digit_classificator import DigitClassificator
from dataset import get_dataset

def testing_with_digits(test, dc : DigitClassificator):
    test_num = len(test)
    for digit in range(10):
        correct_num = 0
        for x in test:
            predict = dc.binary_step(dc.predict_by_digit(digit, x[0]))
            x_label = np.argmax(x[1])
            if x_label == digit:
                if (predict == 1 and x_label == digit) or (predict == 0 and x_label != digit):
                    correct_num += 1
        accuracy = correct_num / test_num
        accuracy = int(np.round(accuracy, 2) * 100)
        print(f"{digit} accuracy: {accuracy}%")
    print()

def testing(test, dc : DigitClassificator, analysing = True) -> None | list:
    confusion_matrix = np.zeros((10, 10), dtype=np.int64)
    res = []
    for x in test:
        raw_predict = dc.predict(x[0])
        actual_label = np.argmax(x[1])
        predicted_label = np.argmax(raw_predict)
        confusion_matrix[actual_label][predicted_label] += 1
        res.append(np.reshape(raw_predict, 10))

    if analysing:
        # print(f"Confusion matrix:\n{confusion_matrix}\n")

        TP = np.diag(confusion_matrix)
        FP = np.sum(confusion_matrix - np.diag(TP), 0)
        FN = np.sum(confusion_matrix - np.diag(TP), 1)

        res = {"precision" : [], "recall" : []}
        for digit in range(10):
            precision = TP[digit] / (TP[digit] + FP[digit])
            recall = TP[digit] / (TP[digit] + FN[digit])
            precision = np.round(precision, 2) * 100
            recall = np.round(recall, 2) * 100

            res["precision"].append(precision)
            res["recall"].append(recall)

        df = pd.DataFrame(res)
        print(df)
        print()
    else:
        return res

def make_scatters(model : TSNE, data : np.ndarray):
    squeezed = model.fit_transform(data)
    plt.scatter(squeezed[:30][:, 0], squeezed[:30][:, 1], color='green', label='0')
    plt.scatter(squeezed[30:60][:, 0], squeezed[30:60][:, 1], color='red', label='1')
    plt.scatter(squeezed[60:90][:, 0], squeezed[60:90][:, 1], color='cyan', label='2')
    plt.scatter(squeezed[90:120][:, 0], squeezed[90:120][:, 1], color='blue', label='3')
    plt.scatter(squeezed[120:150][:, 0], squeezed[120:150][:, 1], color='orange', label='4')
    plt.scatter(squeezed[150:180][:, 0], squeezed[150:180][:, 1], color='purple', label='5')
    plt.scatter(squeezed[180:210][:, 0], squeezed[180:210][:, 1], color='olive', label='6')
    plt.scatter(squeezed[210:240][:, 0], squeezed[210:240][:, 1], color='pink', label='7')
    plt.scatter(squeezed[240:270][:, 0], squeezed[240:270][:, 1], color='gray', label='8')
    plt.scatter(squeezed[270:300][:, 0], squeezed[270:300][:, 1], color='black', label='9')
    plt.legend()

def draw_TSNE(test, dc : DigitClassificator):
    labels_vectors = []
    predicts = []
    for digit in range(10):
        count = 0
        labels = []
        for x in test:
            if np.argmax(x[1]) == digit:
                labels.append(x)
                count += 1
            if count == 30:
                break
        predicts.extend(testing(labels, dc, False))
        labels_vectors.extend([np.reshape(x[0], 784) for x in labels])
        labels.clear()

    predicts = np.array(predicts)
    labels_vectors = np.array(labels_vectors)

    model = TSNE()
    plt.figure(figsize=(6, 6))
    plt.title("Samples from Testing Data")
    make_scatters(model, labels_vectors)

    plt.figure(figsize=(6, 6))
    plt.title("Results from Testing Data")
    make_scatters(model, predicts)

    plt.show()
    
def main():
    print("\nGet dataset...")
    train, test = get_dataset()
    train.sort(key = lambda elem: np.argmax(elem[1]))
    test.sort(key = lambda elem: np.argmax(elem[1]))
    print(f"Complete: train len - {len(train)}, test len - {len(test)}\n")

    print("Initialize classificator...")
    bias = 45
    print(f"b = {bias}")
    biases = np.repeat(bias, 10)
    dc = DigitClassificator(train, biases)
    print("Complete\n")

    print("Testing...")
    testing_with_digits(test, dc)
    testing(test, dc, True)
    print("Drawing TSNE...")
    draw_TSNE(test, dc)
    print()
main()