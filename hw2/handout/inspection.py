import numpy as np
import pandas as pd
import sys
from collections import Counter


class Inspection:
    def __init__(self):
        self.entropy = 0
        self.error_rate = 0

    def get_entropy(self, y):
        labelCounter = Counter(y)
        for label in labelCounter:
            prob = labelCounter[label] / len(y)
            self.entropy -= prob * np.log2(prob)
        return self.entropy

    def get_error_rate(self, y):
        labelCounter = Counter(y)
        label_occurrence = []
        for label in labelCounter:
            label_occurrence.append(labelCounter[label])

        if label_occurrence[0] < label_occurrence[1]:
            self.error_rate = label_occurrence[0] / len(y)
        elif label_occurrence[0] > label_occurrence[1]:
            self.error_rate = label_occurrence[1] / len(y)
        else:
            self.error_rate = 0.5

        return self.error_rate


if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]
    df = pd.read_csv(infile, sep='\t')
    dataset = df.to_numpy()
    x = dataset[:, :-1]
    y = dataset[:, -1]
    IP = Inspection()
    entropy = str(IP.get_entropy(y))
    error_rate = str(IP.get_error_rate(y))
    with open(outfile, 'w') as wf:
        wf.write('entropy: ' + entropy + '\n')
        wf.write('error: ' + error_rate)


# python3 inspection.py small_train.tsv 123.txt
