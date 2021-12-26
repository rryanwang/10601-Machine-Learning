import numpy as np
import pandas as pd
import sys
from collections import Counter


def get_entropy(Y):
    numEntries = len(Y)
    labelCounts = Counter(Y)
    entropy = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        entropy -= prob * np.log2(prob)
    return entropy


def get_info_gain(X, Y):
    M, N = X.shape

    y_entropy = get_entropy(Y)
    IG = [y_entropy for _ in range(N)]

    for i in range(N):
        col = X[:, i]
        binary_val = np.unique(col)
        for val in binary_val:
            Y_hat = Y[col == val]
            IG[i] -= get_entropy(Y_hat) * len(Y_hat) / len(Y)
    return np.array(IG)


def get_split_index(X, Y):
    info_gain = get_info_gain(X, Y)
    split_index = info_gain.argmax()
    return split_index if info_gain[split_index] > 0 else None


class DecisionTree:
    def __init__(self, max_depth):
        # stem node
        self.max_depth = max_depth
        self.children = {}  # key is the attribute value, value is the child node
        self.split_index = None
        # leaf node
        self.label = None

    def train(self, X, Y):
        self.split_index = get_split_index(X, Y)

        # convert this node to a leaf node
        if self.max_depth <= 0 or self.split_index is None:
            split_index = None
            self.label = Counter(Y).most_common(1)[0][0]
            return

        # keep splitting
        binary_values = np.unique(X[:, self.split_index])
        for bv in binary_values:
            self.children[bv] = DecisionTree(self.max_depth - 1)
            self.children[bv].train(X[X[:, self.split_index] == bv], Y[X[:, self.split_index] == bv])

    def predict_recur(self, X):
        if self.split_index is None:
            return self.label
        return self.children[X[self.split_index]].predict_recur(X)

    def predict(self, X):
        Y = []
        for row in X:
            Y.append(self.predict_recur(row))
        return np.array(Y)


def get_error(Y, Y_hat):
    return np.sum(Y != Y_hat) / len(Y)


def main(train_infile, test_infile, max_depth, train_label_outfile,
         test_label_outfile, metric_outfile):
    df_train = pd.read_csv(train_infile, sep='\t')
    df_test = pd.read_csv(test_infile, sep='\t')
    train_dataset = df_train.to_numpy()
    test_dataset = df_test.to_numpy()
    train_x = train_dataset[:, :-1]
    train_y = train_dataset[:, -1]
    test_x = test_dataset[:, :-1]
    test_y = test_dataset[:, -1]

    dtree = DecisionTree(int(max_depth))
    dtree.train(train_x, train_y)

    train_label_predict = dtree.predict(train_x)
    test_label_predict = dtree.predict(test_x)

    train_error = get_error(train_y, train_label_predict)
    test_error = get_error(test_y, test_label_predict)

    with open(train_label_outfile, "w") as fw:
        for i in train_label_predict:
            fw.write(i)
            fw.write('\n')

    with open(test_label_outfile, "w") as fw:
        for i in test_label_predict:
            fw.write(i)
            fw.write('\n')

    with open(metric_outfile, "w") as fw:
        fw.write(f"error(train): {train_error}\n")
        fw.write(f"error(test): {test_error}\n")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
         sys.argv[6])

# python3 decisionTree.py small_train.tsv small_test.tsv 3 small_3_train.labels small_3_test.labels small_3_metrics.txt