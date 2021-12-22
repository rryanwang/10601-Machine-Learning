import numpy as np
import sys
from collections import Counter
import pandas as pd


class DecisionStump:
    def __init__(self, attribute_index):
        self.map = {}
        self.attribute_index = int(attribute_index)

    def train(self, train_x, train_y):
        binary_values = np.unique(train_x)
        for val in binary_values:
            self.map[val] = Counter(train_y[train_x[:, self.attribute_index] == val]).most_common(1)[0][0]

    def predict(self, x):
        y = []
        for row in x:
            y.append(self.map[row[self.attribute_index]])
        return y


def get_error(y, y_hat):
    incorrect_count = 0
    for i in range(len(y)):
        if y[i] != y_hat[i]:
            incorrect_count += 1
    return incorrect_count / len(y)


def main(train_infile, test_infile, attribute_index, train_labels, test_labels, metrics):
    train_dataset = pd.read_csv(train_infile, sep='\t').to_numpy()
    test_dataset = pd.read_csv(test_infile, sep='\t').to_numpy()
    train_x = train_dataset[:, :-1]
    train_y = train_dataset[:, -1]
    test_x = test_dataset[:, :-1]
    test_y = test_dataset[:, -1]

    dStump = DecisionStump(attribute_index)
    dStump.train(train_x, train_y)

    train_y_predict = dStump.predict(train_x)
    test_y_predict = dStump.predict(test_x)
    train_error = get_error(train_y, train_y_predict)
    test_error = get_error(test_y, test_y_predict)

    with open(train_labels, 'w') as fw:
        for label in train_y_predict:
            fw.write(label + '\n')

    with open(test_labels, 'w') as fw:
        for label in test_y_predict:
            fw.write(label + '\n')

    with open(metrics, 'w') as fw:
        fw.write(f'error(train): {str(train_error)}\n')
        fw.write(f'error(test): {str(test_error)}')


if __name__ == '__main__':
    main(sys.argv[1],
         sys.argv[2],
         sys.argv[3],
         sys.argv[4],
         sys.argv[5],
         sys.argv[6])

    # train_infile = sys.argv[1]
    # test_infile = sys.argv[2]
    # attribute_index = sys.argv[3]
    # train_labels = sys.argv[4]
    # test_label = sys.argv[5]
    # metrics = sys.argv[6]
    #
    # df1 = pd.read_csv(train_infile, sep='\t')
    # df2 = pd.read_csv(test_infile, sep='\t')
    # train_dataset = df1.to_numpy()
    # test_dataset = df2.to_numpy()
    # train_x = train_dataset[:, :-1]
    # train_y = train_dataset[:, -1]
    # test_x = test_dataset[:, :-1]
    # test_y = test_dataset[:, -1]
    #
    # dStump = DecisionStump(attribute_index)
    # dStump.train(train_x, train_y)
    # train_error = dStump.get_error(train_y, dStump.predict(train_x))
    # test_error = dStump.get_error(test_y, dStump.predict(test_x))
    #
    # with open(metrics, 'w') as wf:
    #     wf.write(str(train_error) + '\n')
    #     wf.write(str(test_error))

# python3 sss.py politicians_train.tsv politicians_test.tsv 3 pol_3_train.labels pol_3_test.labels pol_3_metrics.txt
# python3 sss.py education_train.tsv education_test.tsv 5 edu_5_train.labels edu_5_test.labels edu_5_metrics.txt
