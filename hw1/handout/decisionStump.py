import numpy as np
import sys
from collections import Counter


class DecisionStump:
    def __init__(self, attribute: str):
        self.map = {}
        self.attribute_index = int(attribute)

    '''
    train_x: numpy array
    train_y: numpy array
    '''

    def train(self, train_x, train_y):
        binary_values = np.unique(train_x)
        for i in binary_values:
            self.map[i] = \
                Counter(train_y[train_x[:, self.attribute_index] == i]).most_common(
                    1)[0][0]

    '''
    x: list or numpy array

    return: list
    '''

    def predict(self, x):
        y = []
        for row in x:
            y.append(self.map[row[self.attribute_index]])
        return y


'''
Y: the true result
Y_hat: the predicted result
'''


def error(Y, Y_hat):
    incorrect_num = 0
    for i in range(len(Y)):
        if Y[i] != Y_hat[i]:
            incorrect_num += 1
    return incorrect_num / len(Y)


def main(train_infile, test_infile, attribute, train_label_outfile,
         test_label_outfile, metric_outfile):
    train_data = np.genfromtxt(train_infile, delimiter='\t', skip_header=1,
                               dtype=object)
    test_data = np.genfromtxt(test_infile, delimiter='\t', skip_header=1,
                              dtype=object)

    train_x = train_data[:, :-1]
    train_y = train_data[:, -1]
    test_x = test_data[:, :-1]
    test_y = test_data[:, -1]

    dtree = DecisionStump(attribute)

    dtree.train(train_x, train_y)

    train_y_predict = dtree.predict(train_x)
    test_y_predict = dtree.predict(test_x)
    train_error = error(train_y, train_y_predict)
    test_error = error(test_y, test_y_predict)

    # write train and test label to file
    with open(train_label_outfile, "wb") as fw:
        for i in train_y_predict:
            fw.write(i)
            fw.write(b'\n')

    with open(test_label_outfile, "wb") as fw:
        for i in test_y_predict:
            fw.write(i)
            fw.write(b'\n')

    with open(metric_outfile, "w") as fw:
        fw.write(f"error(train): {train_error}\n")
        fw.write(f"error(test): {test_error}\n")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5],
         sys.argv[6])

# python3 decisionStump.py politicians_train.tsv politicians_test.tsv 3 pol_3_train.labels pol_3_test.labels pol_3_metrics.txt
# python3 decisionStump.py education_train.tsv education_test.tsv 5 edu_5_train.labels edu_5_test.labels edu_5_metrics.txt