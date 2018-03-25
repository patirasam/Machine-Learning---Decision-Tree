from __future__ import print_function

from random import seed
import random

total_data = []
temp = []
file = open("Car_dataset.txt", "r")
for line in file.readlines():
    temp = line.strip().split(",")
    total_data.append(temp)
seed(2)
random.shuffle(total_data)

total_data1 = total_data[0:1383]
train_data = total_data1[0:1106]
test_data = total_data1[1106:-1]
validate_set = total_data[1383:-1]
tcount = 0
fcount = 0

sample_data = train_data

header = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']


def uniq_values(rows, col):
    return set([row[col] for row in rows])


def classcount(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


class classifying_question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


def tree_division(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


import math


def gini(rows):
    counts = classcount(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl ** 2
    return impurity


def entropy(rows): #Calculates Entropy
    counts = classcount(rows)
    entropy_val = 0
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        entropy_val -= prob_of_lbl * math.log(prob_of_lbl, 2)
    return entropy_val


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def best_split(rows):
    bestgain = 0
    bestquestion = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):

        values = set([row[col] for row in rows])

        for val in values:

            question = classifying_question(col, val)

            true_rows, false_rows = tree_division(rows, question)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= bestgain:
                bestgain, bestquestion = gain, question

    return bestgain, bestquestion


class LeafNode:
    def __init__(self, rows):
        self.predictions = classcount(rows)


class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def decision_tree(rows):
    gain, question = best_split(rows)

    if gain == 0:
        return LeafNode(rows)
    true_rows, false_rows = tree_division(rows, question)
    true_branch = decision_tree(true_rows)
    false_branch = decision_tree(false_rows)
    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    if isinstance(node, LeafNode):
        print(spacing + "Predict", node.predictions)
        return
    print(spacing + str(node.question))
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


my_tree = decision_tree(sample_data)


def classify(row, node):
    if isinstance(node, LeafNode):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


tcount = 0
fcount = 0


def print_acc(label, row):
    global tcount, fcount
    for key in label:
        temp = key
    tlabel = row[-1]
    if temp == tlabel:
        tcount += 1
    else:
        fcount += 1
    return temp


for row in test_data:
    print("Actual: %s. Predicted: %s" %
          (row[-1], print_acc(classify(row, my_tree), row)))


def accuracy():
    print(tcount / (tcount + fcount) * 100)


def errRate():
    print(fcount / (tcount + fcount) * 100)


print_tree(my_tree)