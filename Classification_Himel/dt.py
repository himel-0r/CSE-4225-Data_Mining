import pandas as pd
import collections
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize
import numpy as np

# Load and preprocess CSV data
df = pd.read_csv("iris.csv")

# Handle "more" in "children" feature if present
if 'children' in df.columns:
    df['children'] = df['children'].replace('more', '100')

# Encode all categorical features
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split into train/test
np.random.seed(0)
msk = np.random.rand(len(df)) < 0.8
train_data = df[msk].values.tolist()
test_data = df[~msk].values.tolist()

# Helper: get class distribution
def eachTypeNumbers(lines):
    dic = {}
    for line in lines:
        label = line[-1]
        dic[label] = dic.get(label, 0) + 1
    return dic

class CompareValue:
    def __init__(self, feature, evaluate):
        self.feature = feature
        self.evaluate = evaluate

    def compare(self, row):
        return row[self.feature] == self.evaluate

def separate(rows, query):
    true_rows, false_rows = [], []
    for row in rows:
        (true_rows if query.compare(row) else false_rows).append(row)
    return true_rows, false_rows

def gini(rows):
    counts = eachTypeNumbers(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / len(rows)
        impurity -= prob_of_lbl ** 2
    return impurity

def gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

def find_best_split(rows):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = CompareValue(col, val)
            true_rows, false_rows = separate(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            g = gain(true_rows, false_rows, current_uncertainty)
            if g > best_gain:
                best_gain, best_question = g, question
    return best_gain, best_question

class Leaf:
    def __init__(self, rows):
        self.predictions = eachTypeNumbers(rows)

class Decision_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

def build_tree(rows):
    gain_val, question = find_best_split(rows)
    if gain_val == 0:
        return Leaf(rows)
    true_rows, false_rows = separate(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision_Node(question, true_branch, false_branch)

def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.compare(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def predict_label(distribution):
    return max(distribution, key=distribution.get)

# Build tree
tree = build_tree(train_data)

# Evaluate
y_true = []
y_pred = []

for row in test_data:
    pred_dist = classify(row, tree)
    pred = predict_label(pred_dist)
    y_true.append(row[-1])
    y_pred.append(pred)

# Metrics
average_type = "weighted" if len(set(y_true)) > 2 else "binary"
y_true_arr = np.array(y_true)
y_pred_arr = np.array(y_pred)

print("Accuracy : %.4f" % accuracy_score(y_true_arr, y_pred_arr))
print("Precision: %.4f" % precision_score(y_true_arr, y_pred_arr, average=average_type, zero_division=0))
print("Recall   : %.4f" % recall_score(y_true_arr, y_pred_arr, average=average_type, zero_division=0))
print("F1 Score : %.4f" % f1_score(y_true_arr, y_pred_arr, average=average_type, zero_division=0))

# AUC (only for binary or one-vs-rest multi-class)
try:
    y_true_bin = label_binarize(y_true_arr, classes=list(set(y_true_arr)))
    y_pred_bin = label_binarize(y_pred_arr, classes=list(set(y_true_arr)))
    auc = roc_auc_score(y_true_bin, y_pred_bin, average="macro", multi_class="ovo")
    print("AUC      : %.4f" % auc)
except:
    print("AUC: Cannot be calculated for single-class or invalid prediction case.")
