import pandas as pd
import numpy as np
import csv

from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, label_binarize

from ucimlrepo import fetch_ucirepo

def naive_bayes(dataset_id):
    datab = fetch_ucirepo(id=dataset_id) 
    X = datab.data.features
    y = datab.data.targets

    df = pd.concat([X, y], axis=1)
    df.rename(columns={y.columns[0]: "Label"}, inplace=True)
    
    df.columns
    np.random.seed(0)
    msk = np.random.rand(len(df)) < 0.8
    X_train = df[msk]
    y_train = X_train['Label']

    X_test = df[~msk]
    y_test = X_test['Label']
    X_test = X_test.drop(['Label'],axis=1)
    N, M = df.shape
    
    Likelihood_Dict= {
    } 
    Priors={
    
    }
    Dict_Labels = dict()
    
        
    def NaiveBayesTrain(train,targetCol):
        likelihood_dict= dict() 
        priors=dict()
        data = np.array(train)
        r,c= train.shape
        classes = list(train[targetCol].unique())
        columns = list(train.columns)

        prior = train.groupby(targetCol)[targetCol].agg(['count'])
        priors =  ((prior).T.to_dict())  
        
        for key in priors.keys():
            val = priors.get(key)
            for elm in val:
                val = round((val.get(elm)/r),2)
                priors.update({key:val})
                
        def divide(dic, count):
            n_dic = dict()
            for key in dic.keys():
                val = round(dic.get(key)/count,2)
                n_dic.update({key:val})
            return n_dic
        
        def cal_likelihood(data,cls):
            n,m = data.shape
            
            for col in columns:
                dic = dict()
                att_val = (data[col].unique())
                val = (data.groupby(col)[col].count()).to_dict()
                val =divide(val,n) 
                dic.update({col:val})

                if cls in likelihood_dict.keys():
                    val = likelihood_dict.get(cls)
                    val.update(dic)
                else:    
                    likelihood_dict.update({cls:dic})
                    
        for cls in classes:
            data_ = train.loc[train[targetCol]== cls]
            cal_likelihood(data_,cls)

        return likelihood_dict,priors

    def sorted_prior():
        rlist = list()
        for key in Likelihood_Dict.keys():
            if key in Priors.keys():
                rlist.append(Priors.get(key))        
        return (rlist) 
    
    def feature_post(data,column):
        result =np.zeros(shape=(len(data),))
        
        for key in Likelihood_Dict.keys():
            val = Likelihood_Dict.get(key)
            col = val.get(column)
            res =[]
            
            for dat in data:
                if dat in col.keys():
                    res.append(col.get(dat))
                else:
                    res.append('0.001')
                    
            res = np.array(res,dtype = float)
            result =np.column_stack((result,res))
            
        return (result[:,1:]) 

    def NaiveBayesPredict(test,target):
        
        X_test = np.array(test)
        prediction = np.ones(shape=(len(X_test),len(Priors)))
        columns = list(test.columns)
        labels = list(Likelihood_Dict.keys())
        dict_labels= dict()

        prior = list(Priors.values())
        rlist = sorted_prior()
        
        for i in range(len(columns)):
            ff = X_test[:,i]
            pred = feature_post(ff,columns[i])
            prediction  = prediction*pred
            

        prediction = prediction*rlist
        predict = np.argmax(prediction,axis=1)
        
        for i in range(len(labels)):
            dict_labels.update({i:labels[i]})

        predict =  (predict.reshape(-1,1))
        predict = pd.DataFrame(predict)
        predict = np.array(predict.replace(dict_labels))

        predict =  (predict.T[0])
        return predict

    Likelihood_Dict, Priors = NaiveBayesTrain(X_train,"Label")
    y_pred = NaiveBayesPredict(X_test, y_test)

    y_test_array = np.array(y_test)
    y_pred_array = np.array(y_pred)

    acc = accuracy_score(y_test_array, y_pred_array)
    prec = precision_score(y_test_array, y_pred_array, average='macro')  # or 'weighted'
    f1 = f1_score(y_test_array, y_pred_array, average='macro')
    rec = recall_score(y_test_array, y_pred_array, average='macro')  # or 'weighted'

    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test_array)
    y_pred_bin = lb.transform(y_pred_array)

    try:
        auc = roc_auc_score(y_test_bin, y_pred_bin, average='macro')
    except ValueError:
        auc = 0
        
    return acc, prec, f1, rec, auc


def decision_tree(dataset_id):
    datab = fetch_ucirepo(id=dataset_id) 
    X = datab.data.features
    y = datab.data.targets

    df = pd.concat([X, y], axis=1)
    df.rename(columns={y.columns[0]: "Label"}, inplace=True)
    
    if 'children' in df.columns:
        df['children'] = df['children'].replace('more', '100')

    label_encoders = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    np.random.seed(0)
    msk = np.random.rand(len(df)) < 0.8
    train_data = df[msk].values.tolist()
    test_data = df[~msk].values.tolist()
    
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

    tree = build_tree(train_data)
    
    y_true = []
    y_pred = []

    for row in test_data:
        pred_dist = classify(row, tree)
        pred = predict_label(pred_dist)
        y_true.append(row[-1])
        y_pred.append(pred)
        
    average_type = "weighted" if len(set(y_true)) > 2 else "binary"
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    prec = precision_score(y_true_arr, y_pred_arr, average=average_type, zero_division=0)
    recall = recall_score(y_true_arr, y_pred_arr, average=average_type, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_arr, average=average_type, zero_division=0)
    auc = 0
    try:
        y_true_bin = label_binarize(y_true_arr, classes=list(set(y_true_arr)))
        y_pred_bin = label_binarize(y_pred_arr, classes=list(set(y_true_arr)))
        auc = roc_auc_score(y_true_bin, y_pred_bin, average="macro", multi_class="ovo")
        # print("AUC      : %.4f" % auc)
    except:
        # print("AUC: Cannot be calculated for single-class or invalid prediction case.")
        auc = 0
        
    return acc, prec, recall, f1, auc

def main(id):
    dataset_id = id
    
    nb_acc = 0
    nb_prec = 0
    nb_rec = 0
    nb_f1 = 0
    nb_auc = 0
    
    dt_acc = 0
    dt_prec = 0
    dt_rec = 0
    dt_f1 = 0
    dt_auc = 0
    
    cycle_count = 10
    
    for i in range(cycle_count):
        ac, pr, rc, f1, au = naive_bayes(dataset_id)
        nb_acc += ac
        nb_prec += pr
        nb_rec += rc
        nb_f1 += f1
        nb_auc += au
        
        ac, pr, rc, f1, au = decision_tree(dataset_id)
        dt_acc += ac
        dt_prec += pr
        dt_rec += rc
        dt_f1 += f1
        dt_auc += au
        
    nb_acc /= cycle_count
    nb_prec /= cycle_count
    nb_rec /= cycle_count
    nb_f1 /= cycle_count
    nb_auc /= cycle_count
    
    dt_acc /= cycle_count
    dt_prec /= cycle_count
    dt_rec /= cycle_count
    dt_f1 /= cycle_count
    dt_auc /= cycle_count
    
    print("Final Result:")
    print("Naive-Bayes")
    print(f"Accuracy  : {nb_acc:.4f}")
    print(f"Precision : {nb_prec:.4f}")
    print(f"Recall    : {nb_rec:.4f}")
    print(f"F1 Score  : {nb_f1:.4f}")
    print(f"AUC       : {nb_auc:.4f}")
    
    print("\nDecision Tree")
    print(f"Accuracy  : {dt_acc:.4f}")
    print(f"Precision : {dt_prec:.4f}")
    print(f"Recall    : {dt_rec:.4f}")
    print(f"F1 Score  : {dt_f1:.4f}")
    print(f"AUC       : {dt_auc:.4f}")
    
    
if __name__ == "__main__":
    main(89) # for spambase dataset