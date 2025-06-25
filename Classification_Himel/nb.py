# Converted from naivebayes.ipynb

import pandas as pd
import numpy as np
import csv

from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, recall_score
from sklearn.preprocessing import LabelBinarizer



df = pd.read_csv("iris.csv")
# df = pd.read_csv("heart.csv", delim_whitespace=True, header=None)
df.columns
np.random.seed(0)
msk = np.random.rand(len(df)) < 0.8
X_train = df[msk]
y_train = X_train['Label']

X_test = df[~msk]
y_test = X_test['Label']
X_test = X_test.drop(['Label'],axis=1)
N, M = df.shape

# train.info()

# train.Label.unique()

'''
Likelihood_Dict and Priors as global dictionary which would be updated by training and used while predicting
'''
Likelihood_Dict= {
} 
Priors={
   
}
Dict_Labels = dict()

# ---------------------------------------Training Naive Bayes--------------------------------------------    

    
def NaiveBayesTrain(train,targetCol):
    likelihood_dict= dict() 
    priors=dict()
    data = np.array(train)
    r,c= train.shape
    classes = list(train[targetCol].unique())
    columns = list(train.columns)
    
    
# Calculate prior probablity

    prior = train.groupby(targetCol)[targetCol].agg(['count'])
    priors =  ((prior).T.to_dict())  
    
    for key in priors.keys():
        val = priors.get(key)
        for elm in val:
            val = round((val.get(elm)/r),2)
            priors.update({key:val})
            
# Helper function    

    def divide(dic, count):
        n_dic = dict()
        for key in dic.keys():
            val = round(dic.get(key)/count,2)
            n_dic.update({key:val})
        return n_dic
    
# Calculate likelihood values

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
                
#  Update likeliood dictionary

    for cls in classes:
        data_ = train.loc[train[targetCol]== cls]
        cal_likelihood(data_,cls)

    return likelihood_dict,priors

# Helper function to sort the prior as per the likelihood dictionary. Would be used to map columns to list in further section


def sorted_prior():
    rlist = list()
    for key in Likelihood_Dict.keys():
        if key in Priors.keys():
            rlist.append(Priors.get(key))        
    return (rlist)        

# Helper function that return the feature vs label matrix, column stacked over each other

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

# Convert to arrays if needed
y_test_array = np.array(y_test)
y_pred_array = np.array(y_pred)

# Compute metrics
acc = accuracy_score(y_test_array, y_pred_array)
prec = precision_score(y_test_array, y_pred_array, average='macro')  # or 'weighted'
f1 = f1_score(y_test_array, y_pred_array, average='macro')
rec = recall_score(y_test_array, y_pred_array, average='macro')  # or 'weighted'

# For AUC: Need binary/one-hot encoded labels
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test_array)
y_pred_bin = lb.transform(y_pred_array)

try:
    auc = roc_auc_score(y_test_bin, y_pred_bin, average='macro')
except ValueError:
    auc = "AUC not defined for this case."

print("Final Metrics:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC      : {auc:.4f}")
