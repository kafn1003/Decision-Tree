#for data preparation
import pandas as pd

#for modeling
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#for evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#for graph
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image 
import pydotplus

def importdata():
    heart_data = pd.read_csv('heart.csv', sep = ',', header = None, names = ['data'])
    
    print("Dataset", heart_data)
    
    heart_data = list(heart_data['data'].apply(lambda x:x.split(" ") ))
    
﻿
    col_names =['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholesterol',
    'fasting blood sugar', 'resting electrocardiographic results', 'maximum heart rate achieved',
    'exercise induced angina', 'oldpeak', 'slope of the peak exercise', 'number of major vessels', 'thal', 'label']

    heart_dataframe = pd.DataFrame(heart_data, columns col_names)

    print ("\nDataset Length: ", len(heart_dataframe))
    print ("Dataset Shape: ", heart_dataframe.shape)

    print ("Dataset: ", heart_dataframe)

    return heart_dataframe

def splitdataset(heart_dataframe):
    X = heart_dataframe.values[:, 0:13]
    Y = heart_dataframe.values[:, 13]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
    
    return X, Y, X_train, X_test, Y_train, Y_test

def train_using_entropy(X_train, Y_train):
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=4, min_samples_leaf=5)
    
    clf_entropy.fit(X_train,Y_train)
    
    return clf_entropy
    
def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    
    print("Predicted values:")
    print(y_pred)
    print("\nTest Data: ", len(y_pred))
    
    return y_pred
    
def cal_accuracy(Y_test, y_pred):
    print("Confusion Matrix: \n", confusion_matrix(Y_test, y_pred)
    
    print("\nAccuracy: ", accuracy_score(Y_test, y_pred)*100)
    
    print("Report: ", classification_report(Y_test, y_pred))
    
   ﻿
def decisionTreeGraph(c1f_entropy):
    feature_cols = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholesterol',
    'fasting blood sugar', 'resting electrocardiographic results', 'maximum heart rate achieved', 'exercise induced angina',
    'oldpeak', 'slope of the peak exercise', 'number of major vessels', 'thal']

    dot_data StringIO()
    export_graphviz(clf_entropy, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_cols,
                    class_names=clf_entropy.classes_)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('heart_data.png')
    Image(graph.create_png())
    

def main():
    data = importdata()
    X, Y, X_train, X_test, Y_train,Y_test = splitdataset(data)
    clf_entropy = train_using_entropy(X_train, Y_train)
    
    print("\nResults using Entropy:")
    
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(Y_test, y_pred_entropy)
    
    decisionTreeGraph(clf_entropy)
    
main()