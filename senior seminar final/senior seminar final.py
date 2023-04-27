# Importing the required packages
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sn
import xgboost as xgb
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
  
# Function importing Dataset
def importdata():

    balance_data = pd.read_csv('financial_data.csv')#,sep= ',', header = None
    balance_data = balance_data.drop(columns = ['months_employed'])
    balance_data['personal_account_months'] = (balance_data.personal_account_m + (balance_data.personal_account_y * 12))
    balance_data[['personal_account_m', 'personal_account_y', 'personal_account_months']].head()
    #balance_data = balance_data.drop(balance_data.columns[[0]], axis=1)
    balance_data = balance_data.drop(columns = ['personal_account_m', 'personal_account_y','entry_id'])
      
    # Printing the dataset shape
    print ("Dataset Length: ", len(balance_data))
    print ("Dataset Shape: ", balance_data.shape)
      
    # Printing the dataset obseravtions
    print ("Dataset: ",balance_data.head())
    #convert to number
    
    balance_data=balance_data.replace("weekly",0)
    balance_data=balance_data.replace("bi-weekly",1)
    balance_data=balance_data.replace("semi-monthly",2)
    balance_data=balance_data.replace("monthly",3)
    # Printing the dataset obseravtions
    print ("Dataset: ",balance_data.head())
    corr_matrix = balance_data.corr()
    sn.heatmap(corr_matrix, annot=True)
    plt.show()
    print ("Dataset Length: ", len(balance_data))
    print ("Dataset Shape: ", balance_data.shape)

    return balance_data
  
# Function to split the dataset
def splitdataset(balance_data):
  
    # Separating the target variable
    Y = balance_data.values[:, 16]
    X = balance_data.values[:, 0:15]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
  
    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.2, random_state = 100)
      
    return X, Y, X_train, X_test, y_train, y_test

# Function to make predictions
def prediction(X_test, clf_object):
  
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    return y_pred
      
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
      
    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
      
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
      
    #print("Report : ",
    #classification_report(y_test, y_pred))
def test_transformers(columns,df):
    qt = QuantileTransformer(n_quantiles=500, output_distribution='normal')
    for i in columns:
        array = np.array(df[i]).reshape(-1, 1)
        y = qt.fit_transform(array)
        y=y*100
        df[i]=y
    print ("Dataset: ",df.head())
    return df
        
# Driver code
def main():
      
    # Building Phase
        data = importdata()
        X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
        columns= ["pay_schedule","home_owner","income","months_employed","years_employed","current_address_year","personal_account_m","personal_account_y","has_debt","amount_requested","risk_score","risk_score_2","risk_score_3","risk_score_4","risk_score_5","ext_quality_score","ext_quality_score_2","inquiries_last_month","e_signed"]
        #data=test_transformers(columns,data)
        #regression
        logr = linear_model.LogisticRegression(random_state = 0, penalty = 'l2')
        logr.fit(X_train,y_train)
        print("Results Using logistic regression Index:")
        y_pred = prediction(X_test, logr)
        cal_accuracy(y_test, y_pred)
        #random forest
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_pred = prediction(X_test, rf)
        cal_accuracy(y_test, y_pred)
        #XGBoost Classifier
        xgb_classifier = xgb.XGBClassifier(n_estimators=100, objective='binary:logistic', tree_method='hist', eta=0.1, max_depth=3)
        xgb_classifier.fit(X_train, y_train)
        y_pred = xgb_classifier.predict(X_test)
        cal_accuracy(y_test, y_pred)
        #svm Classifier
        Xsvm, Ysvm, X_trainsvm, X_testsvm, y_trainsvm, y_testsvm = splitdataset(data.sample(n=10000))
        clfsvm = svm.SVC(kernel='linear') # Linear Kernel
        clfsvm.fit(X_trainsvm, y_trainsvm)
        y_pred = clfsvm.predict(X_testsvm)
        cal_accuracy(y_testsvm, y_pred)
    
      
      
# Calling main function
if __name__=="__main__":
    main()
