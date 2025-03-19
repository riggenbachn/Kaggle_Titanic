import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y,r):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=r)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    predictions=[]
    for i in range(len(preds_val)):
        predictions.append(round(preds_val[i]))
    mae = mean_absolute_error(val_y, predictions)
    return(mae)
    
def cabin_as_integer(entry):
    group=ord(entry[0])
    return group
    
def reconfigure_data(X):
    X['Embarked'].replace(to_replace='C', value=0,inplace=True)
    X['Embarked'].replace(to_replace='Q',value=1,inplace=True)
    X['Embarked'].replace(to_replace='S',value =2,inplace=True)
    X['Sex'].replace(to_replace='male',value=0,inplace=True)
    X['Sex'].replace(to_replace='female',value=1,inplace=True)
    
my_imputer=SimpleImputer()
train_data=pd.read_csv(r'C:\Users\noahr\OneDrive\Documents\Kaggle\Intro to machine learning\Kaggle_Titanic\train.csv')
y=train_data["Survived"]
#These features were chosen after finding the correlations between the various columns and survived.
features = ['Pclass','Age','Parch', 'Fare','Embarked','Sex']
X=train_data[features]
reconfigure_data(X)
for r in range(10):
    train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=r)
    train_X_imputed = pd.DataFrame(my_imputer.fit_transform(train_X))
    val_X_imputed = pd.DataFrame(my_imputer.transform(val_X))
    train_X_imputed.columns=train_X.columns
    val_X_imputed.columns=val_X.columns
    print(f"For random seed {r}:")
    for max_leaf_nodes in [5, 50, 500, 5000]:
        my_mae = get_mae(max_leaf_nodes, train_X_imputed, val_X_imputed, train_y, val_y, r)
        print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %f" %(max_leaf_nodes, my_mae))
