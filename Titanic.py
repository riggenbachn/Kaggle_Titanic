import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
    
    
def reconfigure_data(X):
    X['Embarked'].replace(to_replace='C', value=0,inplace=True)
    X['Embarked'].replace(to_replace='Q',value=1,inplace=True)
    X['Embarked'].replace(to_replace='S',value =2,inplace=True)
    X['Sex'].replace(to_replace='male',value=0,inplace=True)
    X['Sex'].replace(to_replace='female',value=1,inplace=True)
    

train_data=pd.read_csv(r'C:\Users\noahr\OneDrive\Documents\Kaggle\Intro to machine learning\Kaggle_Titanic\train.csv')
y=train_data["Survived"]
#These features were chosen after finding the correlations between the various columns and survived.
features = ['Pclass','Age','Parch', 'Fare','Embarked','Sex']
X=train_data[features]
reconfigure_data(X)
titanic_model=DecisionTreeRegressor(max_leaf_nodes=50)
titanic_model.fit(X,y)
test_data=pd.read_csv(r'C:\Users\noahr\OneDrive\Documents\Kaggle\Intro to machine learning\Kaggle_Titanic\test.csv')
test_X=test_data[features]
reconfigure_data(test_X)
prediction_y=titanic_model.predict(test_X)
predictions=[]
for i in range(len(prediction_y)):
    predictions.append(int(round(prediction_y[i])))
dictionary_of_outputs={'PassengerId':test_data['PassengerId'].to_numpy(), 'Survived':predictions}
output=pd.DataFrame(dictionary_of_outputs)
output=output.set_index('PassengerId')
output.to_csv('titanic_submission.csv')