#This code is just to get the correlation of various columns with survival.
import pandas as pd
import numpy as np

#The following two functions are to replace the various string entries of the data with number proxies.
def cabin_as_integer(entry):
    group=ord(entry[0])
    return group
    
def reconfigure_data(X):
    X['Embarked'].replace(to_replace='C', value=0,inplace=True)
    X['Embarked'].replace(to_replace='Q',value=1,inplace=True)
    X['Embarked'].replace(to_replace='S',value =2,inplace=True)
    X['Sex'].replace(to_replace='male',value=0,inplace=True)
    X['Sex'].replace(to_replace='female',value=1,inplace=True)
    X['Cabin']=X['Cabin'].fillna('Z')
    X['Cabin']=X['Cabin'].apply(cabin_as_integer)
    X['Cabin'].replace(90,np.nan,inplace=True)

titanic_data=pd.read_csv(r'C:\Users\noahr\OneDrive\Documents\Kaggle\Intro to machine learning\Kaggle_Titanic\train.csv')
#the passenger ids and names should have no corollation with the survival, so we remove those.
relevant_cols= ['Pclass','Age','SibSp','Parch', 'Fare','Embarked','Sex','Cabin','Survived']
titnic_data=titanic_data[relevant_cols]
reconfigure_data(titanic_data)
#replace the columns with the columns minus the means in order to easily compute covariance.
for i in relevant_cols:
    mean=titanic_data[i].mean()
    titanic_data[i]=titanic_data[i]-mean
test_cols=['Pclass','Age','SibSp','Parch', 'Fare','Embarked','Sex','Cabin'] #all relevant columns except survived. Will test all of the corollations of these columns against survived.
survived_std=titanic_data['Survived'].std()
print(titanic_data[relevant_cols].describe())
for x in test_cols:
    test_table=titanic_data[[x,'Survived']].dropna()
    size=test_table[x].count()
    covariance = (test_table[x]*test_table['Survived']).sum()/(size-1)
    current_std=test_table[x].std()
    correlation=covariance/(survived_std*current_std)#This number itself isnt as important as the t-statistic, so we go on to compute this.
    t_statistic=correlation*np.sqrt((size-2)/(1-(correlation**2)))
    print(f"The t-statistic of {x} being correlated with Survived is: {t_statistic}")