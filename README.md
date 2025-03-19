# Kaggle_Titanic
* [Project description](https://github.com/riggenbachn/Kaggle_Titanic/blob/main/README.md#project-description)
* [Statistical analysis](https://github.com/riggenbachn/Kaggle_Titanic/blob/main/README.md#statistical-analysis)
* [Testing for optimal learning parameters](https://github.com/riggenbachn/Kaggle_Titanic/blob/main/README.md#testing-for-optimal-learning-parameters)
* [Results](https://github.com/riggenbachn/Kaggle_Titanic/blob/main/README.md#results)
* [Future improvements](https://github.com/riggenbachn/Kaggle_Titanic/blob/main/README.md#future-improvements)

## Project description
This repository is my solution to Kaggle's Titanic machine learning competition. Mostly this was done to practice the basic methods of machine learning and statistics I have learned.

There are 6 files, 3 csv files and 3 python files. The csv files train.csv and test.csv are the data files given to us to train and test our model, respectively. The file titanic_submission.csv is the output of our machine learning model. The file titanic_statistics.py is the python code used to run the statistical test described in [Statistical analysis](https://github.com/riggenbachn/Kaggle_Titanic/blob/main/README.md#statistical-analysis). Similarly, the file titanic_model_tester.py is the code used to test which learning parameters where best for this question as described in [Testing for optimal learning parameters](https://github.com/riggenbachn/Kaggle_Titanic/blob/main/README.md#testing-for-optimal-learning-parameters). Finally, the file Titanic.py is the python code used to generate the titanic_submission.csv file and is described in [Results](https://github.com/riggenbachn/Kaggle_Titanic/blob/main/README.md#results).

The machine learning tool we used was scikit-learn, specifically the DecisionTreeRegressor inside of sklearn.tree.

## Statistical analysis

The columns of train.csv are, in order,

* PassengerId
* Survived
* Pclass
* Name
* Sex
* Age
* SibSp
* Parch
* Ticket
* Fare
* Cabin
* Embarked

We can assume that the name and PassengerId are independent of Survived, which is the column we are trying to predict. We will therefore exclude these columns in our analysis and machine learning models. Further we will exclude the Ticket column since all the relevant information should also be contained in the Fare, Embarked, and Cabin columns. 

Of the remaining columns, Embarked, Sex, and Cabin are the only columns which are not integer values. For the Embarked column we replaces 'C' with 0, 'Q' with 1, and 'S' with 2. Similarly for Sex we replaced 'male' with 0 and 'female' with 1. Finally, for Cabin, we made the assumption that which Cabin group the passenger was in was signifigantly more important for survival than the specific room number, and so we only encoded the Cabin as the unicode for the first character of their cabin. 

As a first approach we split the train file into two peices using the test-train-split module of scikit-learn with the remaining features ran the decision tree regression model on half the data and tested our model on the other half. Setting the random state between 0 and 9 (inclusive) gave an absolute mean error of between 0.21 and 0.3 (see screenshot below of the exact absolute mean errors of the first 10 random states.).

![image](https://github.com/user-attachments/assets/e76d426c-5b51-4706-95ec-cc168f65e114)

While this does not seem too bad for a first approximation, there is also room for improvement. One way to improve this, and the direction we decided to persue, is to remove the variables which were not corrolated with the Survived column. We did this by first computing the sample covariance of each column with the Survived column, and then computing the sample correlation. From this we then computed a t-score using the formula $t=\mathrm{Corr}(X,\mathrm{Survived})\sqrt{\frac{n-2}{1-\mathrm{Corr}(X,\mathrm{Survived})}}$ the results of which are recorded below.
![image](https://github.com/user-attachments/assets/3868c0bb-11ab-4367-aefe-9ac860110445)

with a p-value of 0.05 we see that all the columns except SibSp and Cabin are corrolated with Survived and should be included. When we remove SibSp and Cabin we see that the mean absolute error now ranges from .17 to .27 (see the screanshot below of the exact mean absolute error of the first 10 random states.).
![image](https://github.com/user-attachments/assets/3735ca62-5f72-4622-ad13-37da49f3dd97)



## Testing for optimal learning parameters

Now that we have specified our features we will be building our model on, we will want to specify the number of leaf nodes our model can have so that we do not overtrain or undertrain our model. In order to do this we took the first 10 random states, and for each one we split the training data in half using this random seed and for each of $\{5,50,500,5000\}$ we initialized a decision tree regressor using this random seed with the indicated maximum number of  leaves. We then calculated the mean absolute error which resulted in the following:
![image](https://github.com/user-attachments/assets/c6bc7fa8-f851-4348-ac35-6e4c9005af48)
![image](https://github.com/user-attachments/assets/62e932ad-ef8e-4a54-8110-1a5bb4c648e9)


From this we see that the maximum number of leaves we should have for this model is 50. 
## Results

Using 50 as the maximum number of nodes, a feature list of ['Pclass','Age','Parch', 'Fare','Embarked','Sex'], and no random seed specified we generated from test.csv the file titanic_submission.csv. After uploading this to Kaggle we found that we had a 76.5% accuracy rate:
![image](https://github.com/user-attachments/assets/72a1da8d-8c87-4ae5-a411-3fd893af647a)



## Future improvements
The first place where we could improve our results is by using a more sofisticated machine learning tool. The statistical test we used was only for detecting linear relationships, so a model which incorporated more complex relationships between the columns would likely allow us to use the information in Cabin and SibSp to improve our accuracy. Finally it is possible that the Cabin information was encoded poorly and that we could improve on this by encoding the whole cabin and not just the grouping.
