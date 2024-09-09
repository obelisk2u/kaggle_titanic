import autogen
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class data_processing_agent:
    def run(self, data):
        #handle missing values
        data['Age'] = data['Age'].fillna(data['Age'].median())
        data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())

        #make categorical variables binary 
        data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
        data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)
        
        #handle family size
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
        return data

class modelling_agent:
    def run(self, X_train, y_train):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=5)
        model.fit(X_train, y_train)
        
        return model, scores.mean()

class evaluation_agent:
    def run(self, model, X_test):
        predictions = model.predict(X_test)
        return predictions

class submission_agent:
    def run(self, predictions, passenger_ids):
        submission = pd.DataFrame({
            'PassengerId': passenger_ids,
            'Survived': predictions
        })
        submission.to_csv('submission.csv', index=False)
        return submission

#load
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#start agents
data_agent = data_processing_agent()
model_agent = modelling_agent()
eval_agent = evaluation_agent()
submit_agent = submission_agent()

#process the data
train_data = data_agent.run(train_data)
test_data = data_agent.run(test_data)

#split the data into features and target
X_train = train_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y_train = train_data['Survived']
X_test = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

#train model
model, accuracy = model_agent.run(X_train, y_train)
print('Model accuracy:', accuracy)

#make predictions
predictions = eval_agent.run(model, X_test)

#make final csv
submission = submit_agent.run(predictions, test_data['PassengerId'])
print(submission.head())

