import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("C:\\Users\\manik\\OneDrive\\Documents\\codsoft\\titanic\\tested.csv")

df.head()

df.isnull().sum()

df.Age = df['Age'].median()

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder #giving label for categorical columns

label = LabelEncoder()

df['Cabin'] = label.fit_transform(df['Cabin'])
df['Cabin'].unique()

df.Cabin = df['Cabin'].median() # filling missing values with the median of the cabin column

df['Fare'] = df['Fare'].fillna(df['Fare'].median())

df.isnull().sum()

df['Sex'] = label.fit_transform(df['Sex'])
df['Sex'].head()

df.head()

x = df.drop(['PassengerId','Survived','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns')
y = df['Survived']

x.head()

df['Sex'] = label.fit_transform(df['Sex'])
df['Sex'].head()

x.head()

y.head()

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

y_pred

from sklearn.metrics import accuracy_score,confusion_matrix

score = accuracy_score(y_test,y_pred)*100
print(" accuracy score : ",score)

cm = confusion_matrix(y_test,y_pred)

cm

#CONCLUSION : 

# i have created a decision tree model that the passenger on titanic is survived or not 
# the accuracy of the model is very accurate which scores 100
