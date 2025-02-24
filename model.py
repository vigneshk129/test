import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('iris.csv')
print(df.head())
print('--------------------------------------------------------')
print(df.tail())
x = df[["Sepal_Length","Sepal_Width","Petal_Length", "Petal_Width"]]
y = df["Class"]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)

pickle.dump(classifier, open("model.pk1", "wb"))