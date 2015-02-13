"""
Papiya Sen

  Splitting the Iris dataset into Training and Test datasets

  Creating a decision tree model with Training data. Checking the accuracy of model with Test dataset.
"""


from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

# Divide data into training and test sets

from sklearn.cross_validation import train_test_split
iris.data_train, iris.data_test, iris.target_train, iris.target_test = train_test_split(
    iris.data, iris.target, test_size=0.33, random_state=42)
    
print len(iris.data_train)
print len(iris.data)

# Creating the decision tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data_train, iris.target_train)

print clf.predict(iris.data_test)
print iris.target_test

# Checking prediction accuracy with test data
score = clf.score(iris.data_test, iris.target_test)
print score
