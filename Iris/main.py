#imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score              #testing how accurate this model is

from knn import knn
from logistic_regression import logistic_regression
from decision_tree import decision_tree

#loading the dataset
iris = load_iris()

#X contains flower measurements (features)
#y contains the flower species (labels)
#predict the species using flower measurements
X = iris.data
y = iris.target

#splitting for test and train
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2,
                                                    random_state = 42)  #data is shuffled same everytime

knn_accuracy = knn(X_train, X_test, y_train, y_test, k=5)
print(f"KNN acuuracy: {knn_accuracy}")

logistic_accuracy = logistic_regression(X_train, X_test, y_train, y_test)
print(f"Logistic Tree acuuracy: {logistic_accuracy}")

tree_accuracy = decision_tree(X_train, X_test, y_train, y_test)
print(f"Decision Tree acuuracy: {tree_accuracy}")


