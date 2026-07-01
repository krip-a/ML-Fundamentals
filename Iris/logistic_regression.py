from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score              #testing how accurate this model is

def logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression()      #KNN model
    model.fit(X_train, y_train)         #fit the model using training data
    
    y_pred = model.predict(X_test)      #predict using trained model
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy