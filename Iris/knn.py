from sklearn.neighbors import KNeighborsClassifier      #the type of model used
from sklearn.metrics import accuracy_score              #testing how accurate this model is

#Using KNN
def knn(X_train, X_test, y_train, y_test, k):
    model = KNeighborsClassifier(n_neighbors = k)      #KNN model
    model.fit(X_train, y_train)         #fit the model using training data
    
    y_pred = model.predict(X_test)      #predict using trained model
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy