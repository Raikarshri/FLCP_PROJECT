from sklearn.tree import DecisionTreeClassifier

def train_dt(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model