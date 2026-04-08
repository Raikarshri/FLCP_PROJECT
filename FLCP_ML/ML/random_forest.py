from sklearn.ensemble import RandomForestClassifier

def train_rf(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model