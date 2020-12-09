#ypredproba = Reg_Log.predict_proba(X_test)[:,1]
def My_model ( X, y, size, RdomState = 42) :
    #X, y
    X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=size, 
                                                       random_state=RdomState )
    model = LogisticRegression(random_state= RdomState)
    model.fit(X_train, y_train)
    # Run the model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    score_train = model.score(X_train, y_train)
    score_test = model.score(X_test, y_test)
    metric = metrics.classification_report(y_test, ypred)
    
    return {"y_test": y_test, "prediction": y_pred, "proba":y_prob,
           "score_train": score_train, "score_test": score_test,
           "model": model, "metric": print(metric)}



def parse_model_final(X):
    if "Survived" not in X.columns:
        raise ValueError("target column survived should belong to df")
    target = X["Survived"]
    X['title'] = X['Name'].map(lambda x: x.split(',')[1].split('.')[0])
    X['surname'] = X['Name'].map(lambda x: '(' in x)
    X['Cabin'] = X['Cabin'].map(lambda x: x[0] if not pd.isnull(x) else -1)
    to_dummy = ['Pclass', 'Sex', 'title', 'Embarked', 'Cabin']
    for dum in to_dummy:
        split_temp = pd.get_dummies(X[dum], prefix=dum)
        X = X.join(split_temp)
        del X[dum]
    X['Age'] = X['Age'].fillna(X['Age'].median())
    X['is_child'] = X['Age'] <= 8
    to_del = ["PassengerId", "Name", "Survived", "Ticket"]
    for col in to_del:
        del X[col]
    return X, target  


def logloss(p, y, bound_limit=10e-8):
    p = max(min(p, 1 - bound_limit), bound_limit)
 
    return -np.log(p) if y == 1. else -np.log(1. - p) 


def accuracy(obs, pred):
    return np.mean(np.abs(obs - pred))

def rmse(obs, pred):
    return np.sqrt(np.mean((obs - pred) ** 2))