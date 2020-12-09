#Cast functions
def convert_cast(df, categorical_cols):
    for col_name in categorical_cols:
        df[col_name] = df[col_name].astype("object")
    return df

#modify Dataframe ^columns type
def convert_cast(mapper_config=None, dataframe=None):
    """
    Convert and cast DataFrame field types.

    This function achieves the same purpose as pd.read_csv, but it is used on
    actual DataFrame obtained from SQL.
    """
    # field conversion
    for var_name, lambda_string in mapper_config['converters'].items():
        if var_name in dataframe.columns:
            func = eval(lambda_string)
            dataframe[var_name] = dataframe[var_name].apply(func)

    # field type conversion/compression
    dty = dict(dataframe.dtypes)
    for col_name, actual_type in dty.items():
        expected_type = mapper_config['columns'][col_name]['dtype']
        if actual_type != expected_type:
            dataframe[col_name] = dataframe[col_name].astype(expected_type)

    return dataframe


# odef input_missing_values(df):
    for col in df.columns:
        if (df[col].dtype is float) or (df[col].dtype is int):
            df[col] = df[col].fillna(df[col].median())
        if (df[col].dtype == object):
            df[col] = df[col].fillna(df[col].mode()[0])

    return df



def parse_model(X, use_columns):
    if "Survived" not in X.columns:
        raise ValueError("target column survived should belong to df")
    target = X["Survived"]
    X = X[use_columns]
    return X, target


 # On utilise donc la moyenne de plusieurs validation croisées pour augmenter
# la significativité de la validation
def compute_score(clf, X, y, cv=5):
    """compute score in a classification modelisation.
       clf: classifier
       X: features
       y: target
    """
    xval = cross_val_score(clf, X, y, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (xval.mean(), xval.std() * 2))
    return xval


def plot_hist(feature, bins=20):
    x1 = np.array(dead[feature].dropna())
    x2 = np.array(survived[feature].dropna())
    plt.hist([x1, x2], label=["Victime", "Survivant"], bins=bins, color=['r', 'b'])
    plt.legend(loc="upper left")
    plt.title('distribution relative de %s' %feature)
    plt.show()
plot_hist('Pclass')




def transform_df(X, columns_to_dummify, features=["Pclass"], thres=10):
    X = convert_df_columns( X, features, type_var="object")
    X["is_child"] = X["Age"].apply(lambda x: 0 if x < thres else 1)
    X["title"] = X["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
    X['surname'] = X['Name'].map(lambda x: '(' in x)
    for col in columns_to_dummify:
        X_dummies = pd.get_dummies(X[col], prefix=col,
                                   drop_first=False, dummy_na=False, prefix_sep='_')
        X = X.join(X_dummies).drop(col, axis=1)
    return X.drop("Name", axis=1).drop("Age", axis=1)


def dummify_features(df):
    """
    Transform categorical variables to dummy variables.

    Parameters
    ----------
    df: dataframe containing only categorical features

    Returns
    -------
    X: new dataframe with dummified features
       Each column name becomes the previous one + the modality of the feature

    enc: the OneHotEncoder that produced X (it's used later in the processing chain)
    """
    colnames = df.columns
    le_dict = {}
    for col in colnames:
        le_dict[col] = preprocessing.LabelEncoder()
        le_dict[col].fit(df[col])
        df.loc[:, col] = le_dict[col].transform(df[col])

    enc = preprocessing.OneHotEncoder()
    enc.fit(df)
    X = enc.transform(df)

    dummy_colnames = [cv + '_' + str(modality) for cv in colnames for modality in le_dict[cv].classes_]
    # for cv in colnames:
    #     for modality in le_dict[cv].classes_:
    #         dummy_colnames.append(cv + '_' + modality)

    return X, dummy_colnames, enc


 def parse_model2(X):
    if "Survived" not in X.columns:
        raise ValueError("target column survived should belong to df")
    target = X["Survived"]
    to_dummy = ['Pclass', 'Sex']
    for dum in to_dummy:
        split_temp = pd.get_dummies(X[dum], prefix=dum)
        for col in split_temp:
            X[col] = split_temp[col]
        del X[dum]
    X['Age'] = X['Age'].fillna(X['Age'].median())
    to_del = ["PassengerId", "Name", "Cabin", "Embarked", "Survived", "Ticket"]
    for col in to_del:
        del X[col]
    return X, target

def clf_importances(X, clf):
    import pylab as pl
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    pl.title("Feature importances")
    for tree in clf.estimators_:
        pl.plot(xrange(X.shape[1]), tree.feature_importances_[indices], "r")
        pl.plot(xrange(X.shape[1]), importances[indices], "b")
        pl.show();
    for f in range(X.shape[1]):
        print("%d. feature: %s (%f)" %(f + 1, X.columns[indices[f]], importances[indices[f]]))


def space_parameters(model_conf):
    space ={}
    model_hyperparameters = model_conf["rf_hyperparameters"]
    for k in model_hyperparameters.keys():
        param_value = model_hyperparameters[k]
        if (k == "n_estimators") | (k == "min_samples_leaf"):
            space[k] = hp.choice(k, range(param_value[0], param_value[1], param_value[2]))
        elif k == "nthread":
            space[k] = param_value
        else:
            space[k] = hp.quniform(k, param_value[0], param_value[1], param_value[2])
    return space