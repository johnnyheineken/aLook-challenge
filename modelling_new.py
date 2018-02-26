#%%
from data_manipulation import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report


orig = pd.read_csv(
    'Data\\transactions.txt',
    sep='|',
    parse_dates=['txn_time']
)
users = pd.read_csv('Data\\users.txt', sep='|')
products = pd.read_csv('Data\\products.txt', sep='|')

n_quarters = 4

time_reference = max(orig.txn_time) - pd.Timedelta(days=365)

#%%


def get_subset(X, y, percentile=70, rich=True):
    X2 = X.copy()
    X2['churn'] = y
    # X2['churn']
    X2['total'] = X2.txn_total * X2.avg_price_txn
    perc = np.percentile(X2.total, percentile)
    if rich:
        X_subset = X2[(X2.total > perc)]
    else:
        X_subset = X2[(X2.total <= perc)]
    # X_subset
    y_subset = X_subset.churn
    X_subset = X_subset.drop(['churn', 'total'], axis=1)
    return X_subset, y_subset


def get_subset_model(X, y, percentile=70, rich=True):
    Xs, ys = get_subset(X, y, percentile, rich=True)
    if rich:
        rf = RandomForestClassifier(verbose=False,
                                    class_weight={0: 1, 1: 2},
                                    bootstrap=True,
                                    max_depth=20,
                                    max_features='auto',
                                    min_samples_leaf=5,
                                    min_samples_split=4,
                                    n_estimators=100
                                    ) \
            .fit(Xs, ys)
    else:
        rf = RandomForestClassifier(verbose=False,
                                    class_weight={0: 2, 1: 1},
                                    bootstrap=True,
                                    max_depth=5,
                                    max_features='auto',
                                    min_samples_leaf=4,
                                    min_samples_split=5,
                                    n_estimators=150
                                    ) \
            .fit(Xs, ys)
    return rf


def get_model_pair(X, y, percentile=70):
    clf_r = get_subset_model(X, y, percentile=percentile, rich=True)
    clf_p = get_subset_model(X, y, percentile=percentile, rich=False)
    return (clf_r, clf_p)

#%%
X_train, y_train = get_X_y_datasets(orig,
                                    time_reference=time_reference,
                                    users=users,
                                    products=products,
                                    verbose=True,
                                    # ndays_backward=(
                                    # time_reference - min(orig.txn_time)).days,
                                    n_quarters=n_quarters)
full_models = get_model_pair(X_train, y_train)
X_train, y_train = get_X_y_datasets(orig,
                                    time_reference=time_reference,
                                    users=users,
                                    # products=products,
                                    verbose=True,
                                    # ndays_backward=(
                                    # time_reference - min(orig.txn_time)).days,
                                    n_quarters=n_quarters)
model_noprod = get_model_pair(X_train, y_train)
X_train, y_train = get_X_y_datasets(orig,
                                    time_reference=time_reference,
                                    # users=users,
                                    products=products,
                                    verbose=True,
                                    # ndays_backward=(
                                    # time_reference - min(orig.txn_time)).days,
                                    n_quarters=n_quarters)
model_nousers = get_model_pair(X_train, y_train)
X_train, y_train = get_X_y_datasets(orig,
                                    time_reference=time_reference,
                                    # users=users,
                                    # products=products,
                                    verbose=True,
                                    # ndays_backward=(
                                    # time_reference - min(orig.txn_time)).days,
                                    n_quarters=n_quarters)
model_noprodusers = get_model_pair(X_train, y_train)
all_models = [full_models, model_noprod, model_nousers, model_noprodusers]
with open('dual_models_all.pickle', 'wb') as pickle_file:
    pickle.dump(all_models, pickle_file)

#%%
X_train, y_train = get_X_y_datasets(orig,
                                    time_reference=time_reference -
                                    pd.Timedelta(days=365),
                                    users=users,
                                    products=products,
                                    verbose=True,
                                    # ndays_backward=(
                                    # time_reference - min(orig.txn_time)).days,
                                    n_quarters=n_quarters)


X_test, y_test = get_X_y_datasets(
    transactions=orig,
    users=users,
    products=products,
    time_reference=time_reference,
    verbose=True,
    n_quarters=n_quarters)


#%%
for i in np.linspace(start=70, stop=90, num=7):
    percentile = int(i)
    print("______________________________________________")
    print("____________________")
    print("        " + str(percentile))
    print("____________________")
    Xsr_train, ysr_train = get_subset(
        X_train, y_train, percentile=percentile, rich=False)
    Xsr_test, ysr_test = get_subset(
        X_test, y_test, percentile=percentile, rich=False)

    rf = RandomForestClassifier(verbose=False,
                                class_weight={0: 2, 1: 1})

    n_estimators = [150]
    # Number of features to consider at every split
    max_features = ['auto']
    # Maximum number of levels in tree
    max_depth = [5]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [5]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [4]
    # Method of selecting samples for training each tree
    bootstrap = [True]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    clf = GridSearchCV(rf, random_grid, verbose=False,
                       n_jobs=4, scoring='f1').fit(Xsr_train, ysr_train)

    # print(clf.best_params_)
    yr_pred = clf.predict(Xsr_test)
    print(confusion_matrix(ysr_test, yr_pred))
    print(classification_report(ysr_test, yr_pred))
    print("____________________")

    Xsp_train, ysp_train = get_subset(
        X_train, y_train, percentile=percentile, rich=True)
    Xsp_test, ysp_test = get_subset(
        X_test, y_test, percentile=percentile, rich=True)

    rf = RandomForestClassifier(verbose=False,
                                class_weight={0: 1, 1: 2})

    n_estimators = [80]
    # Number of features to consider at every split
    max_features = ['auto']
    # Maximum number of levels in tree
    max_depth = [20]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [4]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [5]
    # Method of selecting samples for training each tree
    bootstrap = [True]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    clf = GridSearchCV(rf, random_grid, verbose=False,
                       n_jobs=4, scoring='f1').fit(Xsp_train, ysp_train)

    # print(clf.best_params_)
    yp_pred = clf.predict(Xsp_test)
    print(confusion_matrix(ysp_test, yp_pred))
    print(classification_report(ysp_test, yp_pred))
    print("____________________")

    y_pred = yp_pred.tolist() + yr_pred.tolist()
    ys_test = ysp_test.tolist() + ysr_test.tolist()
    print(confusion_matrix(ys_test, y_pred))
    print(classification_report(ys_test, y_pred))
    print("____________________")

