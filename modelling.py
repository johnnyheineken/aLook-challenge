#%%
from data_manipulation import *
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

'''
TODO:
WRAP THE MODELS
take all of the three models, pickle them and have them somewhere
make another script which takes models, 
creates dataset and predicts on such dataset


ANALYSE THE X  DATASET
Ill use R, probably to do that
however I need to have dataset which spans from min(date) to max(date) - year
with churn inside
'''
#%%


orig = pd.read_csv(
    'Data\\transactions.txt',
    sep='|',
    parse_dates=['txn_time']
)
users = pd.read_csv('Data\\users.txt', sep='|')
products = pd.read_csv('Data\\products.txt', sep='|')

n_quarters=8

time_reference = max(orig.txn_time) - pd.Timedelta(days=2 * 365)

#%%
X_train, y_train = get_X_y_datasets(orig, 
                    time_reference=time_reference,
                    users=users,
                    products=products,
                    # verbose=True, 
                    ndays_backward=(time_reference - min(orig.txn_time)).days, 
                    n_quarters=n_quarters)
X_test, y_test = get_X_y_datasets(
    transactions=orig,
    users=users,
    products=products,
    time_reference=time_reference + pd.Timedelta(days=365),
    # verbose=True, 
    n_quarters=n_quarters)


# Number of trees in random forest
rf = RandomForestClassifier(verbose=True, random_state=2110)

n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=4)]
# Number of features to consider at every split
max_features = ['auto']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 100, num=4)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]  
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 4]
# Method of selecting samples for training each tree
bootstrap = [True]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


clf = GridSearchCV(rf, random_grid, verbose=True, n_jobs=4).fit(X_train, y_train)

print(clf.best_params_)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))






# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=50, stop=3500, num=4)]
# Number of features to consider at every split
max_features = ['auto']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 200, num=4)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [3, 4, 5]
# Method of selecting samples for training each tree
bootstrap = [True]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

clf = GridSearchCV(rf, random_grid, 
                    scoring='f1',
                    verbose=True,
                    n_jobs=4).fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#%%

X_train, y_train = get_X_y_datasets(orig,
                                    time_reference=time_reference,
                                    users=users,
                                    products=products,
                                    # verbose=True,
                                    ndays_backward=(
                                        time_reference - min(orig.txn_time)).days,
                                    n_quarters=n_quarters)
X_test, y_test = get_X_y_datasets(
    transactions=orig,
    users=users,
    products=products,
    time_reference=time_reference + pd.Timedelta(days=365),
    # verbose=True,
    n_quarters=n_quarters)


clf_all = RandomForestClassifier(bootstrap= True,
        max_depth=70,
        max_features='auto',
        min_samples_leaf=4,
        min_samples_split=2,
        n_estimators=1200)
clf_all = clf_all.fit(X_train, y_train)
y_pred = clf_all.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



#%%

X_train, y_train = get_X_y_datasets(orig,
                                    time_reference=time_reference,
                                    users=users,
                                    # products=products,
                                    # verbose=True,
                                    ndays_backward=(
                                        time_reference - min(orig.txn_time)).days,
                                    n_quarters=n_quarters)
X_test, y_test = get_X_y_datasets(
    transactions=orig,
    users=users,
    # products=products,
    time_reference=time_reference + pd.Timedelta(days=365),
    # verbose=True,
    n_quarters=n_quarters)



clf_no_products = RandomForestClassifier(bootstrap=True,
                               max_depth=70,
                               max_features='auto',
                               min_samples_leaf=4,
                               min_samples_split=2,
                               n_estimators=1200) \
                    .fit(X_train, y_train)

y_pred = clf_no_products.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#%%
X_train, y_train = get_X_y_datasets(orig,
                                    time_reference=time_reference,
                                    # users=users,
                                    products=products,
                                    # verbose=True,
                                    ndays_backward=(
                                        time_reference - min(orig.txn_time)).days,
                                    n_quarters=n_quarters)
X_test, y_test = get_X_y_datasets(
    transactions=orig,
    # users=users,
    products=products,
    time_reference=time_reference + pd.Timedelta(days=365),
    # verbose=True,
    n_quarters=n_quarters)

rf = RandomForestClassifier(verbose=True, random_state=2110)

clf_no_users = RandomForestClassifier(bootstrap=True,
                                         max_depth=70,
                                         max_features='auto',
                                         min_samples_leaf=4,
                                         min_samples_split=2,
                                         n_estimators=1200) \
    .fit(X_train, y_train)

y_pred = clf_no_users.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#%%
X_train, y_train = get_X_y_datasets(orig,
                                    time_reference=time_reference,
                                    # users=users,
                                    # products=products,
                                    # verbose=True,
                                    ndays_backward=(
                                        time_reference - min(orig.txn_time)).days,
                                    n_quarters=n_quarters)
X_test, y_test = get_X_y_datasets(
    transactions=orig,
    # users=users,
    # products=products,
    time_reference=time_reference + pd.Timedelta(days=365),
    # verbose=True,
    n_quarters=n_quarters)

rf = RandomForestClassifier(verbose=True, random_state=2110)

clf_no_users_products = RandomForestClassifier(bootstrap=True,
                                      max_depth=70,
                                      max_features='auto',
                                      min_samples_leaf=4,
                                      min_samples_split=2,
                                      n_estimators=1200) \
    .fit(X_train, y_train)

y_pred = clf_no_users_products.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#%%

import pickle
classifiers_all = [clf_all, clf_no_products, clf_no_users, clf_no_users_products]
with open('model_GS_with_users.pickle', 'wb') as pickle_file:
    pickle.dump(classifiers_all, pickle_file)

