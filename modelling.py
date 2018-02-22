#%%
from data_manipulation import get_X_y_datasets
from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(random_state=0)
from sklearn.metrics import confusion_matrix, classification_report
# print(confusion_matrix(y_test, y_pred))
# print(mean(y_pred))
# print(classification_report(y_test, y_pred))


X_train, y_train = get_X_y_datasets(time_reference, orig)
X_test, y_test = get_X_y_datasets(
    time_reference + pd.Timedelta(days=365), orig)
X_train



# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=5)]
# Number of features to consider at every split
max_features = ['auto']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 100, num=5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]  
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(rf, random_grid, verbose=True, n_jobs=4).fit(X_train, y_train)


X_train, y_train = get_X_y_datasets(time_reference, orig)
X_test, y_test = get_X_y_datasets(
    time_reference + pd.Timedelta(days=365), orig)

#%%
clf.best_params_
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=1500, stop=3000, num=5)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(15, 50, num=5)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(rf, random_grid, verbose=True,
                   n_jobs=3).fit(X_train, y_train)

# X_train
# #%%
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB(priors=[0.5, 0.5])
# y_pred = gnb.fit(X_train, y_train).predict(X_test)

# #%%
# from sklearn.metrics import confusion_matrix, classification_report
# print(confusion_matrix(y_test, y_pred))
# print(mean(y_pred))
# print(classification_report(y_test, y_pred))

# #%%
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(random_state=0)
# # clf.fit(X_train, y_train)
# # print(clf.feature_importances_)
# # X_train.columns


# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start=1000, stop=3000, num=5)]
# # Number of features to consider at every split
# max_features = ['auto']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 100, num=5)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True]
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# from sklearn.model_selection import GridSearchCV
# clf = GridSearchCV(rf, random_grid, verbose=True,
#                    n_jobs=4).fit(X_train, y_train)

# #%%
# import pprint
# rf.get_params()
# (clf.best_params_)
# #%%
# y_pred = clf.predict(X_test)
# print(confusion_matrix(y_test, y_pred))
# print(mean(y_pred))
# print(classification_report(y_test, y_pred))
# #%%

# gnb = GaussianNB(priors=[0.4, 0.6])
# for i in range(12):
#     time = min(orig.txn_time) + pd.Timedelta(days=(3 * 30 * i) + 365)

#     if i == 0:
#         X_train, y_train = get_X_y_datasets(time, orig)
#     else:
#         X_train, y_train = X_test, y_test
#     X_test, y_test = get_X_y_datasets(time + pd.Timedelta(days=30), orig)
#     y_pred = gnb.partial_fit(X_train, y_train, [0, 1]).predict(X_test)
#     print(mean(y_pred))
#     print(confusion_matrix(y_test, y_pred))
