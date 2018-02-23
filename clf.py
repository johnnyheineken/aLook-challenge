#%%
import pickle
from data_manipulation import *
import argparse
import sys
import pandas as pd
from sklearn.metrics import confusion_matrix

classifiers_all = pickle.load(open("classifiers_all.pickle", "rb"))
parser = argparse.ArgumentParser()

parser.add_argument('--transactions', help='Address to transactions.txt')
parser.add_argument('--products', help='(optional) Address to products.txt')
parser.add_argument('--users', help='(optional) Address to users.txt')
parser.add_argument('--refdate', help='(optional) Date for which should be prediction made. If left, the most current date in tnx dataset will be taken')
parser.add_argument('--output', help='(optional) path and name of the result file. If empty, ')
args = parser.parse_args()



transactions = pd.read_csv(
    args.transactions,
    sep='|',
    parse_dates=['txn_time']
)
ref_date = max(transactions.txn_time)
print(type(ref_date))

if args.refdate is None:
    ref_date = max(transactions.txn_time)
    print(type(ref_date))
else:
    try:
        ref_date = pd.Timestamp(args.refdate)
        print(ref_date)
    except ValueError:
        pass


try:
    products = pd.read_csv(args.products, sep='|')
    try:
        users = pd.read_csv(args.users, sep='|')
        # Using model containing all tables
        model_used = classifiers_all[0]
        X, _ = get_X_y_datasets(
        transactions, products=products, users=users, time_reference=ref_date)
    except ValueError:
        # Using model WITHOUT users
        model_used = classifiers_all[2]
        X, _ = get_X_y_datasets(
            transactions, products=products, time_reference=ref_date)
except ValueError:
    try:
        users = pd.read_csv(args.users, sep='|')
        #  Using model without products
        model_used = classifiers_all[1]
        X, _ = get_X_y_datasets(
            transactions, users=users, time_reference=ref_date)
    except ValueError:
        # Using model WITHOUT both products and users
        model_used = classifiers_all[3]
        X, _ = get_X_y_datasets(transactions, time_reference=ref_date)



y_pred = model_used.predict(X)
# print(confusion_matrix(y_train, y_pred))


result = pd.DataFrame(X.index.values, columns=['ID_user'])
result['churn_pred'] = y_pred.tolist()
if args.output is None:
    result.to_csv('result.csv')
else:
    result.to_csv(args.output)
