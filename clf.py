'''
This script serves as interface with data_manipulation and saved models. 
'''
import pickle
from data_manipulation import *
import argparse
import sys
import pandas as pd

classifiers_all = pickle.load(open("dual_models_all.pickle", "rb"))
parser = argparse.ArgumentParser()

parser.add_argument('--transactions', help='Address to transactions.txt')
parser.add_argument('--products', help='(optional) Address to products.txt')
parser.add_argument('--users', help='(optional) Address to users.txt')
parser.add_argument('--refdate', help='(optional) Date for which should be prediction made, passed to pd.Timestamp(). If left, the most current date in transactions dataset will be taken')
parser.add_argument('--saveX', help='(optional) if set to 1, script also saves "Xmatrix.csv", which was used for churn prediction')
# parser.add_argument('--outdir', help='(optional) path to save the resulting file. If the folder does not exist, it is created ')
# parser.add_argument('--filename', help='(optional) name of the resulting file. If not specified, file is saved as "result.csv"')
args = parser.parse_args()



transactions = pd.read_csv(
    args.transactions,
    sep='|',
    parse_dates=['txn_time']
)


if args.refdate is None:
    ref_date = max(transactions.txn_time)
    
else:
    try:
        ref_date = pd.Timestamp(args.refdate)
    except ValueError:
        warnings.warn('Invalid date format')
        pass


try:
    products = pd.read_csv(args.products, sep='|')
    try:
        users = pd.read_csv(args.users, sep='|')
        # Using model containing all tables
        model_used = classifiers_all[0]
        X, y = get_X_y_datasets(
                transactions, 
                products=products, 
                users=users, 
                time_reference=ref_date, 
                ndays_backward=(
                ref_date - min(transactions.txn_time)).days)
    except ValueError:
        # Using model WITHOUT users
        model_used = classifiers_all[2]
        X, y = get_X_y_datasets(
            transactions, products=products, time_reference=ref_date, ndays_backward=(
                ref_date - min(transactions.txn_time)).days)
except ValueError:
    try:
        users = pd.read_csv(args.users, sep='|')
        #  Using model without products
        model_used = classifiers_all[1]
        X, y = get_X_y_datasets(
            transactions, users=users, time_reference=ref_date, ndays_backward=(
                ref_date - min(transactions.txn_time)).days)
    except ValueError:
        # Using model WITHOUT both products and users
        model_used = classifiers_all[3]
        X, y = get_X_y_datasets(transactions, time_reference=ref_date, verbose=True, ndays_backward=(
            ref_date - min(transactions.txn_time)).days)

Xchurned = X[X.last_txn_days > 365]
X = X[X.last_txn_days <= 365]
Xr, _ = get_subset(X, y, rich = True)
Xp, _ = get_subset(X, y, rich = False)

y_churned = [1] * Xchurned.shape[0]
yr_pred = model_used[0].predict(Xr)
yp_pred = model_used[1].predict(Xp)
y_pred = y_churned + yr_pred.tolist() + yp_pred.tolist()
# print(yr_pred)
X = pd.concat([Xchurned, Xr, Xp])




result = pd.DataFrame(X.index.values, columns=['ID_user'])
result['churn_pred'] = y_pred

result.to_csv('prediction.csv')
try:
    if int(args.saveX) == 1:
        X.to_csv('Xmatrix.csv')
    else:
        ValueError('Wrong value of saveX (must be 1)')
except TypeError:
    pass
