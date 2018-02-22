#%%
import pandas as pd
import math
from numpy import mean
import warnings
import numpy as np
import argparse
import sys


'''
TODO:
add variables from users.txt
and products.txt

however, they are not needed for the analysis
'''


orig = pd.read_csv(
    'Data\\transactions.txt',
    sep='|',
    parse_dates=['txn_time']
)
users = pd.read_csv('Data\\users.txt', sep='|')


time_reference = max(orig.txn_time) - pd.Timedelta(days=364)


def get_X_y_datasets(time_reference, transactions, users=None, verbose=False, check_time=True):
    ''' Creates datasets usable in modelling with all features
    requires t
    for ids, which are available in last year from time_reference,
    i will find  date of the last transaction, 
    and look at all transactions in the year before 
    and make quarterly variables.
    if no transaction is in the year after the last txn, 
    i will append churn = 1, otherwise 0'''
    # it is tedious to write year over an over
    year = pd.Timedelta(days=365)

    if transactions.shape[1] != 5:
        sys.exit("Width of transaction matrix is not 5")
    if users is not None:
        if users.shape[1] != 3:
            sys.exit("Width of user matrix is not 3")
    if (time_reference > (max(transactions.txn_time)) - year) and check_time:
        warnings.warn('\n Time reference is not at least one year from the most current date in dataset, \
                        \n function returns X only. \
                        \n You can override this behaviour by setting check_time=False, \
                        \n but target variable might be corrupted')
    
    def process_users(users, X):
        users = users.fillna(0)
        users['female'] = users.gender == 'female'
        users['male'] = users.gender == 'male'
        users['no_age'] = users.age == 0
        users = users.drop(['gender'], axis=1)
        users = users.set_index('ID_user')
        X = X.merge(users, how='left', left_index=True, right_index=True)
        return X



    # unique IDs before time reference - I am looking at all 
    # user Ids with any transaction in the last year
    if verbose:
        print('dividing datasets')
    unq_IDs_b4_timeref = transactions[
            (transactions.txn_time < time_reference) & \
            (transactions.txn_time > (time_reference - year))
        ] \
        .ID_user \
        .unique()
    if verbose:
        print('number of active users in year before time reference: ' + \
            str(len(unq_IDs_b4_timeref)))

    # For feature engeneering, I decided to take only only data 
    # which are already available at the given moment. 
    # This will ensure that this function is applicable for any dataset
    # I am assuming, that only these informations are relevant (strong assumption)

    subset_before = transactions[
        transactions.ID_user.isin(unq_IDs_b4_timeref) & \
        (transactions.txn_time < time_reference)
        ]


    # Unit testing. I am checking for some errors
    def unit_test_sb(subset_before):
        if (max(subset_before.txn_time) > time_reference) or \
            len(subset_before.ID_user.unique()) != len(unq_IDs_b4_timeref):
            print('test failed')

    unit_test_sb(subset_before)


    # When was the last transaction for given user?
    # This will be the ending point for every user - from this time, I am 
    # calculating his churn, and obtaining his features
    users_temp = subset_before \
        .groupby('ID_user') \
        .txn_time.agg(max) \
        .reset_index() \
        .sort_values(by='txn_time', ascending=False)
    



    # I classify the quarters of each transaction in the dataset
    # This is the way I am preserving some temporal dimensions of the data
    # But in this way I will change it to classification task
    if verbose:
        print('Classifying quarters')
    def classify_quarter(date, last_date):
        months = last_date.to_period('M') - date.to_period('M')
        return(math.floor(months/3))

    subset_before = subset_before.merge(users_temp, how='left', on='ID_user')
    subset_before['quarter'] = [
        classify_quarter(
            subset_before.iloc[i, ].txn_time_x,
            subset_before.iloc[i, ].txn_time_y)
        for i in range(subset_before.shape[0])]

    ######################
    ### CHURN CREATION ###
    ######################
    # Now, creation of the churn
    #  I am looking at the original dataset
    # I look at the transactions, subset them for future year from the last transaction
    # and if that dataset is empty
    # I append one, otherwise I append zero
    if verbose:
        print('Creating churn')
    churn = []
    for i, j in users_temp.itertuples(index=False):
        churn.append(int(transactions[
            (transactions.ID_user == i) &
            (transactions.txn_time < j + year) &
            (transactions.txn_time > j)]
            .empty))
    users_temp['churn'] = churn
    users_temp['last_txn_days'] = [i.days for i in (time_reference - users_temp.txn_time)]



    if verbose:
        print('Creating features')
    # In the end, I am interested only in data in the last year.
    # Therefore, I take only data containing last four periods
    d = subset_before[subset_before.quarter <=3]
    # number of items bought, totally (in the last year)
    d = d \
        .groupby(d.columns.tolist()) \
        .size() \
        .reset_index() \
        .rename(columns={0: 'item_count'})
    # Cost of transaction obtained from given user (in the last year)
    d = d \
        .assign(overall_price=lambda x: x['price'] * x['item_count'])
    # number of transactions of given user (in the last year)
    d['txn_total'] = d.groupby('ID_user') \
        .ID_txn \
        .transform(len)
    # revenue obtained from given user
    d['revenue_total'] = d.groupby('ID_user') \
        .overall_price \
        .transform(sum)
    # of things bought by given user
    d['things_total'] = d.groupby('ID_user')['item_count'] \
        .transform(sum)

    # Shortcut
    Grouped = d.groupby('ID_user')
    
    # Helping functions, used in the next part
    def in_work(x):
        m = mean(x)
        return int((9 < m) & (m < 17))

    def in_evening(x): return int(17 <= mean(x))

    # When was the transaction made?
    d['hour_bought'] = [i.hour for i in d.txn_time_x]
    # Was that during usual working hours? (ignoring weekends, though)
    d['in_work'] = Grouped \
        .hour_bought \
        .transform(in_work)
    # Was that in evening?
    d['in_evening'] = Grouped \
        .hour_bought \
        .transform(in_evening)
    # This could provide us with some profile of the customers. Maybe those who buy 
    # things while at work are fired afterwards and have to churn ... :)

    # Proxy for socioeconomic status of the customer
    # Is he or she buying expensive things?
    d['avg_price_txn'] = d.revenue_total / d.txn_total
    d['avg_price_thing'] = d.revenue_total / d.things_total


    # Revenue from given customer per quarter
    revenue_per_q = d[['ID_user', 'overall_price', 'quarter']]
    revenue_per_q = revenue_per_q.groupby(['ID_user', 'quarter']).sum()
    revenue_per_q = revenue_per_q.unstack().fillna(0).reset_index()

    # How many transactions has the customer made in given quarter
    transactions_per_q = d[['ID_user', 'item_count', 'quarter']]
    transactions_per_q= transactions_per_q.groupby(['ID_user', 'quarter']).size()
    transactions_per_q = transactions_per_q.unstack().fillna(0).reset_index()
    
    # Suppressing warnings, as the merge issue several
    # as the tables are multiindex ones, however merging them
    # creates desired structure.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if verbose:
            print('merging temporary datasets')
        users_temp = users_temp.merge(revenue_per_q, how='left', on='ID_user')
        users_temp = users_temp.merge(transactions_per_q, how='left',
                    on='ID_user').drop('txn_time', axis=1)
        # But why some columns are not named afterwards,
        # remains mystery for me
        d = d \
            .merge(users_temp, how='left', on='ID_user') \
            .rename(
            columns={0: ('item_count', 0),
                    1: ('item_count', 1),
                    2: ('item_count', 2),
                    3: ('item_count', 3)}) \
            .drop(['txn_time_y', 'txn_time_x'], axis=1) 
    # dropping unneeded columns
    # these columns either have no use (IDs)
    # or they are transaction dependent (hour_bought)
    # or would cause multicollinearity (revenue_total)
    if verbose:
        print('dropping duplicates and unecessary columns')
    X = d.drop(
            ['quarter',
             'item_count',
             'ID_txn',
             'hour_bought',
             'ID_product',
             'price',
             'overall_price',
             'revenue_total'], axis=1) \
        .sort_values('ID_user') \
        .drop_duplicates() \
        .set_index('ID_user')
    
    if users is not None:
        if verbose:
            print('processing users')
        X = process_users(users, X)
    # One more variable - what is the trend? 
    # Is the customer buying more or less with respect to most recent quarter?
    X[('trend_revenue', 0)] = (
        X[('overall_price', 1)] - X[('overall_price', 0)]) > 0
    X[('trend_revenue', 1)] = (
        X[('overall_price', 2)] - X[('overall_price', 0)]) > 0
    X[('trend_revenue', 2)] = (
        X[('overall_price', 3)] - X[('overall_price', 0)]) > 0

    # Creating y and X.
    y = X.churn
    X = X.drop(['churn'], axis=1)
    if verbose:
        print('churn rate: ' + str(mean(y)))
        print('DONE')
    
    if (time_reference > (max(transactions.txn_time)) - year) and check_time:
        return X
    else:
        return X, y



