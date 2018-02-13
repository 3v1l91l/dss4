import pandas as pd
import numpy as np

def load_data():
    users = pd.read_csv('Users.csv', sep=';', dtype={'User_id': str})
    users.drop(users[users['User_id'].isnull()].index.values, inplace=True)
    users.drop(users[users['Photo'].isnull()].index.values, inplace=True)

    users['User_id'] = users['User_id'].astype(np.int64)
    users.set_index('User_id', inplace=True)

    finder_decisions = pd.read_csv('Finder_decisions_.csv', sep=';')#, nrows=10000)
    finder_decisions.drop(finder_decisions[(~finder_decisions['Receiver_id'].isin(users.index))].index.values, inplace=True)
    return users, finder_decisions
    # return users, 0