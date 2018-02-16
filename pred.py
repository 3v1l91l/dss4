from lib import *
from spotlight.interactions import Interactions
import torch


users, finder_decisions = load_data()
zz = users.index.unique()[:10000]
users.drop(users.index[~users.index.isin(zz)], inplace=True)
finder_decisions.drop(finder_decisions.index[~finder_decisions['Sender_id'].isin(zz)], inplace=True)
finder_decisions.drop(finder_decisions.index[~finder_decisions['Receiver_id'].isin(zz)], inplace=True)
sender_value_counts = finder_decisions['Sender_id'].value_counts()
finder_decisions.drop(
    finder_decisions.index[finder_decisions['Sender_id'].isin(sender_value_counts.index[sender_value_counts == 1])],
    inplace=True)
users.drop(users.index[~users.index.isin(np.union1d(finder_decisions['Sender_id'].values, finder_decisions['Receiver_id'].values))], inplace=True)
users['index'] = np.arange(len(users))
finder_decisions = finder_decisions.merge(users, how='left', left_on='Sender_id', right_index=True)
finder_decisions.rename(columns={'age': 'Sender_age', 'gender': 'Sender_gender', 'index': 'Sender_index'}, inplace=True)
finder_decisions = finder_decisions.merge(users, how='left', left_on='Receiver_id', right_index=True)
finder_decisions.rename(columns={'age': 'Receiver_age', 'gender': 'Receiver_gender', 'index': 'Receiver_index'},
                        inplace=True)

ratings = np.ones(len(finder_decisions))
ratings[finder_decisions['Decision'] == 'skip'] = -1
ratings = ratings.astype(np.float32)
dataset = Interactions(finder_decisions['Sender_index'].values, finder_decisions['Receiver_index'].values, ratings)

from spotlight.cross_validation import random_train_test_split

train, test = random_train_test_split(dataset, random_state=np.random.RandomState(42))

spotlight_model = torch.load('spotlight.model')

predictions = spotlight_model.predict(test.user_ids, test.item_ids)
print((predictions == test.ratings).sum()/len(predictions))
