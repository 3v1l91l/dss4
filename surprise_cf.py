from lib import *
from surprise import SVD, Reader, Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import accuracy
data = Dataset.load_builtin('ml-100k')

# users, finder_decisions = load_data()
# # finder_decisions = finder_decisions[:100000]
# reader = Reader(rating_scale=(0,1))
# ids = finder_decisions['Sender_id'].value_counts()[:20].index.values
# finder_decisions.drop(finder_decisions[~finder_decisions['Sender_id'].isin(ids)].index, inplace=True)
# # finder_decisions = finder_decisions
# finder_decisions['Decision'] = finder_decisions['Decision'] == 'like'
# data = Dataset.load_from_df(finder_decisions[['Sender_id', 'Receiver_id', 'Decision']], reader)
# train, test = train_test_split(data, test_size=0.2, shuffle=True)
# # train = Dataset.load_from_df(train[['Sender_id', 'Receiver_id', 'Decision']], reader)
# # test = Dataset.load_from_df(test[['Sender_id', 'Receiver_id', 'Decision']], reader)
#
# svd = SVD()
# # cross_validate(SVD(), data, cv=2, verbose=True)
# svd.fit(train)
# predictions = svd.test(test)
# preds = np.array([p.est > 0.95 for p in predictions])
# preds_true = np.array([x[2] for x in test])
# print((preds == preds_true).sum()/len(preds))
# # for p in predictions:
# #     p.est = p.est > 0.8
# # # Then compute RMSE
# # accuracy.rmse(predictions)
# # accuracy.mae(predictions)



import numpy as np

from lightfm.datasets import fetch_stackexchange

data = fetch_stackexchange('crossvalidated',
                           test_set_fraction=0.1,
                           indicator_features=False,
                           tag_features=True)

train = data['train']
test = data['test']