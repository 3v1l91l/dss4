from lib import *
from spotlight.interactions import Interactions
from spotlight.factorization.explicit import ExplicitFactorizationModel
import torch

users, finder_decisions = load_data()
# zz = users.index.unique()[:20000]
# users.drop(users.index[~users.index.isin(zz)], inplace=True)
# finder_decisions.drop(finder_decisions.index[~finder_decisions['Sender_id'].isin(zz)], inplace=True)
# finder_decisions.drop(finder_decisions.index[~finder_decisions['Receiver_id'].isin(zz)], inplace=True)
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
model = ExplicitFactorizationModel(loss='logistic',
                                   embedding_dim=128,  # latent dimensionality
                                   n_iter=100,  # number of epochs of training
                                   batch_size=1024,  # minibatch size
                                   l2=1e-9,  # strength of L2 regularization
                                   learning_rate=1e-3,
                                   use_cuda=torch.cuda.is_available())
from spotlight.cross_validation import random_train_test_split

train, test = random_train_test_split(dataset, random_state=np.random.RandomState(42))

print('Split into \n {} and \n {}.'.format(train, test))
model.fit(train, verbose=True)
from spotlight.evaluation import rmse_score

train_rmse = rmse_score(model, train)
test_rmse = rmse_score(model, test)

print('Train RMSE {:.3f}, test RMSE {:.3f}'.format(train_rmse, test_rmse))
torch.save(model, 'spotlight.model')
