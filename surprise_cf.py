from lib import *
from surprise import SVD, Reader, Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import accuracy

users, finder_decisions = load_data()
# finder_decisions = finder_decisions
finder_decisions['Decision'] = finder_decisions['Decision'] == 'like'

train, test = train_test_split(finder_decisions, test_size=0.2)
reader = Reader(rating_scale=(0, 1))
train = Dataset.load_from_df(train[['Sender_id', 'Receiver_id', 'Decision']], reader)
test = Dataset.load_from_df(test[['Sender_id', 'Receiver_id', 'Decision']], reader)

svd = SVD(train)
# cross_validate(SVD(), data, cv=2, verbose=True)

predictions = svd.test(test)

# Then compute RMSE
print(accuracy.rmse(predictions))
print(accuracy.mae(predictions))