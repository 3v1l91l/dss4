from lib import *
from surprise import SVD, Reader, Dataset
from surprise.model_selection import cross_validate

users, finder_decisions = load_data()
# finder_decisions = finder_decisions
reader = Reader(rating_scale=(0, 1))
finder_decisions['Decision'] = finder_decisions['Decision'] == 'like'
data = Dataset.load_from_df(finder_decisions[['Sender_id', 'Receiver_id', 'Decision']], reader)

cross_validate(SVD(), data, cv=2, verbose=True)

