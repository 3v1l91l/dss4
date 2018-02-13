from lib import load_data
import pandas as pd

_, finder_decisions = load_data()

users_df = pd.read_pickle('users_df')
finder_decisions.merge(users_df['feature'].to_frame(), how='left', left_on='Receiver_id', left_index=True)
gr = finder_decisions.groupby('Sender_id').agg({'feature': lambda x: x.sum(axis=0) / len(x)})

users_df['preferences'] = None
# for id in users_df.index.values: