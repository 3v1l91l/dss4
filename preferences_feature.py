from lib import load_data
import pandas as pd
import numpy as np

_, finder_decisions = load_data()

users_df = pd.read_pickle('users_df')
finder_decisions = finder_decisions.merge(users_df['feature'].to_frame(), how='left', left_on='Receiver_id', right_index=True)
like_preferences = finder_decisions[finder_decisions['Decision'] == 'like'].groupby('Sender_id')['feature'].apply(np.mean)
skip_preferences = finder_decisions[finder_decisions['Decision'] == 'skip'].groupby('Sender_id')['feature'].apply(np.mean)
users_df['like_preferences'] = like_preferences
users_df['skip_preferences'] = skip_preferences

users_df.to_pickle('users_df')
# for id in users_df.index.values: