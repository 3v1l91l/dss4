from lib import *
from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing import sequence

max_num_history_decisions = 500

users, finder_decisions = load_data()
zz = users.index.unique()[:20000]
users.drop(users.index[~users.index.isin(zz)], inplace=True)
finder_decisions.drop(finder_decisions.index[~finder_decisions['Sender_id'].isin(zz)], inplace=True)
finder_decisions.drop(finder_decisions.index[~finder_decisions['Receiver_id'].isin(zz)], inplace=True)
sender_value_counts = finder_decisions['Sender_id'].value_counts()
finder_decisions.drop(
    finder_decisions.index[finder_decisions['Sender_id'].isin(sender_value_counts.index[sender_value_counts == 1])],
    inplace=True)
# users.drop(users.index[~users.index.isin(np.union1d(finder_decisions['Sender_id'].values, finder_decisions['Receiver_id'].values))], inplace=True)
users['index'] = np.arange(len(users))
finder_decisions = finder_decisions.merge(users, how='left', left_on='Sender_id', right_index=True)
finder_decisions.rename(columns={'age': 'Sender_age', 'gender': 'Sender_gender', 'index': 'Sender_index'}, inplace=True)
finder_decisions = finder_decisions.merge(users, how='left', left_on='Receiver_id', right_index=True)
finder_decisions.rename(columns={'age': 'Receiver_age', 'gender': 'Receiver_gender', 'index': 'Receiver_index'},
                        inplace=True)
finder_decisions['Decision'] = finder_decisions['Decision'] == 'like'

users.set_index('index', inplace=True)
like_seq = finder_decisions[finder_decisions['Decision']].groupby('Sender_index').agg({'Receiver_index': lambda x: list(x)})
skip_seq = finder_decisions[~finder_decisions['Decision']].groupby('Sender_index').agg({'Receiver_index': lambda x: list(x)})

like_seq_pad = [list(x) for x in sequence.pad_sequences(list([x[0] for x in like_seq.values]),
                                                        maxlen=max_num_history_decisions)]
like_seq = pd.Series(index=like_seq.index, data=like_seq_pad)

skip_seq_pad = [list(x) for x in sequence.pad_sequences(list([x[0] for x in skip_seq.values]),
                                                        maxlen=max_num_history_decisions)]
skip_seq = pd.Series(index=skip_seq.index, data=skip_seq_pad)

users['like_seq'] = like_seq
users['skip_seq'] = skip_seq
finder_decisions = finder_decisions.merge(users[['like_seq', 'skip_seq']], how='left', left_on='Sender_index', right_index=True)
finder_decisions = finder_decisions[(~finder_decisions['like_seq'].isnull()) & (~finder_decisions['skip_seq'].isnull())]
train, test = train_test_split(finder_decisions, test_size=0.2)

n_seq_latent_factors = 500
n_rec_latent_factors = 200
# n_users = len(users)
n_users = len(zz)

like_seq_input = keras.layers.Input(shape=(max_num_history_decisions,))
like_seq_emb = keras.layers.Embedding(n_users + 1, n_seq_latent_factors, input_length=max_num_history_decisions)(like_seq_input)
like_seq_vec = keras.layers.Flatten()(like_seq_emb)

skip_seq_input = keras.layers.Input(shape=(max_num_history_decisions,))
skip_seq_emb = keras.layers.Embedding(n_users + 1, n_seq_latent_factors, input_length=max_num_history_decisions)(skip_seq_input)
skip_seq_vec = keras.layers.Flatten()(skip_seq_emb)

# sender_vec = keras.layers.Dropout(0.2)(sender_vec)
rec_input = keras.layers.Input(shape=(1,))
receiver_emb = keras.layers.Embedding(n_users + 1, n_rec_latent_factors, input_length=1)(rec_input)
receiver_vec = keras.layers.Flatten()(receiver_emb)
# receiver_vec = keras.layers.Dropout(0.2)(receiver_vec)

x = keras.layers.concatenate([like_seq_input,
                                   # skip_seq_input,
                                   rec_input])
x = keras.layers.Dense(8000, activation='relu')(x)
x = keras.layers.Dense(4000, activation='relu')(x)
x = keras.layers.Dense(1000, activation='relu')(x)
x = keras.layers.Dense(500, activation='relu')(x)
x = keras.layers.Dense(200, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(1, activation='sigmoid', name='Activation')(x)
adam = keras.optimizers.Adam(lr=0.005)
model = keras.Model([like_seq_input,
                     # skip_seq_input,
                     rec_input], x)

model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

model.fit([np.stack(train['like_seq']),
           # np.stack(train['skip_seq']),
           train['Receiver_index']],
          train['Decision'], epochs=100,
          batch_size=64)