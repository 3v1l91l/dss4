import keras
import numpy as np
from lib import load_data
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

np.random.seed(42)
BATCH_SIZE = 256
def train():
    users, finder_decisions = load_data()
    zz = users.index.unique()[:10000]
    users.drop(users.index[~users.index.isin(zz)], inplace=True)
    finder_decisions.drop(finder_decisions.index[~finder_decisions['Sender_id'].isin(zz)], inplace=True)
    finder_decisions.drop(finder_decisions.index[~finder_decisions['Receiver_id'].isin(zz)], inplace=True)
    users['index'] = np.arange(len(users))
    finder_decisions =  finder_decisions.merge(users, how='left', left_on='Sender_id', right_index=True)
    finder_decisions.rename(columns={'age': 'Sender_age', 'gender': 'Sender_gender', 'index': 'Sender_index'}, inplace=True)
    finder_decisions =  finder_decisions.merge(users, how='left', left_on='Receiver_id', right_index=True)
    finder_decisions.rename(columns={'age': 'Receiver_age', 'gender': 'Receiver_gender', 'index': 'Receiver_index'}, inplace=True)
    finder_decisions['Decision'] = finder_decisions['Decision'] == 'like'

    n_users = len(users)
    finder_decisions_train, finder_decisions_valid =  train_test_split(finder_decisions, stratify=finder_decisions['Decision'])
    n_latent_factors = 3
    sender_input = keras.layers.Input(shape=[1], name='Item')
    sender_embedding = keras.layers.Embedding(n_users + 1, n_latent_factors, name='Movie-Embedding')(sender_input)
    sender_vec = keras.layers.Flatten(name='FlattenMovies')(sender_embedding)
    sender_vec = keras.layers.Dropout(0.2)(sender_vec)

    receiver_input = keras.layers.Input(shape=[1], name='User')
    receiver_vec = keras.layers.Flatten(name='FlattenUsers')(
        keras.layers.Embedding(n_users + 1, n_latent_factors, name='User-Embedding')(receiver_input))
    receiver_vec = keras.layers.Dropout(0.2)(receiver_vec)

    concat = keras.layers.merge([sender_vec, receiver_vec], mode='concat', name='Concat')
    # concat_dropout = keras.layers.Dropout(0.2)(concat)
    dense = keras.layers.Dense(200, name='FullyConnected')(concat)
    # dropout_1 = keras.layers.Dropout(0.2, name='Dropout')(dense)
    dense_2 = keras.layers.Dense(100, name='FullyConnected-1')(concat)
    # dropout_2 = keras.layers.Dropout(0.2, name='Dropout')(dense_2)
    dense_3 = keras.layers.Dense(50, name='FullyConnected-2')(dense_2)
    # dropout_3 = keras.layers.Dropout(0.2, name='Dropout')(dense_3)
    dense_4 = keras.layers.Dense(20, name='FullyConnected-3', activation='relu')(dense_3)

    result = keras.layers.Dense(1, activation='relu', name='Activation')(dense_4)
    adam = Adam(lr=0.005)
    model = keras.Model([receiver_input, sender_input], result)
    model.compile(optimizer=adam, loss='mean_absolute_error', metrics=['accuracy'])

    history = model.fit([finder_decisions_train['Sender_index'].values, finder_decisions_train['Receiver_index'].values],
                        finder_decisions_train['Decision'].values, epochs=250, verbose=1, batch_size=BATCH_SIZE,
                        steps_per_epoch=len(finder_decisions_train),
                        validation_steps=len(finder_decisions_valid),
                        validation_data=([finder_decisions_valid['Sender_index'].values, finder_decisions_valid['Receiver_index'].values],
                                                        finder_decisions_valid['Decision'].values))

if __name__ == '__main__':
    train()