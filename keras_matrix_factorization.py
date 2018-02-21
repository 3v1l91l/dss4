import keras
from lib import *
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
from keras import backend as K
from keras.constraints import non_neg
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def get_callbacks(model_name='model'):
    model_checkpoint = ModelCheckpoint(model_name + '.model', monitor='val_acc', save_best_only=True, save_weights_only=False,
                                       verbose=1)
    early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1, epsilon=1e-3)
    return [model_checkpoint, early_stopping, reduce_lr]

_, finder_decisions = load_data()
users = pd.read_pickle('users_df')
ids = users[~users['vgg_face'].isnull()].index.values
finder_decisions.drop(finder_decisions[~finder_decisions['Receiver_id'].isin(ids)].index.values, inplace=True)
sender_value_counts = finder_decisions['Sender_id'].value_counts()
finder_decisions.drop(
    finder_decisions.index[finder_decisions['Sender_id'].isin(sender_value_counts.index[sender_value_counts <100])],
    inplace=True)
print(finder_decisions['Sender_id'].nunique())
# ids = finder_decisions['Sender_id'].value_counts()> 10
# finder_decisions.drop(finder_decisions[~finder_decisions['Receiver_id'].isin(ids)].index.values, inplace=True)

# finder_decisions['Decision'] = finder_decisions['Decision'] == 'like'
# finder_decisions, finder_decisions_valid = train_test_split(finder_decisions, test_size=0.2,
#                                                                   stratify=finder_decisions['Sender_id'],
#                                                                   random_state=42)
#
# ids = finder_decisions['Receiver_id'].value_counts() > 10
# popular_decisions = finder_decisions[~finder_decisions['Receiver_id'].isin(ids)]
# popular_decisions = popular_decisions.groupby('Receiver_id').agg({'Decision': 'mean'})
# popular_decisions_like = popular_decisions.index.values[np.squeeze((popular_decisions.values > 0.7))]
# popular_decisions_skip = popular_decisions.index.values[np.squeeze((popular_decisions.values <= 0.7))]
# # predictions = np.zeros(len(finder_decisions_valid))
# predictions = finder_decisions_valid['Receiver_id'].isin(popular_decisions_like)
# print((predictions.values == finder_decisions_valid['Decision'].values).sum()/len(predictions))

finder_decisions['Decision'] = finder_decisions['Decision'] == 'like'

# receiver_value_counts = finder_decisions['Receiver_id'].value_counts()
# receiver_ids = receiver_value_counts.index.values[receiver_value_counts > 500]
# finder_decisions.drop(finder_decisions.index[~finder_decisions['Receiver_id'].isin(receiver_ids)], inplace=True)
# sender_value_counts = finder_decisions['Sender_id'].value_counts()
# sender_ids = sender_value_counts.index.values[sender_value_counts > 500]
# finder_decisions.drop(finder_decisions.index[~finder_decisions['Sender_id'].isin(sender_ids)], inplace=True)
#
# users = users[np.isin(users.index.values,np.union1d(receiver_ids, sender_ids))]

# users.drop(users.index[~users.index.isin(zz)], inplace=True)
# finder_decisions.drop(finder_decisions.index[~finder_decisions['Sender_id'].isin(zz)], inplace=True)
# finder_decisions.drop(finder_decisions.index[~finder_decisions['Receiver_id'].isin(zz)], inplace=True)
# sender_value_counts = finder_decisions['Sender_id'].value_counts()
# finder_decisions.drop(
#     finder_decisions.index[finder_decisions['Sender_id'].isin(sender_value_counts.index[sender_value_counts == 1])],
#     inplace=True)
# users.drop(users.index[~users.index.isin(np.union1d(finder_decisions['Sender_id'].values, finder_decisions['Receiver_id'].values))], inplace=True)
users['index'] = np.arange(len(users))
finder_decisions = finder_decisions.merge(users, how='left', left_on='Sender_id', right_index=True)
finder_decisions.rename(columns={'age': 'Sender_age', 'gender': 'Sender_gender', 'index': 'Sender_index',
                                 'vgg_face': 'Sender_vgg_face'}, inplace=True)
finder_decisions = finder_decisions.merge(users, how='left', left_on='Receiver_id', right_index=True)
finder_decisions.rename(columns={'age': 'Receiver_age', 'gender': 'Receiver_gender', 'index': 'Receiver_index',
                                 'vgg_face': 'Receiver_vgg_face'},
                        inplace=True)

# age_encoder = OneHotEncoder()
# age_encoder.fit(users['age'].values.astype(np.int8).reshape(-1, 1))
# age = age_encoder.transform(finder_decisions['Sender_age'].values.astype(np.int8).reshape(-1, 1))

finder_decisions_train, finder_decisions_valid = train_test_split(finder_decisions, test_size=0.2,
                                                                  stratify=finder_decisions['Sender_index'],
                                                                  random_state=42)

# finder_decisions_valid['Prediction'] = 0
# cosine = 0
# skip = 0
# # for i in tqdm(range(len(finder_decisions_valid))):
# for i in tqdm(range(1000)):
#
#     sender_id = finder_decisions_valid.iloc[i]['Sender_id']
#     sender_finder_decisions = finder_decisions[finder_decisions['Sender_id'] == sender_id]
#     like_features = sender_finder_decisions[sender_finder_decisions['Decision']]['Receiver_vgg_face'].values
#     skip_features = sender_finder_decisions[~sender_finder_decisions['Decision']]['Receiver_vgg_face'].values
#     if (len(like_features) > 0) and (len(skip_features) > 0):
#         cosine = cosine +1
#         like_similarity = cosine_similarity(np.stack(like_features),
#                                             finder_decisions_valid.iloc[i]['Receiver_vgg_face'].reshape(1,-1)).mean()
#         skip_similarity = cosine_similarity(np.stack(skip_features),
#                                                 finder_decisions_valid.iloc[i]['Receiver_vgg_face'].reshape(1,-1)).mean()
#         finder_decisions_valid.iloc[i,finder_decisions_valid.columns.get_loc('Prediction')] = like_similarity > skip_similarity
#         print(finder_decisions_valid.iloc[i,finder_decisions_valid.columns.get_loc('Prediction')] == finder_decisions_valid.iloc[i,finder_decisions_valid.columns.get_loc('Decision')])
#     else:
#         skip = skip + 1
# print('Accuracy: {}'.format((finder_decisions_valid.iloc[:100,finder_decisions_valid.columns.get_loc('Prediction')] == finder_decisions_valid.iloc[:100,finder_decisions_valid.columns.get_loc('Decision')]).sum()/100))
# print(cosine/skip)

n_latent_factors=10
n_senders = len(users)
n_receivers = len(users)
sender_input = keras.layers.Input(shape=(1,))
sender_embedding = keras.layers.Embedding(input_dim=n_senders, output_dim=n_latent_factors, input_length=1, embeddings_constraint=non_neg())(sender_input)
sender_vec = keras.layers.Flatten()(sender_embedding)

meta_input = keras.layers.Input(shape=(512,), name='meta')
meta = keras.layers.Embedding(input_dim=n_receivers, output_dim=n_latent_factors, input_length=1, embeddings_constraint=non_neg())(meta_input)
meta = keras.layers.Flatten()(meta)

# meta_embedding = keras.layers.Embedding(input_dim=n_receivers, output_dim=n_latent_factors, input_length=1, embeddings_constraint=non_neg())(meta_input)
# meta_vec = keras.layers.Flatten()(meta_embedding)

receiver_input = keras.layers.Input(shape=(1,))
receiver_embedding = keras.layers.Embedding(input_dim=n_receivers, output_dim=n_latent_factors, input_length=1, embeddings_constraint=non_neg())(receiver_input)
receiver_vec = keras.layers.Flatten()(receiver_embedding)

x = keras.layers.Concatenate()([
    sender_vec,
    # receiver_vec,
    meta,
    # meta2_input
])


x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.Dense(1, activation='sigmoid')(x)
#
model = keras.Model([
    sender_input,
                     # receiver_input,
                     meta_input,
], x)
opt = keras.optimizers.Adam(lr=0.002)
model.compile(opt, loss='mean_squared_error', metrics=['accuracy'])
model.summary()
# model = load_model('model.model')

history = model.fit([
    finder_decisions_train['Sender_index'].values,
                     # finder_decisions_train['Receiver_index'].values,
                     np.stack(finder_decisions_train['Receiver_vgg_face'].values),
# np.stack(finder_decisions_train['Sender_vgg_face'].values)
],
                    finder_decisions_train['Decision'].values,
                    validation_data=([
                                         finder_decisions_valid['Sender_index'].values,
                                      # finder_decisions_valid['Receiver_index'].values,
                                      np.stack(finder_decisions_valid['Receiver_vgg_face'].values),
                                         # np.stack(finder_decisions_valid['Sender_vgg_face'].values)
                                     ],
                                     finder_decisions_valid['Decision'].values),
                    batch_size=512,
                    callbacks=get_callbacks(),
                    epochs=50, verbose=2)

# history = model.fit([finder_decisions_train[['Sender_index', 'Sender_age', 'Sender_gender']].values,
#                      finder_decisions_train[['Receiver_index', 'Receiver_age', 'Receiver_gender']].values],
#                     finder_decisions_train['Decision'].values,
#                     validation_data=([finder_decisions_valid[['Sender_index', 'Sender_age', 'Sender_gender']].values,
#                                       finder_decisions_valid[['Receiver_index', 'Receiver_age', 'Receiver_gender']].values],
#                                      finder_decisions_valid['Decision'].values),
#                     batch_size=2000,
#                     callbacks=get_callbacks(),
#                     epochs=50, verbose=2)

# model = load_model('model.model')
# predictions = model.predict([finder_decisions_valid['Sender_index'].values, finder_decisions_valid['Receiver_index'].values])
# print('Model accuracy: {}'.format((np.squeeze(predictions > 0.5) == finder_decisions_valid['Decision'].values).sum()/len(predictions)))
# acc = (np.array([True]*(len(finder_decisions_valid))) == finder_decisions_valid['Decision'].values).sum()/len(predictions)
# print('All like accuracy: {}'.format(acc))
# acc = (np.array([False]*(len(finder_decisions_valid))) == finder_decisions_valid['Decision'].values).sum()/len(predictions)
# print('All skip accuracy: {}'.format(acc))

# model = load_model('model.model')
# intermediate_layer_model = keras.Model(inputs=model.input,
#                                  outputs=model.get_layer('sender_embedding').output)
# intermediate_output = intermediate_layer_model.predict([finder_decisions_valid['Sender_index'].values, finder_decisions_valid['Receiver_index'].values])
# senders_per_group = len(intermediate_output)/len(np.unique(intermediate_output))
# print('Sender embedding groups: {}'.format(len(np.unique(intermediate_output))))
# print('Senders per group: {}'.format(senders_per_group))
#
# intermediate_layer_model = keras.Model(inputs=model.input,
#                                  outputs=model.get_layer('sender_embedding').output)
# intermediate_output = intermediate_layer_model.predict([finder_decisions_valid['Receiver_index'].values, finder_decisions_valid['Receiver_index'].values])
# receivers_per_group = len(intermediate_output)/len(np.unique(intermediate_output))
# print('Receiver embedding groups: {}'.format(len(np.unique(intermediate_output))))
# print('Receivers per group: {}'.format(receivers_per_group))
