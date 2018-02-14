import keras


# inception_model = InceptionV3(include_top=False, weights='imagenet', input_tensor=None, input_shape=None,
#                               pooling='avg')
# for layer in inception_model.layers:
#     if hasattr(layer, 'trainable'):
#         layer.trainable = False
#
# n_latent_factors = 20
# embedding_layer = keras.layers.Embedding(n_users + 1, n_latent_factors, name='flattened_embedding')
# features_input = keras.layers.Input(shape=(4,), name='features')
#
# sender_input = keras.layers.Input(shape=[1], name='sender')
# sender_embedding = embedding_layer(sender_input)
# sender_vec = keras.layers.Flatten(name='flattened_sender')(sender_embedding)
# # sender_vec = keras.layers.Dropout(0.2)(sender_vec)
#
# receiver_input = keras.layers.Input(shape=[1], name='receiver')
# receiver_embedding = embedding_layer(receiver_input)
# receiver_vec = keras.layers.Flatten(name='flattened_receiver')(receiver_embedding)
# # receiver_vec = keras.layers.Dropout(0.2)(receiver_vec)
#
# # concat = keras.layers.concatenate([sender_vec, receiver_vec, features_input])
# concat = keras.layers.concatenate([sender_vec, receiver_vec])
#
# # concat_dropout = keras.layers.Dropout(0.2)(concat)
# dense = keras.layers.Dense(200, name='FullyConnected')(concat)
# # dense = keras.layers.Dense(200, name='FullyConnected')(features_input)
#
# # dropout_1 = keras.layers.Dropout(0.2, name='Dropout')(dense)
# dense_2 = keras.layers.Dense(100, name='FullyConnected-1')(dense)
# # dropout_2 = keras.layers.Dropout(0.2, name='Dropout')(dense_2)
# dense_3 = keras.layers.Dense(50, name='FullyConnected-2')(dense_2)
# # dropout_3 = keras.layers.Dropout(0.2, name='Dropout')(dense_3)
# dense_4 = keras.layers.Dense(20, name='FullyConnected-3', activation='relu')(dense_3)
#
# merged = keras.layers.merge([dense_4, inception_model.output], mode='concat')
#
# last = keras.layers.Dense(200, name='last')(merged)
#
# result = keras.layers.Dense(1, activation='sigmoid', name='Activation')(last)
# adam = Adam(lr=0.005)
# # model = keras.Model([receiver_input, sender_input, features_input], result)
# # model = keras.Model([receiver_input, sender_input], result)
# # model = keras.Model(features_input, result)
# model = keras.Model([receiver_input, sender_input, inception_model.input], result)
#
# model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])


def get_model():

    sender_like_input = keras.layers.Input(shape=(2048,), name='sender_like')
    # sender_skip_input = keras.layers.Input(shape=(2048,), name='sender_skip')

    # sender_vec = keras.layers.Dropout(0.2)(sender_vec)

    receiver_input = keras.layers.Input(shape=(2048,), name='receiver')
    # receiver_vec = keras.layers.Dropout(0.2)(receiver_vec)

    # concat = keras.layers.concatenate([sender_vec, receiver_vec, features_input])
    concat = keras.layers.concatenate([sender_like_input,
                                       # sender_skip_input,
                                       receiver_input])

    dense1 = keras.layers.Dense(1000, activation='relu')(concat)

    dense2 = keras.layers.Dense(500, activation='relu')(dense1)

    # concat_dropout = keras.layers.Dropout(0.2)(concat)
    dense = keras.layers.Dense(200, activation='relu')(dense2)
    # dense = keras.layers.Dense(200, name='FullyConnected')(features_input)

    # dropout_1 = keras.layers.Dropout(0.2, name='Dropout')(dense)
    dense_2 = keras.layers.Dense(100, activation='relu')(dense)
    # dropout_2 = keras.layers.Dropout(0.2, name='Dropout')(dense_2)
    dense_3 = keras.layers.Dense(50, activation='relu')(dense_2)
    # dropout_3 = keras.layers.Dropout(0.2, name='Dropout')(dense_3)
    dense_4 = keras.layers.Dense(20, activation='relu')(dense_3)



    result = keras.layers.Dense(1, activation='sigmoid', name='Activation')(dense_4)
    adam = keras.optimizers.Adam(lr=0.005)
    model = keras.Model([sender_like_input,
                         # sender_skip_input,
                         receiver_input], result)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_model_embedding(n_users):
    n_latent_factors = 50
    embedding_layer = keras.layers.Embedding(n_users + 1, n_latent_factors, name='flattened_embedding')

    sender_input = keras.layers.Input(shape=[1], name='sender')
    sender_embedding = embedding_layer(sender_input)
    sender_vec = keras.layers.Flatten(name='flattened_sender')(sender_embedding)
    # sender_vec = keras.layers.Dropout(0.2)(sender_vec)

    receiver_input = keras.layers.Input(shape=[1], name='receiver')
    receiver_embedding = embedding_layer(receiver_input)
    receiver_vec = keras.layers.Flatten(name='flattened_receiver')(receiver_embedding)
    # receiver_vec = keras.layers.Dropout(0.2)(receiver_vec)

    # concat = keras.layers.concatenate([sender_vec, receiver_vec, features_input])
    concat = keras.layers.concatenate([sender_vec, receiver_vec])


    # sender_vec = keras.layers.Dropout(0.2)(sender_vec)

    dense1 = keras.layers.Dense(1000, activation='relu')(concat)

    dense2 = keras.layers.Dense(500, activation='relu')(dense1)

    # concat_dropout = keras.layers.Dropout(0.2)(concat)
    dense = keras.layers.Dense(200, activation='relu')(dense2)
    # dense = keras.layers.Dense(200, name='FullyConnected')(features_input)

    # dropout_1 = keras.layers.Dropout(0.2, name='Dropout')(dense)
    dense_2 = keras.layers.Dense(100, activation='relu')(dense)
    # dropout_2 = keras.layers.Dropout(0.2, name='Dropout')(dense_2)
    dense_3 = keras.layers.Dense(50, activation='relu')(dense_2)
    # dropout_3 = keras.layers.Dropout(0.2, name='Dropout')(dense_3)
    dense_4 = keras.layers.Dense(20, activation='relu')(dense_3)



    result = keras.layers.Dense(1, activation='sigmoid', name='Activation')(dense_4)
    adam = keras.optimizers.Adam(lr=0.005)
    model = keras.Model([sender_input, receiver_input], result)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_model_embedding2(n_users):
    n_latent_factors = 150
    embedding_layer = keras.layers.Embedding(n_users + 1, n_latent_factors, name='flattened_embedding', input_length=1000)

    sender_input = keras.layers.Input(shape=(100,), name='sender')
    sender_embedding = embedding_layer(sender_input)
    sender_vec = keras.layers.Flatten(name='flattened_sender')(sender_embedding)
    # sender_vec = keras.layers.Dropout(0.2)(sender_vec)

    receiver_input = keras.layers.Input(shape=[1], name='receiver')
    receiver_embedding = embedding_layer(receiver_input)
    receiver_vec = keras.layers.Flatten(name='flattened_receiver')(receiver_embedding)
    # receiver_vec = keras.layers.Dropout(0.2)(receiver_vec)

    # concat = keras.layers.concatenate([sender_vec, receiver_vec, features_input])
    concat = keras.layers.concatenate([sender_vec, receiver_vec])


    # sender_vec = keras.layers.Dropout(0.2)(sender_vec)
    dense0 = keras.layers.Dense(2000, activation='relu')(concat)

    dense1 = keras.layers.Dense(1000, activation='relu')(dense0)

    dense2 = keras.layers.Dense(500, activation='relu')(dense1)

    # concat_dropout = keras.layers.Dropout(0.2)(concat)
    dense = keras.layers.Dense(200, activation='relu')(dense2)
    # dense = keras.layers.Dense(200, name='FullyConnected')(features_input)

    # dropout_1 = keras.layers.Dropout(0.2, name='Dropout')(dense)
    dense_2 = keras.layers.Dense(100, activation='relu')(dense)
    # dropout_2 = keras.layers.Dropout(0.2, name='Dropout')(dense_2)
    dense_3 = keras.layers.Dense(50, activation='relu')(dense_2)
    # dropout_3 = keras.layers.Dropout(0.2, name='Dropout')(dense_3)
    dense_4 = keras.layers.Dense(20, activation='relu')(dense_3)



    result = keras.layers.Dense(1, activation='sigmoid', name='Activation')(dense_4)
    adam = keras.optimizers.Adam(lr=0.005)
    model = keras.Model([sender_input, receiver_input], result)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model