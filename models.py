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

    sender_input = keras.layers.Input(shape=(2048,), name='sender')
    # sender_vec = keras.layers.Dropout(0.2)(sender_vec)

    receiver_input = keras.layers.Input(shape=(2048,), name='receiver')
    # receiver_vec = keras.layers.Dropout(0.2)(receiver_vec)

    # concat = keras.layers.concatenate([sender_vec, receiver_vec, features_input])
    concat = keras.layers.concatenate([sender_input, receiver_input])

    # concat_dropout = keras.layers.Dropout(0.2)(concat)
    dense = keras.layers.Dense(200, name='FullyConnected')(concat)
    # dense = keras.layers.Dense(200, name='FullyConnected')(features_input)

    # dropout_1 = keras.layers.Dropout(0.2, name='Dropout')(dense)
    dense_2 = keras.layers.Dense(100, name='FullyConnected-1')(dense)
    # dropout_2 = keras.layers.Dropout(0.2, name='Dropout')(dense_2)
    dense_3 = keras.layers.Dense(50, name='FullyConnected-2')(dense_2)
    # dropout_3 = keras.layers.Dropout(0.2, name='Dropout')(dense_3)
    dense_4 = keras.layers.Dense(20, name='FullyConnected-3', activation='relu')(dense_3)



    result = keras.layers.Dense(1, activation='sigmoid', name='Activation')(dense_4)
    adam = keras.optimizers.Adam(lr=0.005)
    model = keras.Model([receiver_input, sender_input], result)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])