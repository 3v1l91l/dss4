import keras
import numpy as np
from lib import load_data
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.callbacks import Callback
from keras.applications.inception_v3 import InceptionV3
from generator import *
from models import *

np.random.seed(42)
BATCH_SIZE = 32

from keras import backend as K
class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("Learning rate: {}".format(K.eval(lr_with_decay)))

def get_callbacks(model_name='model'):
    model_checkpoint = ModelCheckpoint(model_name + '.model', monitor='val_acc', save_best_only=True, save_weights_only=False,
                                       verbose=1)
    early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=1, verbose=1)
    lr_tracker = LearningRateTracker()
    return [model_checkpoint, early_stopping, reduce_lr, lr_tracker]

def train():
    users, finder_decisions = load_data()
    zz = users.index.unique()[:10000]
    users.drop(users.index[~users.index.isin(zz)], inplace=True)
    finder_decisions.drop(finder_decisions.index[~finder_decisions['Sender_id'].isin(zz)], inplace=True)
    finder_decisions.drop(finder_decisions.index[~finder_decisions['Receiver_id'].isin(zz)], inplace=True)
    sender_value_counts = finder_decisions['Sender_id'].value_counts()
    finder_decisions.drop(finder_decisions.index[finder_decisions['Sender_id'].isin( sender_value_counts.index[sender_value_counts==1])], inplace=True)
    users.drop(users.index[~users.index.isin(np.union1d(finder_decisions['Sender_id'].values, finder_decisions['Receiver_id'].values))], inplace=True)
    users['index'] = np.arange(len(users))
    finder_decisions = finder_decisions.merge(users, how='left', left_on='Sender_id', right_index=True)
    finder_decisions.rename(columns={'age': 'Sender_age', 'gender': 'Sender_gender', 'index': 'Sender_index'}, inplace=True)
    finder_decisions = finder_decisions.merge(users, how='left', left_on='Receiver_id', right_index=True)
    finder_decisions.rename(columns={'age': 'Receiver_age', 'gender': 'Receiver_gender', 'index': 'Receiver_index'}, inplace=True)
    finder_decisions['Decision'] = finder_decisions['Decision'] == 'like'

    n_users = len(users)
    # finder_decisions_train, finder_decisions_valid =  train_test_split(finder_decisions, stratify=finder_decisions['Decision'])
    finder_decisions_train, finder_decisions_valid = train_test_split(finder_decisions, stratify=finder_decisions['Sender_index'], test_size=0.2)

    model = get_model()
    model.summary()
    # model.fit([finder_decisions_train['Sender_index'].values, finder_decisions_train['Receiver_index'].values,
    #            finder_decisions_train[['Sender_age', 'Sender_gender', 'Receiver_age', 'Receiver_gender']].values],
    #                     finder_decisions_train['Decision'].values, epochs=20, verbose=1, batch_size=BATCH_SIZE,
    #                     validation_data=([finder_decisions_valid['Sender_index'].values, finder_decisions_valid['Receiver_index'].values,
    #                                       finder_decisions_valid[['Sender_age', 'Sender_gender', 'Receiver_age', 'Receiver_gender']].values],
    #                                                     finder_decisions_valid['Decision'].values),
    #                     callbacks=get_callbacks())

    # model.fit([finder_decisions_train['Sender_index'].values, finder_decisions_train['Receiver_index'].values],
    #                     finder_decisions_train['Decision'].values, epochs=20, verbose=1, batch_size=BATCH_SIZE,
    #                     validation_data=([finder_decisions_valid['Sender_index'].values, finder_decisions_valid['Receiver_index'].values],
    #                                                     finder_decisions_valid['Decision'].values),
    #                     callbacks=get_callbacks())
    train_generator_ = train_generator(finder_decisions_train, batch_size=BATCH_SIZE)
    valid_generator_ = train_generator(finder_decisions_valid, batch_size=BATCH_SIZE)

    model.fit_generator(train_generator_, epochs=20, verbose=1,
                        steps_per_epoch=len(finder_decisions_train) // BATCH_SIZE // 1000,
                        callbacks=get_callbacks(), validation_data=valid_generator_,
                        validation_steps=len(finder_decisions_valid) // BATCH_SIZE // 1000)
    # predictions = model.predict([finder_decisions_valid['Sender_index'].values, finder_decisions_valid['Receiver_index'].values],
    #                             batch_size=BATCH_SIZE)
    #
    # print(((np.squeeze(predictions) > 0.5) == finder_decisions_valid['Decision'].values).sum() / len(predictions))

if __name__ == '__main__':
    train()