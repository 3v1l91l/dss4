import threading
from PIL import Image
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):  # Py3
        with self.lock:
            return next(self.it)

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def train_generator(finder_decisions, batch_size):
    while True:
        ix = np.random.randint(0, len(finder_decisions), batch_size)
        imgs = np.stack(imgs)
        yield ([finder_decisions['Sender_index'].values[ix], finder_decisions['Receiver_index'].values[ix], imgs],
               finder_decisions['Decision'].values[ix])

