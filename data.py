import numpy as np
import tensorflow as tf

## the following are for one hot encode inputs, no good
# def message_generator(k):
#     '''message generator for validation dataset'''
#     rng = np.random.default_rng()
#     while True:
#         S = tf.one_hot(rng.integers(0, 2**k), 2**k)
#         # yield (input, target), autoencoder the target is the input
#         yield (S, S)

# def sparse_message_generator(k):
#     rng = np.random.default_rng()
#     while True:
#         S = rng.integers(0, 2**k)
#         # yield (input, target), autoencoder the target is the input
#         yield (np.array(S,ndmin=1), np.array(S,ndmin=1))

# def create_dataset(k):
#     ds = tf.data.Dataset.from_generator(message_generator, args=[k],
#                                        output_signature=(tf.TensorSpec(shape=(2**k,), dtype=tf.float32),
#                                                          tf.TensorSpec(shape=(2**k,), dtype=tf.float32)))

#     return ds

# def create_sparse_dataset(k):
#     ds = tf.data.Dataset.from_generator(sparse_message_generator, args=[k],
#                                        output_signature=(tf.TensorSpec(shape=(1), dtype=tf.float32),
#                                                          tf.TensorSpec(shape=(1), dtype=tf.float32)))

#     return ds

def data_gen(L, k):
    rng = np.random.default_rng()
    while True:
        B = rng.integers(0, 2, size=(k*L, 1)).astype(np.float32)
        yield (B, B)

def create_dataset(L, k):
    ds = tf.data.Dataset.from_generator(data_gen, args=[L, k],
                                        output_signature=(tf.TensorSpec(shape=(k*L, 1), dtype=tf.float32),
                                                         tf.TensorSpec(shape=(k*L, 1), dtype=tf.float32)))
    return ds

def create_mac_dataset(mac_k):
    single_user_ds = tuple(create_dataset(k) for k in mac_k)
    mac_ds = tf.data.Dataset.zip(single_user_ds)

    return single_user_ds, mac_ds