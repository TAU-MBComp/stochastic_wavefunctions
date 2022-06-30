import itertools
from sympy.combinatorics import Permutation
import tensorflow as tf
import numpy as np
from tensorflow import keras

space_d = 2
spin_up = 2
spin_down = 0
p = spin_up + spin_down
d = space_d * p


perm_up = list(itertools.permutations(range(spin_up)))
pairity_up = []
for i in perm_up:
    pairity_up.append([Permutation(i).parity()])
pairity_up = tf.constant((-1.) ** np.array(pairity_up))
perm_up = tf.constant(perm_up, dtype=tf.int32)

perm_down = list(itertools.permutations(range(spin_down)))
pairity_down = []
for i in perm_down:
    pairity_down.append([Permutation(i).parity()])
pairity_down = tf.constant((-1.) ** np.array(pairity_down))
perm_down = tf.constant(perm_down, dtype=tf.int32)

blank_perm = np.array([range(d // p)])
blank_perm = tf.cast(tf.tile(blank_perm, [1, p]), tf.int32)
pairity = tf.tile(pairity_up, [perm_down.shape[0], 1]) * tf.repeat(pairity_down, perm_up.shape[0], axis=0)
perm = tf.concat([tf.tile(perm_up, [perm_down.shape[0], 1]), tf.repeat(perm_down, perm_up.shape[0], axis=0) + spin_up], axis=1)

@tf.function(jit_compile=True, experimental_relax_shapes=True)
def permute(x):
    sample_perms = tf.random.uniform([tf.shape(x)[0], ], maxval=tf.shape(perm)[0], dtype=tf.int32)
    perms = tf.gather(perm, sample_perms)
    axis2 = d // p * tf.repeat(perms, d // p, axis=1)
    x_perm = tf.gather(x, blank_perm + axis2, axis=1, batch_dims=1)
    y_sign = tf.gather(pairity, sample_perms)
    return x_perm, tf.cast(y_sign, tf.float32)


class symmetry_sampler(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        init_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size, :]
        init_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size, :]
        batch_x, sign_y = permute(init_x)
        batch_y = init_y * sign_y
        return np.array(batch_x), np.array(batch_y)



val_percent = 0.05
cutoff = int((1.-val_percent)*n_samples)
x_fit = x[:cutoff, :]
y_fit = y[:cutoff, :]
x_val = x[cutoff:, :]
y_val = y[cutoff:, :]
batch_size = 1000
epochs = 1000
dataset = tf.data.Dataset.from_generator(symmetry_sampler,
                                         output_signature=(
                                             tf.TensorSpec(shape=(None, d), dtype=tf.float32),
                                             tf.TensorSpec(shape=(None, 1), dtype=tf.float32)), args=(x_fit, y_fit, batch_size))

history_callback = model.fit(dataset, epochs=epochs, batch_size=batch_size, shuffle=True,
                             validation_data=tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size))