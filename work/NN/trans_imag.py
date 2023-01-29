import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics, regularizers
import tensorflow_addons as tfa
import numpy as np
from sklearn.model_selection import train_test_split
from time import *

filename1 = r'..\data\H\pa_h'
filename2 = r'..\data\H\imag_h'

Network_output = 169
Network_intput = 4


def get_data_imag():
    r = np.genfromtxt(filename1 + '.csv', delimiter=',')[:, :-1]
    spectrum = np.genfromtxt(filename2 + '.csv', delimiter=',')[:, 21:]

    X_mean = r.max(axis=0)
    X_std = r.min(axis=0)
    X = r
    y = spectrum[:, :]

    X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.2, random_state=233)
    X_val, X_test, y_val, y_test, = train_test_split(X_other, y_other, test_size=0.5, random_state=233)
    return X_train, y_train, X_val, y_val, X_test, y_test, X_mean, X_std


def type_conversion(x, y):
    x = tf.cast(x, dtype=tf.float32)  # 将x，y的数据格式转化成dtype
    y = tf.cast(y, dtype=tf.float32)
    return x, y


def preprocess():
    X_train, y_train, X_val, y_val, X_test, y_test, X_mean, X_std = get_data_imag()

    batchsz = 512

    train_db = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_db = train_db.map(type_conversion, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(60000).batch(
        batchsz)
    train_db = train_db.prefetch(tf.data.experimental.AUTOTUNE)

    val_db = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_db = val_db.map(type_conversion, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batchsz)
    val_db = val_db.prefetch(tf.data.experimental.AUTOTUNE)

    return train_db, val_db


def train():
    train_db, val_db = preprocess()

    inputs = tf.keras.layers.Input(shape=(Network_intput))
    out = tf.keras.layers.Dense(150, activation='tanh')(inputs)
    out = tf.keras.layers.Dense(120, activation='tanh')(out)
    out = tf.keras.layers.Dense(80, activation='tanh')(out)
    # out = tf.keras.layers.Dense(200, activation='relu')(out)
    # out.trainable = False
    # out = tf.keras.layers.Dense(80, activation='relu',kernel_regularizer=regularizers.l2(0.001))(out)
    # out = tf.keras.layers.Dense(50, activation='relu')(out)
    predictions = tf.keras.layers.Dense(Network_output)(out)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.load_weights('real_h_2.h5')
    model.layers[1].trainable = False  # frozen layers
    model.layers[2].trainable = False
    # model.layers[3].trainable = False
    # model.layers[4].trainable = False
    model.summary()
    optimizer = tfa.optimizers.AdamW(weight_decay=1e-5, learning_rate=0.01)

    file_name = 'D:\pycharm_work\metasurfaces'
    callback1 = keras.callbacks.ModelCheckpoint(filepath=file_name, monitor='val_loss', verbose=1, save_best_only=True,
                                                save_weights_only=False, mode='min', period=1)
    callback2 = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, verbose=0, mode='min',
                                                  epsilon=0.0001, cooldown=0, min_lr=0.001)

    model.compile(optimizer=optimizer, loss=tf.losses.mse)
    history1 = model.fit(train_db, epochs=1500, validation_data=val_db, validation_freq=1
                         , callbacks=[callback1, callback2]
                         )
    # lossy = np.array(history1.history['loss'])      #record loss
    # lossy = lossy.reshape(-1, 1)
    # loss_val = np.array(history1.history['val_loss'])
    # loss_val = loss_val.reshape(-1, 1)
    # print(history1.history)
    # model.save('r_spectrum.h5')
    # model.save('my_weights_imag.h5')


if __name__ == '__main__':
    begin_time = time()
    train()
    end_time = time()
    run_time = end_time - begin_time
    print('running time：', run_time)
