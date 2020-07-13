from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
import numpy as np
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from time import time


def load_and_generate_noisy_data():
    (x_t, _), (x_te, _) = mnist.load_data()

    x_t = x_t.astype('float32') / 255.
    x_te = x_te.astype('float32') / 255.
    x_t = np.reshape(x_t, (len(x_t), 28, 28, 1))
    x_te = np.reshape(x_te, (len(x_te), 28, 28, 1))

    noise_factor = 0.5
    x_t_noisy = x_t + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_t.shape)
    x_te_noisy = x_te + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_te.shape)

    x_t_noisy = np.clip(x_t_noisy, 0., 1.)
    x_te_noisy = np.clip(x_te_noisy, 0., 1.)

    return x_t, x_te, x_t_noisy, x_te_noisy


def build_network():
    input_img = Input(shape=(28, 28, 1))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    auto_encoder = Model(input_img, decoded)
    return auto_encoder


def train_network(model, iterations, train_data, train_label, test_data, test_label):

    model.compile(optimizer='adadelta', loss='binary_crossentropy')

    model.fit(train_data, train_label, epochs=iterations, batch_size=128, shuffle=True,
              validation_data=(test_data, test_label),
              callbacks=[TensorBoard(log_dir='log_dn/{}'.format(time()))])
    return model


def network_prediction(model, test_data):
    decoded_imgs = auto_encoder_base.predict(x_test_noisy)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


x_train, x_test, x_train_noisy, x_test_noisy = load_and_generate_noisy_data()
auto_encoder_base = build_network()
num_iterations = 1
auto_encoder_base = train_network(auto_encoder_base, num_iterations, x_train_noisy, x_train, x_test_noisy, x_test)
auto_encoder_base.save('basic_autoencoder_denoising.h5')
network_prediction(auto_encoder_base, x_test_noisy)

