from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
import numpy as np
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split


def noisy_data_generation(noise_typ, image, noise_factor):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + noise_factor*gauss
        return noisy

    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals*noise_factor) / float(vals)
        return noisy

    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + noise_factor*image * gauss
        return noisy


def load_and_generate_noisy_data():
    # original data
    (x_t, _), (x_te, _) = mnist.load_data()

    # total number of training and testing data
    x_t = x_t.astype('float32') / 255.
    x_te = x_te.astype('float32') / 255.

    # Gaussian noise corrupted training data 20000
    x_t_gauss = noisy_data_generation("gauss", x_t[0:20000, :, :], 0.5)

    # Speckle noise corrupted training data 20000
    x_t_speckle = noisy_data_generation("speckle", x_t[20000:40000, :, :], 0.5)

    # Gaussian and Speckle noise corrupted training data 20000
    x_t_both = 0.5*noisy_data_generation("gauss", x_t[40000:, :, :], 0.5) + \
        0.5*noisy_data_generation("speckle", x_t[40000:, :, :], 0.5)

    # # Gaussian noise corrupted test data 4000
    # x_te_gauss = x_te[0:4000, :, :]
    #
    # # Speckle noise corrupted test data 20000
    # x_te_speckle = x_te[4000:7000, :, :]
    #
    # # Gaussian and Speckle noise corrupted test data 20000
    # x_te_both = x_te[7000:, :, :]

    # labeled training and test data
    x_t = np.reshape(x_t, (len(x_t), 28, 28, 1))
    # x_te = np.reshape(x_te, (len(x_te), 28, 28, 1))

    # training noisy data
    x_t_noisy = np.vstack((x_t_gauss, x_t_speckle, x_t_both))

    # x_te_noisy = np.vstack((x_te_gauss, x_te_speckle, x_te_both))

    x_t_noisy = np.reshape(x_t_noisy, (len(x_t_noisy), 28, 28, 1))
    # x_te_noisy = np.reshape(x_te_noisy, (len(x_te_noisy), 28, 28, 1))

    # noise_factor = 0.5
    # x_t_noisy = x_t + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_t.shape)
    # x_te_noisy = x_te + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_te.shape)

    x_t_noisy = np.clip(x_t_noisy, 0., 1.)
    # x_te_noisy = np.clip(x_te_noisy, 0., 1.)
    x_t, x_te, x_t_noisy, x_te_noisy = train_test_split(x_t, x_t_noisy, test_size=0.2, random_state=42)

    return x_t, x_te, x_t_noisy, x_te_noisy


def build_network(input_size=(28, 28, 1)):
    input_img = Input(input_size)

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
    decoded_imgs = model.predict(test_data)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_data[i].reshape(28, 28))
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

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
# ax1.imshow(x_train[0, :, :, 0])
# ax2.imshow(x_train_noisy[0, :, :, 0])
# ax3.imshow(x_test[0, :, :, 0])
# ax4.imshow(x_test_noisy[0, :, :, 0])
# plt.show()

auto_encoder_base = build_network()
num_iterations = 1
auto_encoder_base = train_network(auto_encoder_base, num_iterations, x_train_noisy, x_train, x_test_noisy, x_test)
auto_encoder_base.save('basic_auto_encoder_de-noising.h5')
network_prediction(auto_encoder_base, x_test_noisy)
