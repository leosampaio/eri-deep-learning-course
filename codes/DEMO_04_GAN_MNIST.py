import tensorflow as tf
import numpy as np
import keras
import sklearn

import matplotlib
matplotlib.use('Agg')  # this sets matplotlib to plot to files instead of the screen
import matplotlib.pyplot as plt

# import the dataset, load dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# import relevant classes from keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import Input, Flatten, Reshape, Concatenate, BatchNormalization, Dropout
from keras.models import Model
import keras.backend as K


def discriminator_loss(y_true, y_pred):

    # extract our values from the concatenated y's
    label_real = y_true[..., 0]  # labels 0
    label_false = y_true[..., 1]  # labels 1

    # discriminator classification for real samples (restricted so that the log doesn't explode)
    pred_real = K.clip(y_pred[:, 0], K.epsilon(), 1.0 - K.epsilon())
    # discriminator classification for false samples (restricted so that the log doesn't explode)
    pred_false = K.clip(y_pred[:, 1], K.epsilon(), 1.0 - K.epsilon())

    # loss function as defined by Goodfellow et al.
    return -K.mean(K.log(K.abs(pred_real - label_real)) + K.log(K.abs(pred_false - label_false)))


def generator_loss(y_true, y_pred):

    # extract our values from the concatenated y's
    label_real = y_true[..., 0]  # labels 0
    label_false = y_true[..., 1]  # labels 1

    # discriminator classification for real samples (restricted so that the log doesn't explode)
    pred_real = K.clip(y_pred[:, 0], K.epsilon(), 1.0 - K.epsilon())
    # discriminator classification for false samples (restricted so that the log doesn't explode)
    pred_false = K.clip(y_pred[:, 1], K.epsilon(), 1.0 - K.epsilon())

    # the oposite of the discriminator loss, notice the invertion between label_false and label_real
    return -K.mean(K.log(K.abs(pred_real - label_false)) + K.log(K.abs(pred_false - label_real)))

# since MNIST images have a single channel (not RGB, but black and white)
# we need to include a dummy channel at the end of the definition to make this explicit
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))


# definition of the discriminator's architecture
discriminator_input_x = Input((28, 28, 1))

# convolutional layers
x = Conv2D(32, (3, 3), activation='relu')(discriminator_input_x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

# fully connected layers
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
discriminator_output = Dense(1, activation='relu')(x)

discriminator = Model(discriminator_input_x, discriminator_output)

# definition of the generator's archintecture
# notice how the transposed convolution coupled with stride slowly increases
# the resulting output size
generator_input_z = Input((196,))
z = Reshape((7, 7, 4))(generator_input_z)
z = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(z)
z = BatchNormalization()(z)
z = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(z)
z = BatchNormalization()(z)
generator_output = Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='same')(z)

generator = Model(generator_input_z, generator_output)

# now we build the complete GAN model
# made into a funtion so that we can create two models, one to train the
# discriminator, the other to train the generator


def build_GAN(discriminator, generator):
    gan_input_x = Input((28, 28, 1))
    gan_input_z = Input((196,))
    out_real = discriminator(gan_input_x)
    out_false = discriminator(generator(gan_input_z))

    # concatenate the discriminator's classification for real and generated samples
    out_dis = Concatenate(axis=-1)([out_real, out_false])

    return Model([gan_input_x, gan_input_z], out_dis)

# create two models out of this complete architecture, one for each loss function
gan_that_trains_discriminator = build_GAN(discriminator, generator)
gan_that_trains_generator = build_GAN(discriminator, generator)

gan_that_trains_discriminator.compile(loss=discriminator_loss, optimizer='sgd')
gan_that_trains_generator.compile(loss=generator_loss, optimizer='sgd')

# manually perform training, so that we can train generator and discriminator
# sequentially
epochs = 100
batchsize = 100
for e in range(epochs):

    # create a permutation of the data indexes, so that each epoch learns on
    # a different order of the dataset samples (improves convergence)
    dataset_permutation = np.random.permutation(len(x_train))
    for b in range(0, len(x_train), batchsize):

        # get our batch os real samples and generate necessary z samples
        batch_x = x_train[dataset_permutation[b:b + batchsize]]
        batch_z = np.random.normal(0, 1.0, (batchsize, 196))

        # generate 0's and 1's to act as the discriminator labels
        y_real = np.ones(shape=(batchsize, 1))
        y_false = np.zeros(shape=(batchsize, 1))
        y = np.concatenate([y_real, y_false], axis=-1)

        # train discriminator
        d_loss = gan_that_trains_discriminator.train_on_batch([batch_x, batch_z], y)
        if d_loss > 10.:  # train twice if loss is too high
            d_loss = gan_that_trains_discriminator.train_on_batch([batch_x, batch_z], y)

        # train generators
        g_loss = gan_that_trains_generator.train_on_batch([batch_x, batch_z], y)

        # print current status
        print("Epoch #{} | Sample {}/{} | dis_loss: {}, gen_loss: {}".format(e, b, len(x_train), d_loss, g_loss))
        if d_loss == g_loss:
            print("Training Failure: D ang G should never have same loss, "
                  "if it keeps happening, try restarting (it may take some tries to work)")

    # every epoch, generate 100 images from random z's and save

    np.random.seed(14)  # fix the seed so that we always get same samples
    z_samples = np.random.normal(size=(100, 196))
    np.random.seed()  # restore seed

    # generate images
    generated_images = generator.predict(z_samples)

    # plot a grid of images using matplotlib
    fig = plt.figure(figsize=(8, 8))
    grid = matplotlib.gridspec.GridSpec(10, 10, wspace=0.1, hspace=0.1)
    for i in range(100):
        ax = plt.Subplot(fig, grid[i])
        ax.imshow(np.squeeze(generated_images[i, ...]), cmap='gray', interpolation='none', vmin=0.0, vmax=1.0)
        ax.axis('off')
        fig.add_subplot(ax)

    # save to file with epoch number included
    fig.savefig('generated_images_e{}.jpg'.format(e), dpi=200)
    plt.close(fig)
