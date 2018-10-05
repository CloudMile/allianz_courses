import numpy as np
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Nadam
from matplotlib import pyplot as plt


def plot_result(hist):
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 2, 1)
    plt.plot(hist.history['loss'], label='tr_loss')
    plt.plot(hist.history['val_loss'], label='vl_loss')
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(hist.history['acc'], label='acc')
    plt.plot(hist.history['val_acc'], label='val_acc')
    plt.title('Accuracy')

    plt.legend(loc='best')
    plt.show()


def woe_encode(x, label, data):
    """Calculate the Weight of Evidence of given categorical feature and label

    :param x: Given feature name
    :param label: Label name
    :param data:
    :return: Woe encoded dictionary
    """
    total_vc = data[label].value_counts().sort_index()

    def woe(pipe, total_vc):
        # Count by label in this group
        group_vc = pipe[label].value_counts().sort_index()

        # Some class in the feature is missing, fill zero to missing class
        if len(group_vc) < len(total_vc):
            for key in total_vc.index:
                if key not in group_vc:
                    group_vc[key] = 0.
            group_vc = group_vc.sort_index()

        # WOE formula
        r = ((group_vc + 0.5) / total_vc).values

        # Odd ratio => 1 to 0, you can define meaning of each class
        return np.log(r[1] / r[0])

    return data.groupby(x).apply(lambda pipe: woe(pipe, total_vc))


def variation_autoencoder(tr_x, vl_x):
    from keras import backend as K
    from keras.layers import Lambda, Input
    from keras.optimizers import mse
    from keras import callbacks

    K.clear_session()

    original_dim = tr_x.shape[1]
    # network parameters
    input_shape = (original_dim,)
    batch_size = 500
    latent_dim = 32
    epochs = 200

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')

    x = Dense(64, activation='selu')(inputs)
    x = Dense(64, activation='selu')(x)
    z_mean = Dense(latent_dim, activation='linear')(x)
    z_log_var = Dense(latent_dim, activation='linear')(x)

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # build decoder model
    x_decoded = Dense(64, activation='selu')(z)
    x_decoded = Dense(64, activation='selu')(x_decoded)
    x_decoded = Dense(original_dim, activation='linear')(x_decoded)

    # Instantiate VAE model
    vae = Model(inputs, x_decoded)

    # VAE loss = mse_loss or xent_loss + kl_loss
    reconstruction_loss = mse(inputs, x_decoded)
    # reconstruction_loss = binary_crossentropy(inputs, x_decoded)

    reconstruction_loss *= original_dim
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=Nadam())
    vae.summary()

    # train the autoencoder
    hist = vae.fit(tr_x,
                   epochs=epochs,
                   batch_size=batch_size,
                   validation_data=(vl_x, None),
                   callbacks=[callbacks.ModelCheckpoint(filepath='vae_mlp.h5', save_best_only=True)])

    return encoder
    # vae.save_weights('vae_mlp.h5')
    # plot_ae(hist)

