from keras import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, Activation, Input
from keras import backend as K

def mnist_model(input_shape=(28, 28, 1)):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10)(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def mnist_dae(input_shape=(28, 28, 1)):
    kernel_size = 3
    latent_dim = 16
    layer_filters = [32, 64]
    
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    
    for filters in layer_filters:
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=2, activation='relu', padding='same')(x)
    
    shape = K.int_shape(x)
    
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)
    encoder = Model(inputs, latent, name='encoder')
    
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    
    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2, activation='relu', padding='same')(x)
        
    x = Conv2DTranspose(filters=1, kernel_size=kernel_size, padding='same')(x)
    outputs = Activation('sigmoid', name='decoder_output')(x)
    
    decoder = Model(latent_inputs, outputs, name='decoder')
    autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    
    return autoencoder