import setup_mnist
import denoise_layer
from keras import Model
from keras.layers import Input

class DM_GAUSSIAN_BLUR:
    def __init__(self, input_shape=(28, 28, 1)):
        self.model = setup_mnist.mnist_model(input_shape=input_shape)
        
        inputs = Input(shape=input_shape)
        denoiser_outputs = denoise_layer.GaussianBlur()(inputs)
        outputs = self.model(denoiser_outputs)
        
        self.denoiser = Model(inputs=inputs, outputs=denoiser_outputs)
        self.denoise_model = Model(inputs=inputs, outputs=outputs)
        
class DM_MEDIAN_BLUR:
    def __init__(self, input_shape=(28, 28, 1)):
        self.model = setup_mnist.mnist_model(input_shape=input_shape)
        
        inputs = Input(shape=input_shape)
        denoiser_outputs = denoise_layer.MedianBlur()(inputs)
        outputs = self.model(denoiser_outputs)
        
        self.denoiser = Model(inputs=inputs, outputs=denoiser_outputs)
        self.denoise_model = Model(inputs=inputs, outputs=outputs)
        
class DM_NL_MEANS:
    def __init__(self, input_shape=(28, 28, 1)):
        self.model = setup_mnist.mnist_model(input_shape=input_shape)
        
        inputs = Input(shape=input_shape)
        denoiser_outputs = denoise_layer.NonLocalMeans()(inputs)
        outputs = self.model(denoiser_outputs)
        
        self.denoiser = Model(inputs=inputs, outputs=denoiser_outputs)
        self.denoise_model = Model(inputs=inputs, outputs=outputs)
        
class DM_DAE:
    def __init__(self, input_shape=(28, 28, 1)):
        self.denoiser = setup_mnist.mnist_dae(input_shape=input_shape)
        self.model = setup_mnist.mnist_model(input_shape=input_shape)
        
        inputs = Input(shape=input_shape)
        denoiser_outputs = self.denoiser(inputs)
        outputs = self.model(denoiser_outputs)
        
        self.denoise_model = Model(inputs=inputs, outputs=outputs)