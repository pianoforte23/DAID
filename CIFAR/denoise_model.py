import segmentation_models as sm
import setup_cifar
import denoise_layer
from keras import Model
from keras.layers import Input

class DM_GAUSSIAN_BLUR:
    def __init__(self, input_shape=(32, 32, 3)):
        self.model = setup_cifar.resnet_v2(input_shape=input_shape, depth=20)
        
        inputs = Input(shape=input_shape)
        denoiser_outputs = denoise_layer.GaussianBlur()(inputs)
        outputs = self.model(denoiser_outputs)
        
        self.denoiser = Model(inputs=inputs, outputs=denoiser_outputs)
        self.denoise_model = Model(inputs=inputs, outputs=outputs)
        
class DM_MEDIAN_BLUR:
    def __init__(self, input_shape=(32, 32, 3)):
        self.model = setup_cifar.resnet_v2(input_shape=input_shape, depth=20)
        
        inputs = Input(shape=input_shape)
        denoiser_outputs = denoise_layer.MedianBlur()(inputs)
        outputs = self.model(denoiser_outputs)
        
        self.denoiser = Model(inputs=inputs, outputs=denoiser_outputs)
        self.denoise_model = Model(inputs=inputs, outputs=outputs)
        
class DM_NL_MEANS:
    def __init__(self, input_shape=(32, 32, 3)):
        self.model = setup_cifar.resnet_v2(input_shape=input_shape, depth=20)
        
        inputs = Input(shape=input_shape)
        denoiser_outputs = denoise_layer.NonLocalMeans()(inputs)
        outputs = self.model(denoiser_outputs)
        
        self.denoiser = Model(inputs=inputs, outputs=denoiser_outputs)
        self.denoise_model = Model(inputs=inputs, outputs=outputs)
        
class DM_DAE_RESNET:
    def __init__(self, input_shape=(32, 32, 3), model="resnet"):
        self.denoiser = sm.Unet('resnet18', input_shape=input_shape, classes=3, activation='sigmoid')
        if model == "resnet":
            self.model = setup_cifar.resnet_v2(input_shape=input_shape, depth=20)
        elif model == "vgg16":
            self.model = setup_cifar.model_vgg16(input_shape)
        
        inputs = Input(shape=input_shape)
        denoiser_outputs = self.denoiser(inputs)
        outputs = self.model(denoiser_outputs)
        
        self.denoise_model = Model(inputs=inputs, outputs=outputs)
        
class DM_DAE_VGG:
    def __init__(self, input_shape=(32, 32, 3), model="resnet"):
        self.denoiser = sm.Unet('vgg16', input_shape=input_shape, classes=3, activation='sigmoid')
        if model == "resnet":
            self.model = setup_cifar.resnet_v2(input_shape=input_shape, depth=20)
        elif model == "vgg16":
            self.model = setup_cifar.model_vgg16(input_shape)
        
        inputs = Input(shape=input_shape)
        denoiser_outputs = self.denoiser(inputs)
        outputs = self.model(denoiser_outputs)
        
        self.denoise_model = Model(inputs=inputs, outputs=outputs)