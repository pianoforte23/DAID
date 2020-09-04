import numpy as np
import tensorflow as tf
import cv2
import keras.backend as K
from keras.engine.topology import Layer

def gaussian_blur_func(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    if blur.ndim == 2:
        blur = np.reshape(blur, (blur.shape[0], blur.shape[1], 1))
        
    return blur

def gaussian_blur_tensor_func(inputs):
    blur_batch = []
    img_batch = inputs.numpy()
    for img in img_batch:
        blur = gaussian_blur_func(np.float32(img))
        blur_batch.append(np.expand_dims(blur, axis=0))
    
    return np.concatenate(blur_batch, axis=0)

def median_blur_func(img):
    blur = cv2.medianBlur(img, 3)
    if blur.ndim == 2:
        blur = np.reshape(blur, (blur.shape[0], blur.shape[1], 1))
        
    return blur

def median_blur_tensor_func(img_batch):
    blur_batch = []
    for img in img_batch:
        blur = median_blur_func(np.float32(img))
        blur_batch.append(np.expand_dims(blur, axis=0))
    
    return np.concatenate(blur_batch, axis=0)

def non_local_means_func(img):
    denoised = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    if img.ndim == 3:
        if img.shape[2] == 3:
            denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    if denoised.ndim == 2:
        denoised = np.reshape(denoised, (denoised.shape[0], denoised.shape[1], 1))
            
    return denoised

def non_local_means_tensor_func(img_batch):
    denoised_batch = []
    for img in img_batch:
        img_scaled = np.uint8(img * 255)
        denoised = non_local_means_func(img_scaled)
        denoised_scaled = denoised.astype('float32') / 255
        denoised_batch.append(np.expand_dims(denoised_scaled, axis=0))
    
    return np.concatenate(denoised_batch, axis=0)

class GaussianBlur(Layer):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        
    def build(self, input_shape):
        self.out_shape = input_shape
        super(GaussianBlur, self).build(input_shape)
        
    def call(self, xin):
        xout = tf.py_function(gaussian_blur_tensor_func, [xin], 'float32')
        xout.set_shape(self.out_shape)
        return xout
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
class MedianBlur(Layer):
    def __init__(self):
        super(MedianBlur, self).__init__()
     
    def build(self, input_shape):
        self.out_shape = input_shape
        super(MedianBlur, self).build(input_shape)
        
    def call(self, xin):
        xout = tf.py_function(median_blur_tensor_func, [xin], 'float32')
        xout.set_shape(self.out_shape)
        return xout
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
class NonLocalMeans(Layer):
    def __init__(self):
        super(NonLocalMeans, self).__init__()
        
    def build(self, input_shape):
        self.out_shape = input_shape
        super(NonLocalMeans, self).build(input_shape)
        
    def call(self, xin):
        xout = tf.py_function(non_local_means_tensor_func, [xin], 'float32')
        xout.set_shape(self.out_shape)
        return xout
    
    def compute_output_shape(self, input_shape):
        return input_shape