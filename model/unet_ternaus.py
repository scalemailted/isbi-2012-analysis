import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.applications.vgg16 import VGG16

"""
A Keras implementation of TernausNet16: 
https://arxiv.org/abs/1801.05746
https://github.com/ternaus/TernausNet
The architecture is very similar to the original U-net paper:
https://arxiv.org/abs/1505.04597
The key differences are:
- A VGG16 architecture is used for encoder, pretrained on ImageNet
- No batchnorm used
- No dropout used
- Shortcut concatenations are mismatched on number of filters meaning that 
  a larger number of filters is used in decoder.
"""


def decoder_block_ternausV2(inputs, mid_channels, out_channels):
    """
    Decoder block as proposed for TernausNet16 here: 
    https://arxiv.org/abs/1801.05746
    See DecoderBlockV2 here:
    https://github.com/ternaus/TernausNet/blob/master/unet_models.py
    - Concatenate u-net shortcut to input pre-upsample
    - Bilinear upsample input to double Height and Width dimensions
    - Note: The original ternausNet implementation includes option for 
      deconvolution instead of bilinear upsampling. Omitted here because I 
      couldn't find a meaningful performance comparison
    """
    
    conv_kwargs = dict(
        activation='relu',
        padding='same',
        kernel_initializer='he_normal',
        data_format='channels_last'  # (batch, height, width, channels)
    )

    x = UpSampling2D(size=(2, 2))(inputs) # interpolation='bilinear' doesn't work?
    x = Conv2D(mid_channels, 3, **conv_kwargs)(x)
    x = Conv2D(out_channels, 3, **conv_kwargs)(x)
    return x


# INTENDED API
# ------------------------------------------------------------------------------

def ternausNet16(input_size=(256, 256, 3), output_channels=1):
    """
    A Keras implementation of TernausNet16: 
    https://arxiv.org/abs/1801.05746
    https://github.com/ternaus/TernausNet
    """

    # input 
    # convert 1 channel grayscale to 3 channels if needed
    inputs = Input(input_size)
    if input_size[-1] < 3:
        x = Conv2D(3, 1)(inputs)                         # add channels
        input_shape = (input_size[0], input_size[0], 3)  # update input size
    else:
        x = inputs
        input_shape = input_size
    
    # Load pretrained VGG, conv layers include relu activation
    encoder = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
       
    # (None, 256, 256, 3)
    e1 = encoder.get_layer(name='block1_conv1')(x)
    e1 = encoder.get_layer(name='block1_conv2')(e1)
    # (None, 256, 256, 64)
    e2 = MaxPooling2D(pool_size=(2, 2))(e1)
    e2 = encoder.get_layer(name='block2_conv1')(e2)
    e2 = encoder.get_layer(name='block2_conv2')(e2)
    # (None, 128, 128, 128)
    e3 = MaxPooling2D(pool_size=(2, 2))(e2)
    e3 = encoder.get_layer(name='block3_conv1')(e3)
    e3 = encoder.get_layer(name='block3_conv2')(e3)
    e3 = encoder.get_layer(name='block3_conv3')(e3)
    # (None, 64, 64, 256)
    e4 = MaxPooling2D(pool_size=(2, 2))(e3)
    e4 = encoder.get_layer(name='block4_conv1')(e4)
    e4 = encoder.get_layer(name='block4_conv2')(e4)
    e4 = encoder.get_layer(name='block4_conv3')(e4)
    # (None, 32, 32, 512)
    e5 = MaxPooling2D(pool_size=(2, 2))(e4)
    e5 = encoder.get_layer(name='block5_conv1')(e5)
    e5 = encoder.get_layer(name='block5_conv2')(e5)
    e5 = encoder.get_layer(name='block5_conv3')(e5)
    # (None, 16, 16, 512)
    center = MaxPooling2D(pool_size=(2, 2))(e5)
    # (None, 8, 8, 512)
    center = decoder_block_ternausV2(center, 512, 256)
    # (None, 16, 16, 256)
    d5 = concatenate([e5, center], axis=3)
    d5 = decoder_block_ternausV2(d5, 512, 256)
    # (None, 32, 32, 256)
    d4 = concatenate([e4, d5], axis=3)
    d4 = decoder_block_ternausV2(d4, 512, 128)
    # (None, 64, 64, 128)
    d3 = concatenate([e3, d4], axis=3)
    d3 = decoder_block_ternausV2(d3, 256, 64)
    # (None, 128, 128, 64)
    d2 = concatenate([e2, d3], axis=3)
    d2 = decoder_block_ternausV2(d2, 128, 64)
    # (None, 256, 256, 64)
    # Note: no decoder block used at end
    d1 = concatenate([e1, d2], axis=3)
    d1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(d1)
    d1 = ReLU()(d1)
    # (None, 256, 256, 32)

    # Output
    if output_channels > 1:
        # untested
        op = tf.nn.log_softmax_v2(d1, axis=3)
    else:
        op = Conv2D(output_channels, 1)(d1)
        op = Activation('sigmoid')(op)  # note: ternaus excludes

    # Build
    model = Model(inputs=[inputs], outputs=[op])
    return model
