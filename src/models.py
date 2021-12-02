import math
import string

import keras_efficientnet_v2 as efn2
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Concatenate, Conv2D,
                                     Conv2DTranspose, LayerNormalization,
                                     UpSampling2D)
from tensorflow.keras.regularizers import l2

from src.custom_blocks import conv_transpose_norm_block
from src.layers import Identity

KEY_BLOCKS_V2 = [5, 3, 2, 1]
DEPTHS_V2 = {
    's': [1, 3, 3, 5, 8, 14],
    'm': [2, 4, 4, 6, 13, 17, 4],
    'l': [3, 6, 6, 9, 18, 24, 6],
    'xl': [3, 7, 7, 15, 23, 31, 7]
}


def EffNetV2(input_shape=(224, 224, 3), efficientnet_char='s', weights='imagenet'):

    model = getattr(efn2, f'EfficientNetV2{efficientnet_char.upper()}')(
        input_shape=input_shape,
        num_classes=0,
        pretrained=None
    )

    depths = DEPTHS_V2[efficientnet_char]
    adds = [layer.name for layer in model.layers if 'add' in layer.name]
    depth_init = int(adds[0].split('_')[-1]) if adds[0] != 'add' else 0

    features = ['post_swish'] + \
        [f'add_{depth_init + sum(depths[:block])}' for block in KEY_BLOCKS_V2]
    outputs = [model.get_layer(name).output for name in features]

    return tf.keras.Model(inputs=model.inputs, outputs=outputs)


def upscale_block(x1, x2, filters, name, ker_reg, norm=None):
    x1 = UpSampling2D(size=(2, 2), interpolation='bilinear',
                      name=f'{name}/upsampling2d')(x1)
    x = Concatenate(name=f'{name}/concatenate')([x1, x2])
    x = conv_transpose_norm_block(x, filters, ker_reg, f'{name}/block1', norm)
    x = conv_transpose_norm_block(x, filters, ker_reg, f'{name}/block2', norm)

    return x


def decoder_net(
    features,
    decode_filters: int,
    name: str = 'Decoder',
    reg_value: float = 0.,
    norm: str = '',
    num_filters: int = 1
):
    ker_reg = l2(reg_value) if reg_value != 0. else None

    norms = {
        'layer_norm': LayerNormalization,
        'batch_norm': BatchNormalization
    }

    norm = norms.get(norm)

    x, f4, f3, f2, f1 = features

    up0 = Conv2DTranspose(decode_filters, 1, padding='same',
                          name=f'{name}/conv2', kernel_regularizer=ker_reg)(x)
    up1 = upscale_block(up0, f4, filters=decode_filters // 2,
                        name=f'{name}/up1', ker_reg=ker_reg, norm=norm)
    up2 = upscale_block(up1, f3, filters=decode_filters // 4,
                        name=f'{name}/up2', ker_reg=ker_reg, norm=norm)
    up3 = upscale_block(up2, f2, filters=decode_filters // 8,
                        name=f'{name}/up3', ker_reg=ker_reg, norm=norm)
    up4 = upscale_block(up3, f1, filters=decode_filters // 16,
                        name=f'{name}/up4', ker_reg=ker_reg, norm=norm)
    x = Conv2DTranspose(num_filters, 3, padding='same',
                        name=f'{name}/conv3', kernel_regularizer=ker_reg)(up4)
    x = UpSampling2D(size=(2, 2), interpolation='bilinear',
                     name=f'{name}/prediction')(x)

    return x


def UnetEffNetV2(
    input_shape=(512, 512, 3),
    norm='batch_norm',
    weights='imagenet',
    reg_value=0.,
    num_filters=1
):
    encoder = EffNetV2(input_shape,
                       efficientnet_char='m', weights=weights)
    decode_filters = int(encoder.layers[-1].output[0].shape[-1])

    x = decoder_net(encoder.output, decode_filters, name='Decoder',
                    norm=norm, reg_value=reg_value, num_filters=num_filters)

    return tf.keras.Model(inputs=encoder.inputs, outputs=x)


def UnetEffNetV2Ext(
    input_shape=(512, 512, 3),
    norm='batch_norm',
    weights='imagenet',
    reg_value=0.,
    num_filters=1,
    trainable_base_model=True,
    base_model_checkpoint=None
):
    model = UnetEffNetV2(input_shape, norm, weights, reg_value, num_filters)

    if not trainable_base_model:
        model.trainable = False

    if base_model_checkpoint is not None:
        model.load_weights(base_model_checkpoint)

    x = Conv2D(filters=4, kernel_size=4, padding='same')(model.output)
    x = Conv2D(filters=1, kernel_size=1, padding='same')(x)

    embs = Identity(name='Embeddings')(model.output)
    logits = Identity(name='Logits')(x)

    return tf.keras.Model(inputs=model.inputs, outputs=[embs, logits])
