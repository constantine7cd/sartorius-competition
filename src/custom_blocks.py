from tensorflow.keras.layers import Conv2DTranspose, LeakyReLU


def conv_transpose_norm_block(x, filters, ker_reg, name, norm=None):
    x = Conv2DTranspose(filters, 3, padding='same',
                        name=f'{name}/conv_transpose', kernel_regularizer=ker_reg)(x)

    if norm is not None:
        x = norm(name=f'{name}/norm')(x)

    x = LeakyReLU(alpha=0.2, name=f'{name}/leaky_relu')(x)

    return x
