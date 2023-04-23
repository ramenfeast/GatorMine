import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    pool = MaxPooling2D((2, 2))(x)
    return x, pool

def decoder_block(inputs, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    inputs = Input(input_shape)

    stage1, pool1 = encoder_block(inputs, 64)
    stage2, pool2 = encoder_block(pool1, 128)
    stage3, pool3 = encoder_block(pool2, 256)
    stage4, pool4 = encoder_block(pool3, 512)

    bottleneck = conv_block(pool4, 1024)

    upsample1 = decoder_block(bottleneck, stage4, 512)
    upsample2 = decoder_block(upsample1, stage3, 256)
    upsample3 = decoder_block(upsample2, stage2, 128)
    upsample4 = decoder_block(upsample3, stage1, 64)

    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(upsample4)

    model = Model(inputs, outputs, name='unet')
    return model

if __name__ == '__main__':
    input_shape = (512, 512, 3)
    model = unet(input_shape)
    model.summary()
