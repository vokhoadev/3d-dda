import tensorflow as tf
from keras.layers import Conv3D, UpSampling3D, BatchNormalization, add, Input
from keras.models import Model

kernel_initializer = 'he_uniform'
interpolation = "nearest"

def conv_block_3D(x, filters, block_type, repeat=1, dilation_rate=1, size=3, padding='same'):
    result = x

    for i in range(repeat):
        if block_type == 'separated':
            result = separated_conv3D_block(result, filters, size=size, padding=padding)
        elif block_type == 'duckv2':
            result = duckv2_conv3D_block(result, filters, size=size)
        elif block_type == 'midscope':
            result = midscope_conv3D_block(result, filters)
        elif block_type == 'widescope':
            result = widescope_conv3D_block(result, filters)
        elif block_type == 'resnet':
            result = resnet_conv3D_block(result, filters, dilation_rate)
        elif block_type == 'conv':
            result = Conv3D(filters, (size, size, size), activation='relu', kernel_initializer=kernel_initializer, padding=padding)(result)
        elif block_type == 'double_convolution':
            result = double_convolution_with_batch_normalization_3D(result, filters, dilation_rate)
        else:
            return None

    return result

def duckv2_conv3D_block(x, filters, size):
    x = BatchNormalization(axis=-1)(x)
    
    # The widescope, midscope, and resnet blocks should also be adapted for 3D operation as shown in previous examples.
    x1 = widescope_conv3D_block(x, filters)
    x2 = midscope_conv3D_block(x, filters)
    
    x3 = conv_block_3D(x, filters, 'resnet', repeat=1)
    x4 = conv_block_3D(x, filters, 'resnet', repeat=2)
    x5 = conv_block_3D(x, filters, 'resnet', repeat=3)
    
    # The separated block must also be adjusted to use 3D convolutions.
    x6 = separated_conv3D_block(x, filters, size=6, padding='same')

    x = add([x1, x2, x3, x4, x5, x6])
    x = BatchNormalization(axis=-1)(x)

    return x

def separated_conv3D_block(x, filters, size=3, padding='same'):
    x = Conv3D(filters, (1, 1, size), activation='relu', kernel_initializer=kernel_initializer, padding=padding)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv3D(filters, (size, 1, 1), activation='relu', kernel_initializer=kernel_initializer, padding=padding)(x)
    x = BatchNormalization(axis=-1)(x)
    return x

def midscope_conv3D_block(x, filters):
    x = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', dilation_rate=1)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', dilation_rate=2)(x)
    x = BatchNormalization(axis=-1)(x)
    return x

def widescope_conv3D_block(x, filters):
    x = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', dilation_rate=1)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', dilation_rate=2)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', dilation_rate=3)(x)
    x = BatchNormalization(axis=-1)(x)
    return x

def resnet_conv3D_block(x, filters, dilation_rate=1):
    x1 = Conv3D(filters, (1, 1, 1), activation='relu', kernel_initializer=kernel_initializer, padding='same', dilation_rate=dilation_rate)(x)
    x = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization(axis=-1)(x)
    x_final = add([x, x1])
    x_final = BatchNormalization(axis=-1)(x_final)
    return x_final

def double_convolution_with_batch_normalization_3D(x, filters, dilation_rate=1):
    x = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv3D(filters, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization(axis=-1)(x)
    return x

def DuckNet3D(img_depth, img_height, img_width, input_channels, out_classes, starting_filters):
    input_layer = Input((img_depth, img_height, img_width, input_channels))

    print('Starting 3D DUCK-Net')

    # Define downsampling path
    p1 = Conv3D(starting_filters * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(input_layer)
    p2 = Conv3D(starting_filters * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(p1)
    p3 = Conv3D(starting_filters * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(p2)
    p4 = Conv3D(starting_filters * 16, (2, 2, 2), strides=(2, 2, 2), padding='same')(p3)
    p5 = Conv3D(starting_filters * 32, (2, 2, 2), strides=(2, 2, 2), padding='same')(p4)

    # Initial 3D block
    t0 = conv_block_3D(input_layer, starting_filters, 'duckv2', repeat=1)

    # Connecting blocks with downsampling path
    l1i = Conv3D(starting_filters * 2, (2, 2, 2), strides=(2, 2, 2), padding='same')(t0)
    s1 = add([l1i, p1])
    t1 = conv_block_3D(s1, starting_filters * 2, 'duckv2', repeat=1)

    l2i = Conv3D(starting_filters * 4, (2, 2, 2), strides=(2, 2, 2), padding='same')(t1)
    s2 = add([l2i, p2])
    t2 = conv_block_3D(s2, starting_filters * 4, 'duckv2', repeat=1)

    l3i = Conv3D(starting_filters * 8, (2, 2, 2), strides=(2, 2, 2), padding='same')(t2)
    s3 = add([l3i, p3])
    t3 = conv_block_3D(s3, starting_filters * 8, 'duckv2', repeat=1)

    l4i = Conv3D(starting_filters * 16, (2, 2, 2), strides=(2, 2, 2), padding='same')(t3)
    s4 = add([l4i, p4])
    t4 = conv_block_3D(s4, starting_filters * 16, 'duckv2', repeat=1)

    l5i = Conv3D(starting_filters * 32, (2, 2, 2), strides=(2, 2, 2), padding='same')(t4)
    s5 = add([l5i, p5])
    t51 = conv_block_3D(s5, starting_filters * 32, 'resnet', repeat=2)
    t53 = conv_block_3D(t51, starting_filters * 16, 'resnet', repeat=2)

    # Upsampling path
    l5o = UpSampling3D(size=(2, 2, 2))(t53)
    c4 = add([l5o, t4])
    q4 = conv_block_3D(c4, starting_filters * 8, 'duckv2', repeat=1)

    l4o = UpSampling3D(size=(2, 2, 2))(q4)
    c3 = add([l4o, t3])
    q3 = conv_block_3D(c3, starting_filters * 4, 'duckv2', repeat=1)

    l3o = UpSampling3D(size=(2, 2, 2))(q3)
    c2 = add([l3o, t2])
    q6 = conv_block_3D(c2, starting_filters * 2, 'duckv2', repeat=1)

    l2o = UpSampling3D(size=(2, 2, 2))(q6)
    c1 = add([l2o, t1])
    q1 = conv_block_3D(c1, starting_filters, 'duckv2', repeat=1)

    l1o = UpSampling3D(size=(2, 2, 2))(q1)
    c0 = add([l1o, t0])
    z1 = conv_block_3D(c0, starting_filters, 'duckv2', repeat=1)

    # Output layer
    output = Conv3D(out_classes, (1, 1, 1), activation='sigmoid')(z1)

    model = Model(inputs=input_layer, outputs=output)

    return model
