import tensorflow as tf

def L1loss(x, y): # shape(# batch, h, w, 2)
    return tf.reduce_mean(tf.pow(tf.abs(x-y), 1))

def sobel_edges(img):
    ch = img.get_shape().as_list()[3]
    kerx = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype='float32')
    kery = tf.constant([[-1, -2, 1], [0, 0, 0], [1, 2, 1]], dtype='float32')
    kerx = tf.expand_dims(kerx, 2)
    kerx = tf.expand_dims(kerx, 3)
    kerx = tf.tile(kerx, [1, 1, ch, 1])
    kery = tf.expand_dims(kery, 2)
    kery = tf.expand_dims(kery, 3)
    kery = tf.tile(kery, [1, 1, ch, 1])
    gx = tf.nn.depthwise_conv2d_native(img, kerx, strides=[1,1,1,1], padding="VALID")
    gy = tf.nn.depthwise_conv2d_native(img, kery, strides=[1,1,1,1], padding="VALID")
    return tf.concat([gx, gy], 3)

def sobel_gradient_loss(guess, truth):
    g1 = sobel_edges(guess)
    g2 = sobel_edges(truth)
    return tf.reduce_mean(tf.pow(tf.abs(g1 - g2), 1))

def sobel_gradient_loss_l2(guess, truth):
    g1 = sobel_edges(guess)
    g2 = sobel_edges(truth)
    return tf.reduce_mean(tf.pow(tf.abs(g1 - g2), 2))

def L2loss(x, y): # shape(# batch, h, w, 2)
    return tf.reduce_mean(tf.pow(tf.abs(x-y), 2))

def Lrloss(x, y):
     return tf.reduce_mean(tf.pow(tf.abs(x-y)+1e-3, 0.4))


