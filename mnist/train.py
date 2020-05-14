import numpy as np
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)

'''
construct
train
evaluate

network architecture...
convs: 32 feature maps, shaped 5x5, with ReLU activation
pool: 2x2 with stride of 2
convs: 64 feature maps, shaped 5x5, with ReLU activation
pool: 2x2 with stride of 2
dense: 1024 neurons. dropout rate = 0.4.
softmax: 10 outputs  
'''

def cnn_model_fn(features, labels, mode):
    we_are_training = mode == tf.estimator.ModeKeys.TRAIN
    """Model function for CNN..."""
    # Input layer... like a feed?
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])  #TODO: How does this shape param work? What does the -1 do?
    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dense1_dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=we_are_training)
    logits = tf.layers.dense(inputs=dense1_dropout, units=10)
    predictions = dict(
        classes=tf.argmax(input=logits, axis=1),
        probabilities=tf.nn.softmax(logits=logits, name="softmax_tensor")
    )
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())  #TODO: what does this line do?
        #TODO: EstimatorSpec can take `predictions`... what does that do?
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = dict(accuracy=tf.metrics.accuracy(labels=labels, predictions=predictions["classes"]))
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    from tensorflow.contrib.learn.python.learn.datasets import load_dataset
    mnist_dataset = load_dataset("mnist")
    train_data = mnist_dataset.train.images
    train_labels = np.asarray(mnist_dataset.train.labels, dtype=np.int32)  #TODO: what if we don't do np.asarray here?
    eval_data = mnist_dataset.test.images
    eval_labels = np.asarray(mnist_dataset.test.labels, dtype=np.int32)
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000)
    return


if __name__ == "__main__":
    tf.app.run()
