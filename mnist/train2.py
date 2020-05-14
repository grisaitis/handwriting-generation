import tensorflow as tf


def convnet_fn(features, labels, mode):
    inputs = tf.reshape(features["x"], shape=[-1, ])
    if mode == tf.estimator.ModeKeys.TRAIN:


def main(argv):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001, use_locking=False, name="GradientDescentOptimizer")
    train_op = optimizer.minimize(loss)
    estimator = tf.estimator.Estimator(convnet_fn, model_dir="/tmp/william")
if __name__ == "__main__":
    tf.app.run()
