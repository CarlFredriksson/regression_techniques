import numpy as np
import tensorflow as tf

def linear_regression_1D(X_train, Y_train, X_test, Y_test, learning_rate, num_iterations):
    y_est = []

    # Create parameters
    a_estimate = tf.Variable(0, dtype=tf.float32)
    b_estimate = tf.Variable(0, dtype=tf.float32)

    # Compute cost
    X_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="X_placeholder")
    Y_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="Y_placeholder")
    Y_estimate = a_estimate * X_placeholder + b_estimate
    J = tf.reduce_mean((Y_placeholder - Y_estimate)**2)

    # Create train op
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(J)

    # Start session
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        # Training loop
        for i in range(num_iterations):
            sess.run(train_op, feed_dict={X_placeholder: X_train, Y_placeholder: Y_train})

        # Create estimated Y values
        y_est, J_train = sess.run([Y_estimate, J], feed_dict={X_placeholder: X_train, Y_placeholder: Y_train})
        J_test = sess.run(J, feed_dict={X_placeholder: X_test, Y_placeholder: Y_test})

    return y_est, J_train, J_test
