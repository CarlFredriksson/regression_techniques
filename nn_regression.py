import numpy as np
import tensorflow as tf

def nn_regression_1D(X_train, Y_train, X_test, Y_test, learning_rate, num_iterations):
    tf.reset_default_graph()
    y_est = []
    
    # Create parameters
    W_1 = tf.get_variable("W_1", shape=(1, 10), initializer=tf.contrib.layers.xavier_initializer())
    b_1 = tf.get_variable("b_1", shape=(1, 10), initializer=tf.zeros_initializer())

    W_2 = tf.get_variable("W_2", shape=(10, 10), initializer=tf.contrib.layers.xavier_initializer())
    b_2 = tf.get_variable("b_2", shape=(1, 10), initializer=tf.zeros_initializer())

    W_3 = tf.get_variable("W_3", shape=(10, 1), initializer=tf.contrib.layers.xavier_initializer())
    b_3 = tf.get_variable("b_3", shape=(1, 1), initializer=tf.zeros_initializer())

    # Forward propagation
    X_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="X_placeholder")
    Y_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="Y_placeholder")

    X = tf.matmul(X_placeholder, W_1) + b_1
    X = tf.nn.relu(X)
    X = tf.matmul(X, W_2) + b_2
    X = tf.nn.relu(X)
    Y_estimate = tf.matmul(X, W_3) + b_3

    # Compute cost
    J = tf.reduce_mean((Y_placeholder - Y_estimate)**2)

    # Create training operation
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
