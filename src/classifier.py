import numpy as np
import tensorflow as tf


# ---CONFIGURATION PARAMETERS--- #

# Flag for verbose printing.
VERBOSE = True

# Paths of the datasets and the predictions.
DATA_PATH = "..\\data\\ocr\\"
OUT_PATH = "..\\out\\"

# Number of considered examples, for debug reasons. Use "None" to take the whole dataset.
TRAIN_SIZE = None
TEST_SIZE = None


# ---GLOBAL VARIABLES--- #

# Placeholders for the input and the output.
x = tf.placeholder(tf.float32, [None, 128])
y = tf.placeholder(tf.float32, [None, 26])

# Placeholder for the dropout probability.
keep_prob = tf.placeholder(tf.float32)


# ---FUNCTIONS--- #

def printv(text):
    """Prints additional information if the VERBOSE flag is active."""

    if VERBOSE:
        print("[", text, "]")


def load_data():
    """Loads the input data and the output targets from the specified path."""

    print("")
    print("*** DATA LOADING ***")

    # Load the input data.
    printv("Loading the input data...")
    X_train = np.genfromtxt(DATA_PATH + "train-data.csv", delimiter=",")[:TRAIN_SIZE]
    X_test = np.genfromtxt(DATA_PATH + "test-data.csv", delimiter=",")[:TEST_SIZE]

    # Load the output targets.
    printv("Loading the output targets...")
    y_train = np.genfromtxt(DATA_PATH + "train-target.csv", dtype="str")[:TRAIN_SIZE]
    y_test = np.genfromtxt(DATA_PATH + "test-target.csv", dtype="str")[:TEST_SIZE]

    print("Number of training examples: ", len(X_train))
    print("Number of test examples: ", len(X_test))

    return X_train, X_test, y_train, y_test


def weight_variable(shape):
    """Initializes a weight variable by sampling from a truncated normal distribution with standard deviation of 0.1."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Initializes a bias variable with a constant value 0.1"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """Defines a 2D convolution."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', use_cudnn_on_gpu=True)


def max_pool_2x2(x):
    """Defines a 2x2 pooling."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def declaration():
    """Declares the network architecture."""

    # Reshape the image from vector to matrix.
    x_image = tf.reshape(x, [-1, 16, 8, 1])

    # First convolution/pooling: [16x8x1] -> [8x4x32]
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolution/pooling: [8x4x32] -> [4x2x64]
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer: [4x2x64] -> [1024]
    W_fc1 = weight_variable([4 * 2 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 4*2*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Output layer: [1024] -> [26]
    W_fc2 = weight_variable([1024, 26])
    b_fc2 = bias_variable([26])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_hat = tf.nn.softmax(y_conv)

    # Use the cross entropy as loss function.
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_hat), reduction_indices=[1]))

    # Define the optimization method.
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Define the accuracy measure.
    predicted_y = tf.argmax(y_hat, 1)
    real_y = tf.argmax(y, 1)
    correct_prediction = tf.equal(predicted_y, real_y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Declare the session.
    sess = tf.InteractiveSession()

    # Run the global variables initializer.
    tf.global_variables_initializer().run()

    return sess, accuracy, train_step


# ---MAIN--- #

def main():
    """Main function."""

    # Load the data.
    X_train, X_test, y_train, y_test = load_data()

    # Declare the network architecture.
    sess, accuracy, train_step = declaration()

    # Save the predictions to file.
    # np.savetxt(OUT_PATH + "test-pred.txt", y_pred, fmt='%s', delimiter='\n')


# Start the program.
main()
