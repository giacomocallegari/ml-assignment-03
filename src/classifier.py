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


def char_to_one_hot(y):
    """Converts the class labels to one-hot encoded vectors."""

    # Initialize an empty matrix of the length of y.
    y_one_hot = np.empty((len(y), 26))

    for i in range(len(y)):
        # Find the ASCII code and shift it onto the [0, 25] range.
        index = ord(y[i]) - 97

        # Obtain the one-hot representation.
        vector = np.zeros(26)
        vector[index] = 1
        y_one_hot[i] = vector

    return y_one_hot


def index_to_char(y):
    """Converts the target indices to characters."""

    # Initialize an empty matrix of the length of y.
    y_char = np.empty(len(y), dtype="str")

    for i in range(len(y)):
        # Find the character.
        y_char[i] = chr(y[i] + 97)

    return y_char


def load_data():
    """Loads the input data and the output targets from the specified path."""

    print("")
    print("*** DATA LOADING ***")

    # Load the input data.
    printv("Loading the input data...")
    X_train = np.genfromtxt(DATA_PATH + "train-data.csv", delimiter=",", max_rows=TRAIN_SIZE)
    X_test = np.genfromtxt(DATA_PATH + "test-data.csv", delimiter=",", max_rows=TEST_SIZE)

    # Load the output targets.
    printv("Loading the output targets...")
    y_train = char_to_one_hot(np.genfromtxt(DATA_PATH + "train-target.csv", dtype="str", max_rows=TRAIN_SIZE))
    y_test = char_to_one_hot(np.genfromtxt(DATA_PATH + "test-target.csv", dtype="str", max_rows=TEST_SIZE))

    # Convert the targets to one-hot encoding.
    printv("Converting the targets to one-hot encoded vectors...")

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

    print("")
    print("*** DECLARATION ***")

    printv("Declaring the architecture...")

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

    printv("Declaring additional variables...")

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
    # tf.global_variables_initializer().run()

    return sess, accuracy, train_step, predicted_y


def gen_batch_rand(X, y, size):
    """Generates a random batch of the provided size."""

    # Sample random indices.
    idx = np.arange(0, len(X))
    np.random.shuffle(idx)
    idx = idx[:size]

    # Create the batches.
    X_batch = [X[i] for i in idx]
    y_batch = [y[i] for i in idx]

    return [X_batch, y_batch]


def gen_batch(X, y, size, index):
    """Generates a batch of the provided size, starting from the provided index."""

    # Compute the indices.
    start = size * index
    end = size * (index + 1)

    # Create the batches.
    X_batch = X[start:end]
    y_batch = y[start:end]

    return [X_batch, y_batch]


def train(X_train, y_train, sess, accuracy, train_step):
    """Fits the network to the training set."""

    print("")
    print("*** TRAINING ***")

    # Run the global variable initializer.
    sess.run(tf.global_variables_initializer())

    # Train the network.
    printv("Training the network...")
    for i in range(1000):
        # Generate the batch.
        batch = gen_batch_rand(X_train, y_train, 50)

        # Compute the training accuracy every 100 steps.
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
            print('step {}, training accuracy {}'.format(i, train_accuracy))

        # Run a training step.
        train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})


def test(X_test, y_test, sess, accuracy, predicted_y):
    """Evaluates the network on the test set and returns the predictions."""

    print("")
    print("*** TESTING ***")

    # Accuracy and predictions.
    accuracy_values = []
    predicted_targets = []

    # Test the network.
    printv("Testing the network...")
    iterations = -(len(X_test) // -100)
    for j in range(iterations):
        # Generate the batch.
        batch = gen_batch(X_test, y_test, 100, j)

        # Compute the accuracy and the predictions.
        b_acc, b_pred_y = sess.run([accuracy, predicted_y], feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
        accuracy_values.append(b_acc)
        predicted_targets.extend(b_pred_y)

    print('test accuracy {}'.format(np.mean(accuracy_values)))

    return index_to_char(predicted_targets)


# ---MAIN--- #

def main():
    """Main function."""

    # Load the data.
    X_train, X_test, y_train, y_test = load_data()

    # Declare the network architecture.
    sess, accuracy, train_step, predicted_y = declaration()

    # Train the network.
    train(X_train, y_train, sess, accuracy, train_step)

    # Test the network.
    predicted_targets = test(X_test, y_test, sess, accuracy, predicted_y)

    # Save the predictions to file.
    # np.savetxt(OUT_PATH + "test-pred.txt", y_pred, fmt='%s', delimiter='\n')


# Start the program.
main()
