import numpy as np


# ---CONFIGURATION PARAMETERS--- #

# Flag for verbose printing.
VERBOSE = True

# Paths of the datasets and the predictions.
DATA_PATH = "..\\data\\ocr\\"
OUT_PATH = "..\\out\\"

# Number of considered examples, for debug reasons. Use "None" to take the whole dataset.
TRAIN_SIZE = None
TEST_SIZE = None


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


# ---MAIN--- #

def main():
    """Main function."""

    # Load the data.
    X_train, X_test, y_train, y_test = load_data()

    # ...

    # Save the predictions to file.
    # np.savetxt(OUT_PATH + "test-pred.txt", y_pred, fmt='%s', delimiter='\n')


# Start the program.
main()
