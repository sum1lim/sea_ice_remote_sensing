import sys
import csv
import yaml
import numpy as np
import pandas
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from seaborn import heatmap
from imblearn.over_sampling import RandomOverSampler


def config_parser(dl_config):
    """
    Parse the parameters defined in the configuration part.
    """
    num_epochs, hidden_size, verbosity, K, kernel_size = 100, None, 2, 1, 5

    if dl_config:
        stream = open(dl_config, "r")
        config_dict = yaml.safe_load(stream)
    else:
        return num_epochs, hidden_size, verbosity, K

    if "configuration" in config_dict.keys():
        params = config_dict["configuration"]
        train_data = params["train_data"]
        test_data = params["test_data"]
        if "epochs" in params.keys():
            num_epochs = params["epochs"]
        if "hidden_size" in params.keys():
            hidden_size = params["hidden_size"]
        if "verbosity" in params.keys():
            verbosity = params["verbosity"]
        if "K-fold" in params.keys():
            K = params["K-fold"]
        if "kernel_size" in params.keys():
            kernel_size = params["kernel_size"]

    return train_data, test_data, num_epochs, hidden_size, verbosity, K, kernel_size


def claculate_hidden_layer_size(input_layer_size, output_layer_size, user_defined=None):
    """
    Calculate the hidden layer size if user did not define the size
    """
    if user_defined == None:
        hidden_layer_size = ((input_layer_size + output_layer_size) * 2) // 3
    else:
        hidden_layer_size = user_defined

    if hidden_layer_size > 2 * input_layer_size:
        hidden_layer_size = 2 * input_layer_size

    if hidden_layer_size < 2:
        hidden_layer_size = 2

    return hidden_layer_size


def process_data(data_file, dl_config=None):
    """
    Merge labels and/or select feautres for learning
    based on the user definition in the configuration file
    """
    if dl_config:
        stream = open(dl_config, "r")
        config_dict = yaml.safe_load(stream)
    else:
        config_dict = None

    dataframe = pandas.read_csv(data_file, header=0)

    if config_dict:
        if "labels" in config_dict.keys():
            try:
                for key, value in config_dict["labels"].items():
                    for i in value:
                        dataframe["label"].replace({i: key}, inplace=True)
            except KeyError:
                print("Error in configuration format", file=sys.stderr)
                sys.exit(1)

        if "features" in config_dict.keys():
            try:
                dataframe.drop(
                    dataframe.columns.difference(config_dict["features"] + ["label"]),
                    1,
                    inplace=True,
                )
            except KeyError:
                print("Error in configuration format", file=sys.stderr)
                sys.exit(1)

    dataset = dataframe.values
    X = dataset[:, 1:].astype(float)
    Y = dataset[:, 0]

    print(f"Before oversampling: {Counter(Y)}", file=sys.stdout)
    oversample = RandomOverSampler(sampling_strategy="not majority")
    X, Y = oversample.fit_resample(X, Y)
    print(f"After oversampling: {Counter(Y)}", file=sys.stdout)

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    return X, encoded_Y, encoder.classes_


def learning_curve(model_hist, result_dir, iter):
    """
    This chunk of code is sourced and modified from Machin Learning Mastery [1].

    [1] J. Brownlee, “Display Deep Learning Model Training History in Keras,” Machine Learning Mastery, 03-Oct-2019. [Online].
    Available: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/. [Accessed: 16-Jul-2021].
    """
    # summarize history for accuracy
    plt.plot(model_hist["accuracy"])
    plt.plot(model_hist["val_accuracy"])
    plt.title(f"Learning Curve (iteration: {iter+1})")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.savefig(f"{result_dir}/learning_curve_{iter}.png")
    plt.clf()
    # summarize history for loss
    plt.plot(model_hist["loss"])
    plt.plot(model_hist["val_loss"])
    plt.title(f"Loss Curve (iteration: {iter+1})")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.savefig(f"{result_dir}/loss_curve_{iter}.png")
    plt.clf()


def construct_confusion_matrix(classes, Y_te, y_pred, result_dir):
    """
    Construct the confusion matrix and output the results
    """
    np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=1, suppress=True)
    print(np.asarray([classes]), file=sys.stdout)
    cm_percentage = (
        confusion_matrix(Y_te, y_pred, labels=np.unique(Y_te), normalize="true") * 100
    )
    cm_counts = confusion_matrix(Y_te, y_pred, labels=np.unique(Y_te))

    print(cm_percentage, file=sys.stdout)

    cm_csv = open(f"{result_dir}/confusion_matrix.csv", "w", newline="")
    cm_writer = csv.writer(cm_csv)
    cm_writer.writerow(np.insert(classes, 0, None, axis=0))
    for i, row in enumerate(cm_counts):
        cm_writer.writerow(np.insert(row, 0, classes[i], axis=0))

    heatmap(cm_percentage, vmin=0, vmax=100)
    plt.savefig(f"{result_dir}/heat_map.png")
    plt.clf()


def tr_val_split(K, X_tr, Y_tr):
    if K < 2:
        K = 2
    kfold = StratifiedKFold(n_splits=K, shuffle=False)
    tr_val_pairs = kfold.split(X_tr, Y_tr)

    return tr_val_pairs
