import argparse
import configparser
import os
import logging
from pathlib import Path
import time

from configobj import ConfigObj
from sklearn import metrics
import numpy as np
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import sktime
import pandas as pd

from utils.program_stats import timeit
from utils.sklearn_utils import report_average, plot_confusion_matrix
from utils.util_functions import create_directory_if_not_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FILE_NAME_X = '{}_{}_X'
FILE_NAME_Y = '{}_{}_Y'
FILE_NAME_PID = '{}_{}_pid'


def read_dataset(path):
    x_train, y_train = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                  FILE_NAME_X.format("TRAIN", data_type) + ".ts"))

    logger.info("Training data shape {} {} {}".format(x_train.shape, len(x_train.iloc[0, 0]), y_train.shape))
    x_test, y_test = load_from_tsfile_to_dataframe(os.path.join(path,
                                                                FILE_NAME_X.format("TEST", data_type) + ".ts"))
    logger.info("Testing data shape: {} {}".format(x_test.shape, y_test.shape))

    logger.info("Testing data shape: {} {}".format(x_test.shape, y_test.shape))
    test_pid = np.load(os.path.join(path, FILE_NAME_PID.format("TEST", data_type) + ".npy"), allow_pickle=True)
    train_pid = np.load(os.path.join(path, FILE_NAME_PID.format("TRAIN", data_type) + ".npy"), allow_pickle=True)

    try:
        x_val, y_val = load_from_tsfile_to_dataframe(os.path.join(input_data_path,
                                                                  FILE_NAME_X.format("VAL", data_type) + ".ts"))
        logger.info("Validation data shape: {} {}".format(x_val.shape, y_val.shape))
    except (sktime.utils.data_io.TsFileParseException, FileNotFoundError):
        logger.info("Validation data is empty:")
        x_val, y_val = None, None

    return x_train, y_train, x_test, y_test, x_val, y_val, train_pid, test_pid


class RocketTransformerClassifier:
    def __init__(self, exercise):
        self.exercise = exercise
        self.classifiers_mapping = {}

    @timeit
    def fit_rocket(self, x_train, y_train, train_pid, kernels=10000):
        rocket = Rocket(num_kernels=kernels, normalise=False)  # random_state=100343
        rocket.fit(x_train)
        x_training_transform = rocket.transform(x_train)
        self.classifiers_mapping["transformer"] = rocket
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        classifier.fit(x_training_transform, y_train)

        # Training Predictions
        # predictions = classifier.predict(x_training_transform)
        # d = classifier.decision_function(x_training_transform)
        # probs = np.exp(d) / np.sum(np.exp(d), axis=1).reshape(-1, 1)
        self.classifiers_mapping["classifier"] = classifier

    @timeit
    def predict_rocket(self, x_test, y_test, test_pid, x_val=None, y_val=None):
        rocket = self.classifiers_mapping["transformer"]
        classifier = self.classifiers_mapping["classifier"]
        x_test_transform = rocket.transform(x_test)

        # Test Predictions
        predictions = classifier.predict(x_test_transform)
        # d = self.classifiers_mapping["classifier"].decision_function(x_test_transform)
        # probs = np.exp(d) / np.sum(np.exp(d), axis=1).reshape(-1, 1)

        # Confusion Matrix
        labels = list(np.sort(np.unique(y_test)))
        confusion_matrix = metrics.confusion_matrix(y_test, predictions)
        classification_report = metrics.classification_report(y_test, predictions)
        logger.info("-----------------------------------------------")
        logger.info("Metrics on testing data")
        logger.info("Accuracy {}".format(metrics.accuracy_score(y_test, predictions)))
        logger.info("\n Confusion Matrix: \n {}".format(confusion_matrix))
        logger.info("\n Classification report: \n{}".format(classification_report))

        classification_report_list.append(classification_report)

        plot_confusion_matrix(output_results_path, seed_value, confusion_matrix, labels)

        if x_val:
            logger.info("-----------------------------------------------")
            logger.info("Metrics on validation data")
            x_val_transform = rocket.transform(x_val)
            predictions = classifier.predict(x_val_transform)
            confusion_matrix = metrics.confusion_matrix(y_val, predictions)
            classification_report = metrics.classification_report(y_val, predictions)
            logger.info("Accuracy {}".format(metrics.accuracy_score(y_test, predictions)))
            logger.info("\n Confusion Matrix: \n {}".format(confusion_matrix))
            logger.info("\n Classification report: \n{}".format(classification_report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rocket_config", required=True, help="path of the config file")
    args = parser.parse_args()
    rocket_config = ConfigObj(args.rocket_config)

    # home_path = str(Path.home())
    base_path = os.path.dirname(os.getcwd())
    # base_path = os.path.join(home_path, rocket_config["BASE_PATH"])
    seed_values = rocket_config["SEED_VALUES"]
    input_data_path = os.path.join(base_path, rocket_config["INPUT_DATA_PATH"])
    exercise = rocket_config["EXERCISE"]
    output_path = rocket_config["OUTPUT_PATH"]
    data_type = rocket_config["DATA_TYPE"]
    config_parser = configparser.RawConfigParser()

    output_results_path = os.path.join(output_path, "Rocket")
    create_directory_if_not_exists(output_results_path)

    classification_report_list = []
    for seed_value in seed_values:
        logger.info("----------------------------------------------------")
        logger.info("Fitting Rocket for seed value: {}".format(seed_value))
        input_path_combined = os.path.join(input_data_path, exercise, seed_value)
        if not os.path.exists(input_path_combined):
            logger.info("Path does not exist for seed: {}".format(seed_value))
            continue
        x_train, y_train, x_test, y_test, x_val, y_val, train_pid, test_pid = read_dataset(input_path_combined)
        ts = time.time()
        rocket_classifier = RocketTransformerClassifier(exercise)
        rocket_classifier.fit_rocket(x_train, y_train, train_pid)
        rocket_classifier.predict_rocket(x_test, y_test, test_pid, x_val, y_val)
        te = time.time()
        total_time = (te - ts)
        logger.info('Total time preprocessing: {} seconds'.format(total_time))

    logger.info("Average classification report")
    logger.info(report_average(*classification_report_list))
