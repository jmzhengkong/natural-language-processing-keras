"""Trains a neural network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import argparse
import functools
import os
import sys
import numpy as np
import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
from keras.utils import plot_model


def main():
    checkpoint = ModelCheckpoint(
        FLAGS.model_dir,
        monitor=FLAGS.monitor,
        verbose=1,
        save_best_only=True,
        mode='auto')
    early_stopping = EarlyStopping(
        monitor=FLAGS.monitor, patience=8, verbose='auto', mode='auto')
    reduce_lr = ReduceLROnPlateau(
        monitor=FLAGS.monitor, factor=0.8, patience=5, min_lr=0.00001)
    callbacks_list = [checkpoint, early_stopping, reduce_lr]
    callbacks_list = [checkpoint]

    data = np.load(FLAGS.data_dir)
    label_dims = 3
    maxlen = data["maxlen"]
    X_train, X_test, y_train, y_test = train_test_split(
        data["X"], data["y"], test_size=0.2, random_state=8848)
    class_weight = {
        0: len(y_train) / y_train.sum(axis=0)[0],
        1: len(y_train) / y_train.sum(axis=0)[1],
        2: len(y_train) / y_train.sum(axis=0)[2]}

    if FLAGS.model == "cnn":
        model = models.cnn(FLAGS.hparam, data["W"], label_dims, maxlen)
    elif FLAGS.model == "rnn":
        model = models.rnn(FLAGS.hparam, data["W"], label_dims, maxlen)

    if FLAGS.retrain:
        model.fit(
            X_train,
            y_train,
            batch_size=FLAGS.batch_size,
            epochs=FLAGS.epoch,
            validation_data=(X_test, y_test),
            callbacks=callbacks_list,
            class_weight=class_weight)
    else:
        model.load_weights(FLAGS.model_dir)

    print("Evaluation on training set")
    y_pred = model.predict(X_train, batch_size=FLAGS.batch_size)
    y_true = np.argmax(y_train, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, y_pred).astype(np.float32)
    for i in range(cm.shape[0]):
        cm[i] = cm[i] / cm[i].sum()
    print(cm)
    print(classification_report(y_true, y_pred))
    print("Accuracy: {:.4f}".format((y_true == y_pred).sum()/y_true.size))
    dataframe = pd.DataFrame({"ground-truth": y_true, "prediction": y_pred})
    dataframe.to_csv("{:s}.train.csv".format(FLAGS.model))

    print("Evaluation on test set")
    y_pred = model.predict(X_test, batch_size=FLAGS.batch_size)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, y_pred).astype(np.float32)
    for i in range(cm.shape[0]):
        cm[i] = cm[i] / cm[i].sum()
    print(cm)
    print(classification_report(y_true, y_pred))
    print("Accuracy: {:.4f}".format((y_true == y_pred).sum()/y_true.size))
    dataframe = pd.DataFrame({"ground-truth": y_true, "prediction": y_pred})
    dataframe.to_csv("{:s}.test.csv".format(FLAGS.model))

    print("Evaluation on overall dataset")
    y_pred = model.predict(data["X"], batch_size=FLAGS.batch_size)
    y_true = np.argmax(data["y"], axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_true, y_pred).astype(np.float32)
    for i in range(cm.shape[0]):
        cm[i] = cm[i] / cm[i].sum()
    print(cm)
    print(classification_report(y_true, y_pred))
    print("Accuracy: {:.4f}".format((y_true == y_pred).sum()/y_true.size))

    dataframe = pd.DataFrame({"ground-truth": y_true, "prediction": y_pred})
    dataframe.to_csv("{:s}.all.csv".format(FLAGS.model))

    plot_model(model, to_file="{:s}.png".format(FLAGS.model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hparam",
        type=str,
        default="cnn.json",
        help="The JSON file of hyper parameters.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size.")
    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="The number of training epoch.")
    parser.add_argument(
        "--monitor",
        type=str,
        default="val_loss",
        help="The type of monitor.")
    parser.add_argument(
        "--gpu",
        type=str,
        default="1",
        help="The gpu used for training.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="The directory to save models.")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        help="The model to train.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="The path to training data.")
    parser.add_argument(
        "--retrain",
        dest="retrain",
        action="store_true",
        help="Re-train the neural network.")
    parser.set_defaults(retrain=False)
    FLAGS, unparsed = parser.parse_known_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    main()
