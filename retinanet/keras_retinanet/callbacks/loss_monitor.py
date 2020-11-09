import csv

import keras
import matplotlib.pyplot as plt


class LossMonitor(keras.callbacks.Callback):
    """ Saves training and validation loss(es) plot(s) on train end.
    """

    def __init__(self, labdir_path, output_name="losses_plot.png", save_logs=True):
        """ LossMonitor callback initializer.

           Args
               labdir_path  : directory path to which the monitor results are saved
               output_name  : filename of plot
               save_logs    : True in order to save log data in csv
        """
        super().__init__()
        self.labdir_path = labdir_path
        self.output_name = output_name
        self.save_logs = save_logs

        # init epoch count from 0 (used as x axis in plot)
        self.i = 0
        # init arrays
        self.x_epoch = []
        self.logs = []
        self.val_losses = []
        self.losses = []
        self.regression_losses = []
        self.classification_losses = []

        self.fig = plt.figure()

        if self.save_logs:
            loss_name = ['losses', 'classification_losses', 'regression_losses', 'val_losses']
            for i, loss in enumerate(
                    [self.losses, self.classification_losses, self.regression_losses, self.val_losses]):
                with open(self.labdir_path + '/' + loss_name[i], 'a', newline='') as csv_log:
                    csv_log_file = csv.writer(csv_log, delimiter=',')
                    csv_log_file.writerow([loss_name[i]])

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x_epoch.append(self.i)
        self.losses.append(logs.get('loss'))
        self.regression_losses.append(logs.get('regression_loss'))
        self.classification_losses.append(logs.get('classification_loss'))
        self.val_losses.append(logs.get('val_loss'))
        # print(logs)

        if self.save_logs:
            loss_name = ['losses', 'classification_losses', 'regression_losses', 'val_losses']
            for li, loss in enumerate(
                    [self.losses, self.classification_losses, self.regression_losses, self.val_losses]):
                with open(self.labdir_path + '/' + loss_name[li], 'a', newline='') as csv_log:
                    csv_log_file = csv.writer(csv_log, delimiter=',')
                    csv_log_file.writerow([loss[self.i]])

        self.i += 1

    def on_train_end(self, logs=None):
        if logs is None:
            logs = {}

        # save plot data to csv for future usage
        # if self.save_logs:
        #     loss_name = ['losses', 'classification_losses', 'regression_losses', 'val_losses']
        #     for i, loss in enumerate(
        #             [self.losses, self.classification_losses, self.regression_losses, self.val_losses]):
        #         with open(self.labdir_path + '/' + loss_name[i], 'w', newline='') as csv_log:
        #             csv_log_file = csv.writer(csv_log, delimiter=',')
        #             csv_log_file.writerow([loss_name[i]])
        #             for l_i in loss:
        #                 csv_log_file.writerow([l_i])

        # create and save plot img
        plt.plot(self.x_epoch, self.losses, 'C0', label="loss")
        plt.plot(self.x_epoch, self.classification_losses, 'C5--', label="classification_loss")
        plt.plot(self.x_epoch, self.regression_losses, 'C7--', label="regression_loss")
        plt.plot(self.x_epoch, self.val_losses, 'C1', label="val_loss")
        plt.legend(loc='upper right')
        # plt.show()
        plt.savefig(self.labdir_path + '/' + self.output_name)
