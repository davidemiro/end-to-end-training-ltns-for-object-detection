from keras.callbacks import ModelCheckpoint


class TrainingCheckpoint(ModelCheckpoint):
    """ Customized version of keras.callbacks.ModelCheckpoint
        Adds support for model backups while also saving the best model.
    """

    def __init__(self, filepath, backup_freq, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        """ TrainingCheckpoint callback initializer.

           Args
                filepath     : string, path to save the model file.
                backup_freq  : Interval (number of epochs) between checkpoints.
                monitor: quantity to monitor.
                verbose: verbosity mode, 0 or 1.
                save_best_only: if `save_best_only=True`, The latest best model according to the quantity monitored
                    will not be overwritten.
                save_weights_only: if True, then only the model's weights wi    ll be saved
                    (`model.save_weights(filepath)`), else the full model is saved (`model.save(filepath)`).
                mode: one of {auto, min, max}. If `save_best_only=True`, the decision to overwrite the current save
                    file is made based on either the maximization or the minimization of the monitored quantity.
                    For `val_acc`, this should be `max`, for `val_loss` this should be `min`, etc.
                    In `auto` mode, the direction is automatically inferred from the name of the monitored quantity.
                period: Interval (number of epochs) between checkpoints.
        """
        super(TrainingCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode,
                                                 period)
        self.backup_freq = backup_freq
        self.epochs_since_last_backup = 0
        self.backup_filepath = self.filepath[:len(self.filepath) - 3] + '_bkup.h5'  # backup suffix

    def on_epoch_end(self, epoch, logs=None):
        ModelCheckpoint.on_epoch_end(self, epoch, logs)
        logs = logs or {}
        self.epochs_since_last_backup += 1
        if self.epochs_since_last_backup >= self.backup_freq:
            self.epochs_since_last_backup = 0
            filepath = self.backup_filepath.format(epoch=epoch + 1, **logs)
            if self.verbose > 0:
                print('\nEpoch %05d: saving model backup to %s' % (epoch + 1, filepath))
            if self.save_weights_only:
                self.model.save_weights(filepath, overwrite=True)
            else:
                self.model.save(filepath, overwrite=True)
