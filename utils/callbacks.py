import keras.backend as K
from keras.callbacks import Callback, ModelCheckpoint
import yaml
import h5py
import numpy as np

class Step(Callback):

    def __init__(self, steps, learning_rates, verbose=0):
        self.steps = steps
        self.lr = learning_rates
        self.verbose = verbose

    def change_lr(self, new_lr):
        old_lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, new_lr)
        if self.verbose == 1:
            print('Learning rate is %g' %new_lr)

    def on_epoch_begin(self, epoch, logs={}):
        for i, step in enumerate(self.steps):
            if epoch < step:
                self.change_lr(self.lr[i])
                return
        self.change_lr(self.lr[i+1])

    def get_config(self):
        config = {'class': type(self).__name__,
                  'steps': self.steps,
                  'learning_rates': self.lr,
                  'verbose': self.verbose}
        return config

    @classmethod
    def from_config(cls, config):
        offset = config.get('epoch_offset', 0)
        steps = [step - offset for step in config['steps']]
        return cls(steps, config['learning_rates'],
                   verbose=config.get('verbose', 0))

class TriangularCLR(Callback):

    def __init__(self, learning_rates, half_cycle):
        self.lr = learning_rates
        self.hc = half_cycle

    def on_train_begin(self, logs={}):
        # Setup an iteration counter
        self.itr = -1

    def on_batch_begin(self, batch, logs={}):
        self.itr += 1
        cycle = 1 + self.itr/int(2*self.hc)
        x = self.itr - (2.*cycle - 1)*self.hc
        x /= self.hc
        new_lr = self.lr[0] + (self.lr[1] - self.lr[0])*(1 - abs(x))/cycle

        K.set_value(self.model.optimizer.lr, new_lr)


class MetaCheckpoint(ModelCheckpoint):
    """
    Checkpoints some training information with the model. This should enable
    resuming training and having training information on every checkpoint.
    Thanks to Roberto Estevao @robertomest - robertomest@poli.ufrj.br
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, training_args=None, meta=None):

        super(MetaCheckpoint, self).__init__(filepath, monitor='val_loss',
                                             verbose=0, save_best_only=False,
                                             save_weights_only=False,
                                             mode='auto', period=1)

        self.filepath = filepath
        self.meta = meta or {'epochs': []}

        if training_args:
            self.meta['training_args'] = training_args

    def on_train_begin(self, logs={}):
        super(MetaCheckpoint, self).on_train_begin(logs)

    def on_epoch_end(self, epoch, logs={}):
        super(MetaCheckpoint, self).on_epoch_end(epoch, logs)

        # Get statistics
        self.meta['epochs'].append(epoch)
        for k, v in logs.items():
            # Get default gets the value or sets (and gets) the default value
            self.meta.setdefault(k, []).append(v)

        # Save to file
        filepath = self.filepath.format(epoch=epoch, **logs)

        if self.epochs_since_last_save == 0:
            with h5py.File(filepath, 'r+') as f:
                meta_group = f.create_group('meta')
                meta_group.attrs['training_args'] = yaml.dump(
                    self.meta.get('training_args', '{}'))
                meta_group.create_dataset('epochs',
                                          data=np.array(self.meta['epochs']))
                for k in logs:
                    meta_group.create_dataset(k, data=np.array(self.meta[k]))