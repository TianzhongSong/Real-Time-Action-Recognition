import os
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt


def plot_history(history, result_dir, save_prefix):
    """
    plot training log
    :param history: training log
    :param result_dir: save path
    :param save_prefix: prefix for saved file
    :return:None
    """
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, '{}_model_accuracy.png'.format(save_prefix)))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, '{}_model_loss.png'.format(save_prefix)))
    plt.close()


def save_history(history, result_dir, save_prefix):
    """
    save training log to .txt file
    :param history: training log
    :param result_dir: save path
    :param save_prefix: prefix for saved file
    :return:None
    """
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, '{}_result.txt'.format(save_prefix)), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()
