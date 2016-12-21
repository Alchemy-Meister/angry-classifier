import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools

class EpochDrawer(object):
    '''
    Based on code made by :Philipp Schmidt https://github.com/tdhd/keras/blob/514cd0c3ed159b4f87c32defcf2542e6d95a5f7f/keras/utils/drawing_utils.py
    takes the history of keras.models.Sequential().fit() and
    plots training and validation loss over the epochs
    '''
    def __init__(self, history, save_filename = None):
        self.x = history.epoch
        self.legend = ['loss']
        plt.plot(self.x, history.history['loss'], marker='.')
        if 'val_loss' in history.history:
            self.legend.append('val loss')
            plt.plot(self.x, history.history['val_loss'], marker='.')
        plt.title('Loss over epochs')
        plt.xlabel('Epochs')
        plt.xticks(history.epoch, history.epoch)
        #plt.xticks(np.arange(min(x), max(x)+1, 1.0))
        plt.legend(self.legend, loc = 'upper right')
        if save_filename is not None:
            plt.savefig(save_filename)
        #plt.show()


class ConfusionMatrixDrawer(object):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    def __init__(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        filename = ""
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            filename = "confusion_matrix_normalized"
            print("Normalized confusion matrix")
        else:
            filename = "confusion_matrix_without_normalization"
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, float("{0:.2f}".format(cm[i, j])),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('./' + filename + '.png')



if __name__ == '__main__':
    cnf_matrix =np.array([[ 328 , 1  , 23 ,  50 ,  39  ,  6  ,  7],[  18  ,463,   93,   31,   75,   36,    6],[  11,   26, 1286,  159,  158,   27,   18],[  34   ,17,  113, 5583,  579,   21,  225],[  29   ,37,  126,  466, 6486,   68,  122],[  16,   25,   47,   91,  211,  792,   20],[  19,   12,   64,  332,  367,   21, 1578]])
    classes = [1,2,3,4,5,6,7]
    ConfusionMatrixDrawer(cnf_matrix, classes)