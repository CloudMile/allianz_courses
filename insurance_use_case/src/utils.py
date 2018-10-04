from matplotlib import pyplot as plt
import numpy as np

def plot_result(hist):
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 2, 1)
    plt.plot(hist.history['loss'], label='tr_loss')
    plt.plot(hist.history['val_loss'], label='vl_loss')
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(hist.history['acc'], label='acc')
    plt.plot(hist.history['val_acc'], label='val_acc')
    plt.title('Accuracy')

    plt.legend(loc='best')
    plt.show()


def woe_encode(x, label, data):
    """Calculate the Weight of Evidence of given categorical feature and label

    :param x: Given feature name
    :param label: Label name
    :param data:
    :return: Woe encoded dictionary
    """
    total_vc = data[label].value_counts().sort_index()

    def woe(pipe, total_vc):
        # Count by label in this group
        group_vc = pipe[label].value_counts().sort_index()

        # Some class in the feature is missing, fill zero to missing class
        if len(group_vc) < len(total_vc):
            for key in total_vc.index:
                if key not in group_vc:
                    group_vc[key] = 0.
            group_vc = group_vc.sort_index()

        # WOE formula
        r = ((group_vc + 0.5) / total_vc).values

        # Odd ratio => 1 to 0, you can define meaning of each class
        return np.log(r[1] / r[0])

    return data.groupby(x).apply(lambda pipe: woe(pipe, total_vc))