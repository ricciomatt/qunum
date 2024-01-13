import numpy as np 
import matplotlib.pyplot as plt

def confusion_matrix_general(yact, yhat, labels = None, cmap = 'coolwarm'):
    if(labels is None):
        labels = np.arange(yhat.shape[1])
    v = np.empty((labels.shape[0], labels.shape[0]), dtype=np.int32)
    extent = [0, labels.shape[0], 0, labels.shape[0]]
    size = labels.shape[0]
    # The normal figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    for i in range(yact.shape[1]):
        for j in range(yhat.shape[1]):
            v[i,j] = np.where((yhat[:,j] == 1) & (yact[:,i] == 1))[0].shape[0]
            


    im = ax.imshow(v, extent=extent, origin='lower', interpolation='None', cmap=cmap)

    # Add the text
    jump_x = (extent[0] - extent[1]) / (2.0 * size)
    jump_y = (extent[2] - extent[3]) / (2.0 * size)
    x_positions = np.linspace(start=extent[0]+1, stop=extent[1]+1, num=size, endpoint=False)
    y_positions = np.linspace(start=extent[2]+1, stop=extent[3]+1, num=size, endpoint=False)

    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = '{:.3e}'.format(v[y_index, x_index]/yact[:, y_index].sum())
            text_x = x + jump_x
            text_y = y + jump_y
            ax.text(text_x, text_y, label, color='black', ha='center', va='center')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    fig.colorbar(im)
    plt.show()
    return