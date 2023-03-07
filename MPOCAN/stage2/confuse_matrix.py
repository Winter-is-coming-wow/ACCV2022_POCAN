# -*-coding:utf-8-*-
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# labels表示你不同类别的代号，比如这里的demo中有13个类别
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']

tick_marks = np.array(range(len(labels))) + 0.5


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm1 = [[0.45348837, 0.51162791, 0., 0., 0., 0.,
        0.03488372],
       [0.10843373, 0.89156627, 0., 0., 0., 0.,
        0.],
       [0.06896552, 0., 0.75862069, 0., 0., 0.11494253,
        0.05747126],
       [0.01401869, 0., 0., 0.87383178, 0., 0.02336449,
        0.08878505],
       [0.216, 0., 0.024, 0., 0.584, 0.,
        0.176],
       [0., 0., 0., 0., 0., 1.,
        0., ],
       [0., 0., 0., 0., 0.01702128, 0.00851064,
        0.97446809]]

cm = [[0.32163743, 0.60818713, 0.06432749, 0., 0., 0.00584795,
       0.],
      [0.04444444, 0.95555556, 0., 0., 0., 0.,
       0.],
      [0., 0., 0.92857143, 0., 0., 0.07142857,
       0., ],
      [0.01869159, 0.02803738, 0.03271028, 0.87383178, 0., 0.,
       0.04672897],
      [0.25899281, 0.02158273, 0.18705036, 0., 0.49640288, 0.,
       0.03597122],
      [0., 0., 0.04888889, 0., 0., 0.95111111,
       0.],
      [0.08201893, 0.01892744, 0.00315457, 0., 0.02523659, 0.03154574,
       0.83911672]]

cm = np.asarray(cm)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# show confusion matrix
plt.savefig('confusion_matrix0.png', format='png')
plt.show()
