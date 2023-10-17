from matplotlib import pyplot as plt
import numpy as np

img_array = np.load('experiments/experiment1/results-adwin/ht/DDM/plots/1_HT_0-01.eps',allow_pickle=True)
plt.imshow(img_array, cmap='gray')
plt.show()