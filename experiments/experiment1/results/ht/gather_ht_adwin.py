from strlearn.streams import StreamGenerator
import numpy as np

# Variables
clfs = [
    "HT",

]
methods = [
    "HT-KNORAU1", "HT-KNORAU1", "HT-KNORAU1","HT-KNORAU1",
]
random_states = [100]
Multi_distributions = [[0.96, 0.01, 0.01, 0.01, 0.01]]
binary_distributions = [[0.97, 0.03]]
distributions=binary_distributions
label_noises = [
    0.01,
    0.03,
    0.05,
]
drifttype = [(5,True),(5,False), (100,True)]
n_drifts = 10
n_chunks = 199
metrics = ["BAC", "geometric_mean_score", "f_score", "precision", "recall", "specificity"]

scores = np.zeros(
    (
        len(clfs),
        len(random_states),
        len(drifttype),
        len(distributions),
        len(label_noises),
        len(methods),
        n_chunks,
        len(metrics),
    )
)

print(scores.shape)

# Prepare streams
streams = {}
name='krkopt'
scores = np.load("experiments/experiment2/results_%s.npy" % name)

scores_metrics_22 = scores
np.save("scores_exp_1_matrix_ht_adwin", scores_metrics_22)
scores = np.mean(scores_metrics_22, axis=1)
print(scores.shape)
np.save("scores_exp_1_ht_adwin", scores)
