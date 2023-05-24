from strlearn.streams import StreamGenerator
import numpy as np

# Variables
clfs = [
    "HT",

]
methods = [
    "HT-KNORAU1", "HT-KNORAU1", "HT-KNORAU1", "HT-KNORAU1",
]

n_chunks = 19
metrics = ["BAC", "geometric_mean_score", "f_score", "precision", "recall", "specificity"]
data = ['krkopt']
scores = np.zeros(
    (
        len(clfs),
        len(data),
        n_chunks,
        len(methods),
        len(metrics),
    )
)

print(scores.shape)

# Prepare streams
streams = {}
for i,d in enumerate(data):
    for j in range(len(clfs)):
        results = np.load('experiments/experiment2/results_%s.npy' % d)
        scores = results
        print(results.shape)
        print(scores[i].shape)
        scores[i] = results[j]
scores_metrics_22 = scores
np.save("experiments/experiment2/results/ht/ADWIN/matrix", scores_metrics_22)
scores = np.mean(scores_metrics_22, axis=1)
print(scores.shape)
np.save("experiments/experiment2/results/ht/ADWIN/score_svm_adwin", scores)