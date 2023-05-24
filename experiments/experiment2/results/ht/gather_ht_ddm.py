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
for i, clf in enumerate(clfs):
    for j, random_state in enumerate(random_states):
        for k, kurwa in enumerate(drifttype):
            for l, distribution in enumerate(distributions):
                for m, flip_y in enumerate(label_noises):
                    # for n, spacing in enumerate(css):
                    spacing, drift_type = kurwa
                    stream = StreamGenerator(
                        incremental=drift_type,
                        weights=distribution,
                        random_state=random_state,
                        y_flip=flip_y,
                        concept_sigmoid_spacing=spacing,
                        n_drifts=n_drifts,
                        chunk_size=250,
                        n_chunks=200,
                        n_clusters_per_class=1,
                        n_features=8,
                        n_informative=8,
                        n_redundant=0,
                        n_repeated=0,
                        n_classes=5
                    )
                    if spacing == None and drift_type == True:
                        pass
                    else:
                        print(stream)
                        results = np.load(
                            # "results/experiment3_%s/%s.npy" % (clf, stream)
                            "experiments/experiment1/results/ht/DDM/%s" % str(stream).replace('d97','d96') + '.npy'
                        )

                        scores[i, j, k, l, m] = results
scores_metrics_22 = scores
np.save("scores_exp_1_matrix_ht_ddm", scores_metrics_22)
scores = np.mean(scores_metrics_22, axis=1)
print(scores.shape)
np.save("scores_exp_1_ht_ddm", scores)
