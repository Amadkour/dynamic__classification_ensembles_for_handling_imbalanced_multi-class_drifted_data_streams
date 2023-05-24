
from strlearn.streams import StreamGenerator, ARFFParser


def realstreams():
    return {
        # "krkopt": ARFFParser("datasets/allclass.arff", n_chunks=50, chunk_size=250),
        # "krkopt2": ARFFParser("datasets/krkopt.arff", n_chunks=50, chunk_size=250),
        # "TCP": ARFFParser("datasets/TCP.arff", n_chunks=300, chunk_size=1000),
        # "abalone": ARFFParser("datasets/winequalitywhite/train.arff", n_chunks=20, chunk_size=100),
        # "poker-lsn-1-2vsAll": ARFFParser("datasets/poker-lsn-1-2vsAll-pruned.arff", n_chunks=100, chunk_size=250),
        "covtypeNorm-1-2vsAll": ARFFParser("datasets/covtypeNorm-1-2vsAll-pruned.arff", n_chunks=100, chunk_size=1000),

    }

def realstreams2():
    return {
        "covtypeNorm-1-2vsAll": ARFFParser("datasets/covtypeNorm-1-2vsAll-pruned.arff", n_chunks=265, chunk_size=1000),
        "poker-lsn-1-2vsAll": ARFFParser("datasets/poker-lsn-1-2vsAll-pruned.arff", n_chunks=359, chunk_size=1000),
    }

def moa_streams():
    return {
        # "gr_css5_rs804_nd1_ln1_d85_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln1_d85_50000.arff", n_chunks=200, chunk_size=250),
        # "gr_css5_rs804_nd1_ln1_d90_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln1_d90_50000.arff", n_chunks=200, chunk_size=250),
        # "gr_css5_rs804_nd1_ln1_d95_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln1_d95_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css5_rs804_nd1_ln1_d97_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln1_d97_50000.arff", n_chunks=200, chunk_size=250),
        # "gr_css5_rs804_nd1_ln3_d85_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln3_d85_50000.arff", n_chunks=200, chunk_size=250),
        # "gr_css5_rs804_nd1_ln3_d90_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln3_d90_50000.arff", n_chunks=200, chunk_size=250),
        # "gr_css5_rs804_nd1_ln3_d95_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln3_d95_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css5_rs804_nd1_ln3_d97_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln3_d97_50000.arff", n_chunks=200, chunk_size=250),
        # "gr_css5_rs804_nd1_ln5_d85_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln5_d85_50000.arff", n_chunks=200, chunk_size=250),
        # "gr_css5_rs804_nd1_ln5_d90_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln5_d90_50000.arff", n_chunks=200, chunk_size=250),
        # "gr_css5_rs804_nd1_ln5_d95_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln5_d95_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css5_rs804_nd1_ln5_d97_50000": ARFFParser("streams/gr_css5_rs804_nd1_ln5_d97_50000.arff", n_chunks=200, chunk_size=250),
        # "gr_css999_rs804_nd1_ln1_d85_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln1_d85_50000.arff", n_chunks=200, chunk_size=250),
        # "gr_css999_rs804_nd1_ln1_d90_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln1_d90_50000.arff", n_chunks=200, chunk_size=250),
        # "gr_css999_rs804_nd1_ln1_d95_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln1_d95_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css999_rs804_nd1_ln1_d97_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln1_d97_50000.arff", n_chunks=200, chunk_size=250),
        # "gr_css999_rs804_nd1_ln3_d85_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln3_d85_50000.arff", n_chunks=200, chunk_size=250),
        # "gr_css999_rs804_nd1_ln3_d90_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln3_d90_50000.arff", n_chunks=200, chunk_size=250),
        # "gr_css999_rs804_nd1_ln3_d95_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln3_d95_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css999_rs804_nd1_ln3_d97_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln3_d97_50000.arff", n_chunks=200, chunk_size=250),
        # "gr_css999_rs804_nd1_ln5_d85_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln5_d85_50000.arff", n_chunks=200, chunk_size=250),
        # "gr_css999_rs804_nd1_ln5_d90_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln5_d90_50000.arff", n_chunks=200, chunk_size=250),
        # "gr_css999_rs804_nd1_ln5_d95_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln5_d95_50000.arff", n_chunks=200, chunk_size=250),
        "gr_css999_rs804_nd1_ln5_d97_50000": ARFFParser("streams/gr_css999_rs804_nd1_ln5_d97_50000.arff", n_chunks=200, chunk_size=250),
        # "inc_css5_rs804_nd1_ln1_d85_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln1_d85_50000.arff", n_chunks=200, chunk_size=250),
        # "inc_css5_rs804_nd1_ln1_d90_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln1_d90_50000.arff", n_chunks=200, chunk_size=250),
        # "inc_css5_rs804_nd1_ln1_d95_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln1_d95_50000.arff", n_chunks=200, chunk_size=250),
        "inc_css5_rs804_nd1_ln1_d97_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln1_d97_50000.arff", n_chunks=200, chunk_size=250),
        # "inc_css5_rs804_nd1_ln3_d85_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln3_d85_50000.arff", n_chunks=200, chunk_size=250),
        # "inc_css5_rs804_nd1_ln3_d90_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln3_d90_50000.arff", n_chunks=200, chunk_size=250),
        # "inc_css5_rs804_nd1_ln3_d95_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln3_d95_50000.arff", n_chunks=200, chunk_size=250),
        "inc_css5_rs804_nd1_ln3_d97_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln3_d97_50000.arff", n_chunks=200, chunk_size=250),
        # "inc_css5_rs804_nd1_ln5_d85_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln5_d85_50000.arff", n_chunks=200, chunk_size=250),
        # "inc_css5_rs804_nd1_ln5_d90_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln5_d90_50000.arff", n_chunks=200, chunk_size=250),
        # "inc_css5_rs804_nd1_ln5_d95_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln5_d95_50000.arff", n_chunks=200, chunk_size=250),
        "inc_css5_rs804_nd1_ln5_d97_50000": ARFFParser("streams/inc_css5_rs804_nd1_ln5_d97_50000.arff", n_chunks=200, chunk_size=250),
    }


def toystreams(random_state):
    # Variables
    Multi_distributions = [[0.96, 0.01,0.01,0.01,0.01]]
    binary_distributions = [[0.97, 0.03]]
    distributions=Multi_distributions
    label_noises = [
        0.01,
        0.03,
        0.05,
    ]
    incremental = [(5,True),(5,False), (100,True)]
    n_drifts = 10

    # Prepare streams
    streams = {}
    for drift_type in incremental:
        for distribution in distributions:
            for flip_y in label_noises:
                    spacing ,type=drift_type
                    stream = StreamGenerator(
                        incremental=type,
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
                    if spacing is None and drift_type == True:
                        pass
                    else:
                        streams.update({str(stream): stream})
                    print(str(stream))

    return streams


def streams(random_state):
    # Variables
    # distributions = [[0.95, 0.05], [0.90, 0.10], [0.85, 0.15]]
    distributions = [[0.97, 0.03]]
    label_noises = [
        0.01,
        0.03,
        0.05,
    ]
    incremental = [False, True]
    ccs = [5, None]
    n_drifts = 1

    # Prepare streams
    streams = {}
    for drift_type in incremental:
        for distribution in distributions:
            for flip_y in label_noises:
                for spacing in ccs:
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
                    )
                    if spacing == None and drift_type == True:
                        pass
                    else:
                        streams.update({str(stream): stream})

    return streams


def timestream(chunk_size):
    # Variables
    distributions = [[0.80, 0.20]]
    label_noises = [
        0.01,
    ]
    incremental = [False]
    ccs = [None]
    n_drifts = 1

    # Prepare streams
    streams = {}
    for drift_type in incremental:
        for distribution in distributions:
            for flip_y in label_noises:
                for spacing in ccs:
                    stream = StreamGenerator(
                        incremental=drift_type,
                        weights=distribution,
                        random_state=1994,
                        y_flip=flip_y,
                        concept_sigmoid_spacing=spacing,
                        n_drifts=n_drifts,
                        chunk_size=chunk_size,
                        n_chunks=2,
                        n_clusters_per_class=1,
                        n_features=8,
                        n_informative=8,
                        n_redundant=0,
                        n_repeated=0,
                    )
                    if spacing == None and drift_type == True:
                        pass
                    else:
                        streams.update({str(stream): stream})

    return streams
toystreams(100)