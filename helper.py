from strlearn.streams import ARFFParser, StreamGenerator


def realstreams():
    dataset={
        "covertype": ARFFParser("X:\\Ahmed\\faculty\\phd_implementation\\paper+2+ISA\\code\\dataset\\covtype.arff",
                                n_chunks=100, chunk_size=2000),

    }
    return  dataset
def realstreams2():
    dataset={
        "sensors": ARFFParser("X:\\Ahmed\\faculty\\phd_implementation\\paper+2+ISA\\code\\dataset\\sensors.arff",
                                n_chunks=100, chunk_size=2000),

    }
    return  dataset


def toystreams(random_state):
    # Variables
    Multi_distributions = [[0.96, 0.01,0.01,0.01,0.01]]
    binary_distributions = [[0.97, 0.03]]
    distributions=Multi_distributions
    label_noises = [

        0.05,
        0.01,
        0.03,
    ]
    incremental = [(5,True),(5,False), (100,True)]
    n_drifts = 80

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
                        n_chunks=100,
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


def synthetic_streams(random_state):
    # Variables
    distribution= [0.77, 0.03,0.22]
    n_drifts = 30

    # Prepare streams
    streams = {}

    stream = StreamGenerator(
        incremental=False,
        weights=distribution,
        n_classes=3,
        random_state=random_state,
        y_flip=0.05,
        concept_sigmoid_spacing=5,
        n_drifts= 30,
        chunk_size=2000,
        n_chunks=100,
        n_clusters_per_class=1,
        n_features=8,
        n_informative=8,
        n_redundant=0,
        n_repeated=0,
    )

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