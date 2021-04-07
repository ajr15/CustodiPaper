from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import settings
import deepchem as dc
import numpy as np
import os
from dask import delayed, compute, dataframe
from dask.distributed import Client
from rdkit import Chem
import pandas as pd
from time import time
import sys; sys.path.append(settings.torina_parent_dir)
from Torina.Model.Model import Model
from Torina.Model.utils import grid_estimation
from commons import *
import sys

# wrapper around DeepChemModel to make a good comparision with other results
class GraphConv (Model):
    
    """Wrapper around deepchem.models.GraphConvModel. to allow for Torina.Model functionalities
    ARGS:
        - graph_conv_layers
        - dense_layer_size"""

    def __init__(self, graph_conv_layers=[64, 64], dense_layer_size=128):
        self.model = dc.models.GraphConvModel(n_tasks=1, 
                                                graph_conv_layers=graph_conv_layers, 
                                                dense_layer_size=dense_layer_size, 
                                                mode='regression')

    def train(self, x, y):
        dataset = dc.data.DiskDataset.from_numpy(x, y=y, ids=range(len(x)))
        self.model.fit(dataset)

    def predict(self, X):
        return self.model.predict_on_batch(X)

class DeepTensorNN (Model):
    
    """Wrapper around deepchem.models.DTNNModel. to allow for Torina.Model functionalities.
    Model hyperparameters are teken from https://github.com/deepchem/deepchem/blob/master/examples/qm9/qm9_DTNN.py"""

    def __init__(self):
        # self.model = dc.models.DTNNModel(n_tasks=1, 
        #                                     batch_size=50,
        #                                     n_embedding=20,
        #                                     n_distance=51,
        #                                     distance_min=-1.,
        #                                     distance_max=9.2,
        #                                     n_hidden=15,
        #                                     mode='regression')
        self.model = dc.models.DTNNModel(1)

    def train(self, x, y):
        dataset = dc.data.DiskDataset.from_numpy(x, y=y, ids=range(len(x)))
        self.model.fit(dataset)

    def predict(self, X):
        return self.model.predict_on_batch(X)

class MessagePassingNN (Model):
    
    """Wrapper around deepchem.models.DTNNModel. to allow for Torina.Model functionalities.
    Model hyperparameters are teken from https://github.com/deepchem/deepchem/blob/master/examples/qm9/qm9_DTNN.py"""

    def __init__(self):
        self.model = dc.models.MPNNModel(1)

    def train(self, x, y):
        dataset = dc.data.DiskDataset.from_numpy(x, y=y, ids=range(len(x)))
        self.model.fit(dataset)

    def predict(self, X):
        return self.model.predict_on_batch(X)

def data_prep(target_label, train_size, model, dataset, sample='all'):
    data = loaders[dataset](target_label, normalization_method='z_score', pad_smiles=False)
    if not sample == 'all':
        data = data.sample(sample)
    # make a set of rd-molecules for featurizer
    mols = []
    idxs = []
    for i, s in enumerate(data.vectorized_inputs):
        mol = Chem.MolFromSmiles(''.join(s), sanitize=True)
        if mol is None:
            continue
        mols.append(mol)
        idxs.append(i)
    # filtering data to include only rdkit-readable mols
    data = data.data_from_idxs(idxs)
    # featurizing data
    if model == "GC":
        featurizer = dc.feat.ConvMolFeaturizer()
        data.vectorized_inputs = featurizer.featurize(mols)
    elif model == "DTNN":
        featurizer = dc.feat.CoulombMatrix(29)
        data.vectorized_inputs = featurizer.featurize(mols)
        data.remove_entries(['empty_arrays'])
    else:
        featurizer = dc.feat.ConvMolFeaturizer()
        data.vectorized_inputs = featurizer.featurize(mols)

    if train_size <= 0.8:
        return data.split_to_groups([train_size, 0.1], add_fill_group=True, random_seed=0)
    else:
        return data.split_to_groups([train_size, 0.05], add_fill_group=True, random_seed=0)

def run_fit(target_label, train_size, model, dataset, sample='all'):
    counter = 0
    fname = lambda x: "{}_{}_{}_{}_{}.csv".format(x, target_label, train_size, model, dataset)
    while True:
        if os.path.isfile(os.path.join(settings.results_dir, fname(counter))):
            counter += 1
        else:
            results_file = os.path.join(settings.results_dir, fname(counter))
            break
    train, val, test = data_prep(target_label, train_size, model, dataset, sample=sample)
    models_dict = {
        "GC": GraphConv,
        "DTNN": DeepTensorNN,
        "MPNN": MessagePassingNN
    }
    # running grid optimization
    additional_descrps = {'model': model, 'train_size': len(train), 'label': target_label, 'count': counter}
    grid_estimation(models_dict[model],
                        train,
                        [("val", val), ("test", test)],
                        estimators=['r_squared', 'rmse','mae', 'mare'],
                        additional_descriptors=additional_descrps,
                        write_to=results_file,
                        init_kwargs=settings.model_params[model])

def main():
    # Running on the rest
    parallel_args_scan(run_fit, 
                        [[1], [0.1, 0.5, 0.8], ['DTNN', "GC"], ["delaney", "lipophilicity", "sampl"]], 
                        addtional_kwargs={},
                        scheduler='distributed')
    # Running on QM9
    parallel_args_scan(run_fit, 
                        [settings.qm9_labels, [0.001, 0.01, 0.1], ['DTNN', "GC"], ["qm9"]], 
                        addtional_kwargs={},
                        scheduler='distributed')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Parser for running files")
    parser.add_argument("-n_workers", type=int, default=1)
    parser.add_argument("-threads_per_worker", type=int, default=1)
    parser.add_argument("-memory_limit", type=str, default="2GB", help="max amount of memory, string such as \'4GB\'")
    args = parser.parse_args()

    client = Client(n_workers=args.n_workers, threads_per_worker=args.threads_per_worker, memory_limit=args.memory_limit)
    main()
    client.close()
