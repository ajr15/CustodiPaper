# =======================================================
#
#    RUN FILE FOR RUNNING FITS WITH CUSTODI MODEL OR
#                   REPRESENTATION
#    
#    Run by running the command:
#    python degree_dependency.py
#    
#    The file creates a csv file with fit results of
#    CUSTODI with different degrees and alpha values
# 
# =======================================================

from dask.distributed import Client
import numpy as np
import os

import settings
import sys; sys.path.append(settings.torina_parent_dir)
from time import time
from Torina.Model.utils import grid_estimation
from Torina.Data.Tokenizer.Custodi import Custodi as custodi_rep
from Torina.Model.Custodi import Custodi as custodi_model
from Torina.Model.commons import kw_cartesian_prod
from commons import *


def data_prep(target_label: str, train_size: float, dataset: str, custodi_params={}, for_custodi=False, sample='all'):
    """Data preperation method for run.
    ARGS:
        - target_label (str): name of target property in qm9 dataset
        - train_size (float): relative size of train set
        - dataset (str): dataset name to use in the computation (qm9, lipophilicity, delaney, sampl)
        - custodi_params (dict): dict of parameters for custodi tokenizer (no need of for_custodi=True)
        - for_custodi (bool): weather the data is meant for custodi model
    RETURN:
        (list) three data objects (train (train size), validation (10%), test (rest)) ready to use"""
    if for_custodi:
        # prepare data for CUSTODI model
        # CUSTODI model doesn't require padding and tokenization!
        data = loaders[dataset.lower()](target_label, normalization_method='z_score', pad_smiles=False)
    else:
        # prepare data for non-CUSTODI model with CUSOTODI rep
        # data with padded smiles and normalized labels (z-score)
        data = loaders[dataset.lower()](target_label)
        data.vectorized_labels = data.vectorized_labels.tolist()
    # taking a sample of the data (if requested)
    if not sample == 'all':
        data = data.sample(sample)
    if train_size <= 0.8:
        groups = data.split_to_groups([train_size, 0.1], add_fill_group=True, random_seed=0)
    else:
        groups = data.split_to_groups([train_size, 0.05], add_fill_group=True, random_seed=0)
    # training and tokenizing data using CUSTODI
    if not for_custodi:
        tok = custodi_rep(**custodi_params)
        tok.train(groups[0].vectorized_inputs, groups[0].vectorized_labels)
        for i in range(len(groups)):
            groups[i].set_tokenizer(tok)
            groups[i].vectorized_inputs = groups[i].tokenize(groups[i].vectorized_inputs)
    return groups

def run_custodi_fit(target_label, train_size, dataset, sample='all'):
    counter = 0
    fname = lambda x: "{}_{}_{}_CUTODIMODEL_{}.csv".format(x, target_label, train_size, dataset)
    while True:
        if os.path.isfile(os.path.join(settings.results_dir, fname(counter))):
            counter += 1
        else:
            results_file = os.path.join(settings.results_dir, fname(counter))
            break
    train, val, test = data_prep(target_label, train_size, dataset, for_custodi=True, sample=sample)
    grid_estimation(custodi_model, 
                    train,
                    [("val", val), ("test", test)],
                    estimators=['r_squared', 'rmse','mae', 'mare'], 
                    additional_descriptors={'model': "CUSTODI", 'tok': "None", 'train_size': len(train), 'label': target_label, 'count': counter},
                    write_to=results_file,
                    init_kwargs=settings.model_params["CUSTODI"])

def run_model_fit(target_label, model, train_size, dataset, sample='all'):
    counter = 0
    fname = lambda x: "{}_{}_{}_{}_{}.csv".format(x, target_label, train_size, model, dataset)
    while True:
        if os.path.isfile(os.path.join(settings.results_dir, fname(counter))):
            counter += 1
        else:
            results_file = os.path.join(settings.results_dir, fname(counter))
            break
    custodi_ps = kw_cartesian_prod(settings.model_params["CUSTODI"])
    for params in custodi_ps:
        train, val, test = data_prep(target_label, train_size, dataset, custodi_params=params, for_custodi=False, sample=sample)
        if model == "NN":
            model_params = {}
            model_params["NN"] = add_input_shape_to_params(train.vectorized_inputs, settings.model_params["NN"])
        else:
            model_params = settings.model_params
        additional_descrps = {'model': model, 'tok': "CUSTODI", 'train_size': len(train), 'label': target_label, 'dataset': dataset}
        additional_descrps.update(params)
        grid_estimation(settings.models_dict[model],
                        train,
                        [("val", val), ("test", test)],
                        estimators=['r_squared', 'rmse','mae', 'mare'],
                        additional_descriptors=additional_descrps,
                        write_to=results_file,
                        train_kwargs=settings.model_params[model]["train"],
                        init_kwargs=settings.model_params[model]["init"])

def main():
    # Running on the rest
    parallel_args_scan(run_custodi_fit, 
                        [[None], [0.1, 0.5, 0.8], ["delaney", "lipophilicity", "sampl"]], 
                        addtional_kwargs={},
                        scheduler='distributed')
    # Running on QM9
    parallel_args_scan(run_model_fit, 
                        [[None], ["KRR", "NN"], [0.1, 0.5, 0.8], ["delaney", "lipophilicity", "sampl"]], 
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
