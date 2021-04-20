# =======================================================
#
#          RUN FILE FOR RUNNING FITS WITH A GENERAL
#                       REPRESENTATION
#    
#    Run by running the command:
#    python RepRun.py
#    
#    The file creates a csv file with fit results of
#    CUSTODI with different degrees and alpha values
# 
# =======================================================

import settings
from commons import *
import sys; sys.path.append(settings.torina_parent_dir)
from Torina.Model.utils import grid_estimation
import numpy as np
import os

def data_prep(target_label: str, train_size: float, tokenizer: str, dataset: str, sample='all'):
    """Data preperation method for run.
    ARGS:
        - target_label (str): name of target property in qm9 dataset
        - train_size (float): relative size of train set
        - dataset (str): dataset name to use in the computation (qm9, lipophilicity, delaney, sampl)
        - tokenizer (str): name of tokenizer to use (ONEHOT, CM, ECFP4)
    RETURN:
        (list) three data objects (train (train size), validation (10%), test (rest)) ready to use"""
    data = loaders[dataset.lower()](target_label)
    data.vectorized_labels = data.vectorized_labels.tolist()
    # taking a sample of the data (if requested)
    if not sample == 'all':
        data = data.sample(sample)
    # encoding data
    data.set_tokenizer(get_tokenizer(tokenizer))
    if tokenizer == 'ONEHOT':
        data.tokenizer.set_word_tokenization_dict(data.vectorized_inputs)
    data.vectorized_inputs = np.array(data.tokenize('vectorized_inputs'))
    data.remove_entries(['empty_arrays'])
    data.vectorized_attributes_to_nparrays()
    # splitting data
    if train_size <= 0.8:
        groups = data.split_to_groups([train_size, 0.1], add_fill_group=True, random_seed=target_label * 10)
    else:
        groups = data.split_to_groups([train_size, 0.05], add_fill_group=True, random_seed=target_label * 10)
    return groups

def run_fit(target_label, model, train_size, tokenizer, dataset, sample='all'):
    # setting file name (with proper count)
    counter = 0
    fname = lambda x: "{}_{}_{}_{}_{}_{}.csv".format(x, target_label, train_size, model, tokenizer, dataset)
    while True:
        if os.path.isfile(os.path.join(settings.results_dir, fname(counter))):
            counter += 1
        else:
            results_file = os.path.join(settings.results_dir, fname(counter))
            break
    # making data
    train, val, test = data_prep(target_label, train_size, tokenizer, dataset, sample=sample)
    print("NUMBER OF TRAIN SAMPLES:", len(train.vectorized_inputs))
    # setting up model parameters
    if model == "NN" or model == "RNN":
        model_params = {}
        model_params[model] = add_input_shape_to_params(train.vectorized_inputs, settings.model_params[model])
    else:
        model_params = settings.model_params
    # setting additional descriptors
    additional_descrps = {'model': model, 'tok': tokenizer, 'train_size': len(train), 'label': target_label, 'dataset': dataset, 'count': counter}
    # running computation
    grid_estimation(settings.models_dict[model],
                    train,
                    [("val", val), ("test", test)],
                    estimators=['r_squared', 'rmse','mae', 'mare'],
                    additional_descriptors=additional_descrps,
                    write_to=results_file,
                    train_kwargs=settings.model_params[model]["train"],
                    init_kwargs=settings.model_params[model]["init"])

def main():
    parallel_args_scan(run_fit, 
                        [[1], ["RNN"], [0.05], ["ONEHOT"], ["lipophilicity"]], 
                        addtional_kwargs={},
                        scheduler='synchronous')
        
def main1():
    # Running non-RNN models with some reps
    #parallel_args_scan(run_fit, 
    #                    [[1, 2, 3, 4], ["NN", "KRR"], [0.1, 0.5, 0.8], ["ECFP4"], ["delaney", "lipophilicity", "sampl"]], 
    #                    addtional_kwargs={},
    #                    scheduler='distributed')
    # Running RNN models
    parallel_args_scan(run_fit, 
                    [[1, 2, 3, 4], ["RNN"], [0.1, 0.5, 0.8], ["ONEHOT"], ["delaney", "lipophilicity", "sampl"]], 
                    addtional_kwargs={},
                    scheduler='distributed')

if __name__ == '__main__':
    import argparse
    from dask.distributed import Client
    parser = argparse.ArgumentParser(description="Parser for running files")
    parser.add_argument("-n_workers", type=int, default=1)
    parser.add_argument("-threads_per_worker", type=int, default=1)
    parser.add_argument("-memory_limit", type=str, default="2GB", help="max amount of memory, string such as \'4GB\'")
    args = parser.parse_args()

    client = Client(n_workers=args.n_workers, threads_per_worker=args.threads_per_worker, memory_limit=args.memory_limit)
    main()
    client.close()