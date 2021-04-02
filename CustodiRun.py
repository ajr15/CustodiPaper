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

from dask import delayed, compute
from dask.distributed import Client
import numpy as np

import settings
import sys; sys.path.append(settings.torina_parent_dir)
from time import time
from Torina.Model.utils import grid_estimation
from Torina.Data.Tokenizer.Custodi import Custodi as custodi_rep
from Torina.Model.Custodi import Custodi as custodi_model
from Torina.Model.commons import kw_cartesian_prod
from commons import qm9_data, CustomNN, add_input_shape_to_params


def data_prep(target_label, train_size, custodi_params={}, for_custodi=False, sample='all'):
    """Data preperation method for run.
    ARGS:
        - target_label (str): name of target property in qm9 dataset
        - train_size (float): relative size of train set
        - custodi_params (dict): dict of parameters for custodi tokenizer (no need of for_custodi=True)
        - for_custodi (bool): weather the data is meant for custodi model
    RETURN:
        (list) three data objects (train (train size), validation (10%), test (rest)) ready to use"""
    if for_custodi:
        # prepare data for CUSTODI model
        # CUSTODI model doesn't require padding and tokenization!
        qm9 = qm9_data(target_label, normalization_method='z_score', pad_smiles=False)
    else:
        # prepare data for non-CUSTODI model with CUSOTODI rep
        # data with padded smiles and normalized labels (z-score)
        qm9 = qm9_data(target_label)
        qm9.vectorized_labels = qm9.vectorized_labels.tolist()
    # taking a sample of the data (if requested)
    if not sample == 'all':
        qm9 = qm9.sample(sample)
    print("Standard Deviation:", np.std(qm9.vectorized_labels))
    if train_size < 0.8:
        groups = qm9.split_to_groups([train_size, 0.1], add_fill_group=True, random_seed=0)
    else:
        groups = qm9.split_to_groups([train_size, 0.05], add_fill_group=True, random_seed=0)
    # training and tokenizing data using CUSTODI
    if not for_custodi:
        tok = custodi_rep(**custodi_params)
        tok.train(groups[0].vectorized_inputs, groups[0].vectorized_labels)
        for i in range(len(groups)):
            groups[i].set_tokenizer(tok)
            groups[i].vectorized_inputs = groups[i].tokenize(groups[i].vectorized_inputs)
    return groups

@delayed
def run_custodi_fit(target_label, train_size, results_file, sample='all'):
    train, val, test = data_prep(target_label, train_size, for_custodi=True, sample=sample)
    grid_estimation(custodi_model, 
                    train,
                    [("val", val), ("test", test)],
                    estimators=['r_squared', 'rmse','mae', 'mare'], 
                    additional_descriptors={'model': "CUSTODI", 'tok': "None", 'train_size': len(train)},
                    write_to=results_file,
                    init_kwargs=settings.model_params["CUSTODI"])

@delayed
def run_model_fit(target_label, model, train_size, results_file, sample='all'):
    custodi_ps = kw_cartesian_prod(settings.model_params["CUSTODI"])
    for params in custodi_ps:
        train, val, test = data_prep(target_label, train_size, custodi_params=params, for_custodi=False, sample=sample)
        if model == "NN":
            model_params = {}
            model_params["NN"] = add_input_shape_to_params(train.vectorized_inputs, settings.model_params["NN"])
        else:
            model_params = settings.model_params
        additional_descrps = {'model': model, 'tok': "CUSTODI", 'train_size': len(train)}
        additional_descrps.update(params)
        grid_estimation(settings.models_dict[model],
                        train,
                        [("val", val), ("test", test)],
                        estimators=['r_squared', 'rmse','mae', 'mare'],
                        additional_descriptors=additional_descrps,
                        write_to=results_file,
                        train_kwargs=settings.model_params[model]["train"],
                        init_kwargs=settings.model_params[model]["init"])

def custodi_model_test_run():
    counter = 1
    c = []
    ti = time()
    for label in ["dipole moment [Debye]", "gap [Hartree]"]:
        for train_size in [0.1]:
            print("Running for {} and {}% train size (~{} samples)".format(label, round(train_size * 100), round(1.1e5 * train_size)))
            res_file = './Results/Res{}.csv'.format(counter)
            c.append(run_custodi_fit(label, train_size, res_file, sample=10000))
            counter += 1
    compute(c, scheduler='distributed')
    tf = time()
    print("Computation is Done ! Computation time:", tf - ti, "seconds")

def main():
    counter = 1
    c = []
    ti = time()
    for label in ["dipole moment [Debye]", "gap [Hartree]"]:
        for train_size in [0.1]:
            print("Running for {} and {}% train size (~{} samples)".format(label, round(train_size * 100), round(1.1e5 * train_size)))
            res_file = './Results/Res{}.csv'.format(counter)
            c.append(run_model_fit(label, "NN", train_size, res_file, sample=10000))
            counter += 1
    compute(c, scheduler='distributed')
    tf = time()
    print("Computation is Done ! Computation time:", tf - ti, "seconds")


if __name__ == '__main__':
    dask_client = Client(n_workers=2, threads_per_worker=1)
    main()
    dask_client.close()
