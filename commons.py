# =======================================================
#
#       FILE WITH COMMON FUNCTIONS FOR CALCULATIONS
#
# =======================================================

import pandas as pd
import numpy as numpy
import tensorflow as tf
from time import time

import settings
import sys; sys.path.append(settings.torina_parent_dir)
from Torina.Model.KerasNN import KerasNN
from Torina.Data.Data.SMILES import SMILES
# import tokenizers
from Torina.Data.Tokenizer.FunctionTok import FunctionTok
# import mol functionalities
from Torina.Molecule.Base import BaseMol
from Torina.Molecule.utils import calculate_ecfp4
# import important computation settings


# ================
#   DATA LOADERS
# ================

def _data_loader(smiles, labels, normalization_method, pad_smiles):
    """Load generic smiles and label vectors into SMILES data format"""
    data = SMILES()
    # loading SMILES as inputs
    data.load_inputs_from_text('\n'.join(smiles), sep='chars')
    # padding SMILES
    if pad_smiles:
        data.vectorized_inputs = data.pad_data("_", " ", pad="inputs")
    else:
        data.vectorized_inputs = data.inputs
    # loading target label
    data.vectorized_labels = labels
    # normalizing labels
    if not normalization_method == None:
            data.noramlize_vectors(normalize='labels', method=normalization_method)
    return data

def _load_delaney(x, normalization_method='z_score', pad_smiles=True):
    # x is dummy variable to be used for unified use of these functions
    df = pd.read_csv(settings.delaney_csv_file)
    return _data_loader(df['smiles'].values, 
                        df['measured log solubility in mols per litre'].values, 
                        normalization_method, 
                        pad_smiles)

def _load_lipophilicity(x, normalization_method='z_score', pad_smiles=True):
    df = pd.read_csv(settings.lipophilicity_csv_file)
    return _data_loader(df['smiles'].values, 
                        df['exp'].values, 
                        normalization_method, 
                        pad_smiles)

def _load_qm9(target_label, normalization_method='z_score', pad_smiles=True):
    """Get the QM9 database parsed as SMILES data object
    ARGS:
        - target_label (str): target label to load from QM9
        - normalization_method (str): normalization method to normalize labels. default=z_score
        - pad_smiles (bool): weather to pad the smiles. default=True
    RETURNS:
        SMILES data object"""
    df = pd.read_csv(settings.qm9_csv_file)
    # check input validity
    if not target_label in df.columns:
        raise ValueError("unrecognized target label. allowed values are {}".format(", ".join(df.columns)))
    # removing non readable molecules (cause ECFP4 problems)
    df = df[df['SMILES'] != 'C[C@H](C=C=C(O)=O)C[NH3]']
    df = df[df['SMILES'] != 'C[C@@H]([NH3])c1noc(=O)n1']
    df = df[df['SMILES'] != 'NC[C@H]([NH3])c1nnnn1']
    return _data_loader(df['SMILES'].values, 
                        df[target_label].values, 
                        normalization_method, 
                        pad_smiles)

def _load_sampl(x, normalization_method='z_score', pad_smiles=True):
    df = pd.read_csv(settings.sampl_csv_file)
    return _data_loader(df['smiles'].values, 
                        df['expt'].values, 
                        normalization_method, 
                        pad_smiles)

loaders = {
    "qm9": _load_qm9,
    "delaney": _load_delaney,
    "lipophilicity": _load_lipophilicity,
    "sampl": _load_sampl
}

# =============
#   QM9 Utils
# =============

def get_cms_dict():
    """Method to get a list of coulumb matrices"""
    # readig original dataset
    qm9 = pd.read_csv(settings.qm9_cm_file)
    qm9 = qm9[qm9['SMILES'] != 'C[C@H](C=C=C(O)=O)C[NH3]']
    qm9 = qm9[qm9['SMILES'] != 'C[C@@H]([NH3])c1noc(=O)n1']
    qm9 = qm9[qm9['SMILES'] != 'NC[C@H]([NH3])c1nnnn1']
    qm9 = qm9.set_index('gdb entry')

    # reading cm dataset
    cmdf = pd.read_csv(settings.qm9_cm_file)
    cmdf = cmdf.set_index('gdb entry')
    # appending cm column to qm9
    qm9  = qm9.join(cmdf)
    
    # pulling cms for qm9
    cms = qm9['coulomb matrix'].values
    # getting max size of cm
    max_size = max(qm9['# of atoms'].values)
    # parsing to np.array
    for i, m in enumerate(cms):
        x = np.zeros((max_size, max_size))
        m = np.array([np.array([np.float(v) for v in l.split('\t')]) for l in m.split(';')])
        x[:m.shape[0], :m.shape[1]] = m
        x = x.astype(np.float32)
        cms[i] = x
    return dict([(k, v) for k, v in zip(qm9['SMILES'].values, cms)])



# ===========================================
#   Custom NN model for using grid_estimate
# ===========================================

def add_input_shape_to_params(inputs, params):
    shape = inputs[0].shape
    params["init"]['input_shape'] = [shape]
    return params

class CustomNN (KerasNN):
    '''Custom KerasNN wrapper to be easily used in grid_estimate method on NNs. Provides architectre, optimizer and loss.
    ARGS:
        - input_shape (tuple): input shape of network
        - lr (float): learning rate
        - dropout_rate (float): dropout rate (to prevent overfit)'''

    def __init__(self, input_shape, lr, dropout_rate):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(input_shape))
        if len(input_shape) > 1:
            model.add(tf.keras.layers.Dense(1, activation='tanh'))
            model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(40, activation='tanh'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        super().__init__(model, tf.keras.optimizers.Adam(lr=lr), tf.keras.losses.MSE)

class CustomRNN (KerasNN):
    '''Custom KerasNN wrapper to be easily used in grid_estimate method on RNNs. Provides architectre, optimizer and loss.
    ARGS:
        - input_shape (tuple): input shape of network
        - lr (float): learning rate
        - dropout_rate (float): dropout rate (to prevent overfit)'''

    def __init__(self, input_shape, lr, dropout_rate):
        _model = tf.keras.models.Sequential()
        _model.add(tf.keras.layers.Input(input_shape))
        # fixing input shapes for some tokenizations   
        if len(input_shape) < 2:
            _model.add(tf.keras.layers.Reshape(input_shape + (1, )))
        print(input_shape)
        _model.add(tf.keras.layers.LSTM(50, activation='tanh'))
        _model.add(tf.keras.layers.Dropout(dropout_rate))
        _model.add(tf.keras.layers.Dense(50, activation='tanh'))
        _model.add(tf.keras.layers.Dense(1, activation='linear'))
        super().__init__(_model, tf.keras.optimizers.Adam(lr=lr), tf.keras.losses.MSE)

# ===========================================
#         Dask Parallelizatoin Utils
# ===========================================

def car_prod_2_vec(vecs1, vecs2):
    prod = []
    for vec1 in vecs1:
        for vec2 in vecs2:
            prod.append(vec1 + [vec2])
    return prod

def cartesian_prod(vecs):
    """calculate cartesian product on a list of vectors"""
    prod = [v if type(v) is list else [v] for v in vecs[0]]
    for vec in vecs[1:]:
        prod = car_prod_2_vec(prod, vec)
    return prod

def parallel_args_scan(func: callable, agrs_list: list, res_file: str='', addtional_kwargs: dict={}, scheduler: str='distributed', checkpoint: int=0):
    """Scan function values for different arguments. parallelize for loop of the type:
        >>> for args in args_list:
        ...     func(*arg, **additional_kwargs)
    function should return a dictionary of reaults, that is written to a results file (res_file) as a dataframe. 
    Note that in order to use the \'distributed\' scheduler one must have a runnig dask Client.
    ARGS:
        - func (callable): function to run in each for loop iteration. must retrun a dictionary
        - args_list (list): a list of arguments to feed to the functions. if a nested list is given, performes a cartesian product to get all argument combinations
                            for example: args_list = [[1, 2], [0, 1]]   -> [[1, 0], [2, 0], [1, 1], [2, 1]]
                                                     [[[1, 2]], [0, 1]] -> [[[1, 2], 0], [[1, 2], 1]]
        - res_file (path): (optional) path to the results csv file. default='' (no results file)
        - additional_kwargs (dict): (optional) additional non-changing kwargs to feed to the function. default={}
        - scheduler (str): the dask scheduler to use. default=distributed
        - checkpoint (int): number of iterations afterwhich results are written to res_file. might slow down performance, use with care. defaults=0 (no checkpoints)"""
    t1 = time()
    import dask as da

    # setting up argument combinations
    agrs_list = [args if type(args) is list else [args] for args in agrs_list]
    agrs_list = cartesian_prod(agrs_list)
    # initializing
    counter = 1 # checkpoint counter
    c = [] # must have a list to save the results to
    # running loop
    for args in agrs_list:
        # add delayed function results to list
        c.append(da.delayed(func) (*args, **addtional_kwargs))
        # calculate on checkpoints
        if counter == checkpoint and not len(res_file) == 0:
            counter = 0
            df = pd.DataFrame(da.compute(c, scheduler=scheduler)[0])
            df.to_csv(res_file)
        # update counter
        counter += 1
    if not len(res_file) == 0:
        df = pd.DataFrame(da.compute(c, scheduler=scheduler)[0])
        df.to_csv(res_file)
    else:
        da.compute(c, scheduler=scheduler)
    t2 = time()
    print("***** Computation is Finished Normally *****")
    print("Total computation time:", t2 - t1, "seconds")

# ==================================
#         Tokenization Utils
# ==================================

def cm_tokenizer():
    """Get the cm_tokenizer"""
    cm_dict = get_cms_dict()
    f = lambda v: cm_dict[v]
    return FunctionTok(f)

def ecfp4_tokenizer():
    """Get the ECFP4 tokenizer"""
    # define the tokenization function
    def f(x):
        mol = BaseMol()
        mol.from_str(''.join(x))
        return calculate_ecfp4(mol)
    # return function tokenizer
    return FunctionTok(f)

def get_tokenizer(name):
    from Torina.Data.Tokenizer.OneHot import OneHot
    d = {
            "ONEHOT": OneHot,
            "CM": cm_tokenizer,
            "ECFP4": ecfp4_tokenizer
        }
    return d[name]()