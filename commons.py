# =======================================================
#
#       FILE WITH COMMON FUNCTIONS FOR CALCULATIONS
#
# =======================================================

import pandas as pd
import numpy as numpy
import tensorflow as tf

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


def get_cms_dict():
    """Method to get a list of coulumb matrices"""
    # readig original dataset
    qm9 = pd.read_csv('/home/shaharpit/Torina/Testing/CoSToDi_PAPER/QM9/database.csv')
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
        mol.from_str(''.join(smiles))
        return calculate_ecfp4(mol, n_bits=1024)
    # return function tokenizer
    return FunctionTok(f)

def qm9_data(target_label, normalization_method='z_score', pad_smiles=True):
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
    # setting up data object
    data = SMILES()
    # loading SMILES as inputs
    data.load_inputs_from_text('\n'.join(df['SMILES'].values), sep='chars')
    # padding SMILES
    if pad_smiles:
        data.vectorized_inputs = data.pad_data("_", " ", pad="inputs")
    else:
        data.vectorized_inputs = data.inputs
    # loading target label
    data.vectorized_labels = df[target_label].values
    # normalizing labels
    if not normalization_method == None:
            data.noramlize_vectors(normalize='labels', method=normalization_method)
    return data

# ===========================================
#   Custom NN model for using grid_estimate
# ===========================================

def add_input_shape_to_params(inputs, params):
    shape = inputs[0].shape
    params["init"]['input_shape'] = [shape]
    return params

class CustomNN (KerasNN):
    '''Custom KerasNN wrapper to be easily used in grid_estimate method. Provides architectre, optimizer and loss.
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