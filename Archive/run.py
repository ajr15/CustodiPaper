import sys; sys.path.append('/home/shaharpit/Documents')
import numpy as np
import os
from copy import copy 
import warnings; warnings.simplefilter(action = 'ignore', category=FutureWarning)
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use('Agg')
import pandas as pd

from Torina.Data.Objects import SMILES
from Torina.Data.utils import inverse_zscore_normalization
from Torina.Data.Base import flatten, generate_data_using_comp_protocol
from Torina.Data.utils import *
from Torina.Model.utils import KerasArchitectures
from Torina.Model.Objects import *
from Torina.Molecule.Base import BaseMol
from Torina.Molecule.utils import calculate_ecfp4

# =======================
#      Utils Funcs
# =======================

def select_idxs(max_idx, sample_size):
    idxs = []
    while True:
        i = np.random.randint(0, max_idx)
        if not i in idxs:
            idxs.append(i)
        if len(idxs) == sample_size:
            break
    return idxs

def gen_train_val_test_idxs(train_size, val_size, data_size, train_idxs, val_idxs):
    print("setting train idxs...")
    while True:
        if len(train_idxs) == round(train_size * data_size):
            break
        i = np.random.randint(0, data_size)
        if not i in train_idxs:
            train_idxs.append(i)
    print("setting val idxs...")
    while True:
        if len(val_idxs) == round(val_size * data_size):
            break
        i = np.random.randint(0, data_size)
        if not i in train_idxs and not i in val_idxs:
            val_idxs.append(i)
    print("setting test idxs...")
    test_idxs = [i for i in range(data_size) if not i in train_idxs and not i in val_idxs]
    return train_idxs, val_idxs, test_idxs

def cartesian_prod(vecs1, vecs2):
    prod = []
    for vec1 in vecs1:
        for vec2 in vecs2:
            prod.append(vec1 + [vec2])
    return prod

def kw_cartesian_prod(kwargs_dict):
    '''generates a list of dict for all combinations of keywords in kwargs_dict'''
    vec_cartesian = [[]]
    for vals in kwargs_dict.values():
        vec_cartesian = cartesian_prod(vec_cartesian, vals)
    dict_list = []
    for vec in vec_cartesian:
        d = dict([(k, v) for k, v in zip(kwargs_dict.keys(), vec)])
        dict_list.append(d)
    return dict_list

def _safe_calc_diff_with_func(pred, true, func):
    s = 0
    for p, t in zip(pred, true):
        val = func(p, t)
        if not val == [np.inf] and not val == np.inf:
            s = s + val
    return s / len(pred)

def calc_descps(pred, true, prefix='', un_normalize_params=None):
    if not un_normalize_params == None:
        pred = inverse_zscore_normalization(pred, *un_normalize_params)
        true = inverse_zscore_normalization(true, *un_normalize_params)
    descps = {}
    rmse_func = lambda p, t: np.square(p - t)
    descps[prefix + 'rmse'] = np.sqrt(_safe_calc_diff_with_func(pred, true, rmse_func))
    mae_func = lambda p, t: np.abs(p - t)
    descps[prefix + 'mae'] = _safe_calc_diff_with_func(pred, true, mae_func)
    mare_func = lambda p, t: np.abs((p - t) / t) if not t == 0 else 0
    descps[prefix + 'mare'] = _safe_calc_diff_with_func(pred, true, mare_func)
    # corrects shape of descrps values in case a numpy array is returned
    for k, v in descps.items():
        try:
            descps[k] = v[0]
        except IndexError:
            continue
    return descps

def data_from_idxs(data, idxs, t):
    if t == 'inputs':
        try:
            return np.array([data.vectorized_inputs[i] for i in idxs])
        except IndexError:
            l = len(data.vectorized_inputs)
            for idx in idxs:
                if idx > l - 1:
                    raise IndexError(f"{idx} in list is greater than list length {l}")
    elif t == 'labels':
        return np.array([data.vectorized_labels[i] for i in idxs])
    else:
        raise ValueError(f'unrecognized data type {t}, can be labels or inputs')


def KRR_model(train_inputs, train_labels, model_alpha=0.01, kernel='rbf'):
    model = KernalRidge(model_alpha, kernel=kernel)
    if len(train_inputs[0].shape) > 1:
        train_inputs = [flatten(s) for s in train_inputs]
    model.train(train_inputs, train_labels)
    return model

def NN_model(train_inputs, train_labels, lr=0.001, dropout_rate=3, epochs=250):
    input_shape = train_inputs[0].shape
    input_size = np.prod(train_inputs[0].shape)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(input_shape))
    if len(input_shape) > 1:
      model.add(tf.keras.layers.Dense(1, activation='tanh'))
      model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(40, activation='tanh'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model = KerasNN(model, tf.keras.optimizers.Adam(lr=lr), tf.keras.losses.MSE)
    model.train(train_inputs, train_labels, epochs=epochs, verbose=1)
    return model

def custodi_model(train_inputs, train_labels, model_degree=2, model_alpha=0.05, max_iter=10000):
    model = CuSToDi(model_degree, model_alpha, max_iter)
    model.train(train_inputs, train_labels)
    return model

def RNN_model(train_inputs, train_labels, dropout_rate=0, lr=0.01, epochs=100):
    input_shape = np.array(train_inputs[0]).shape
    _model = tf.keras.models.Sequential()
    _model.add(tf.keras.layers.Input(input_shape))
    # fixing input shapes for some tokenizations   
    if len(input_shape) < 2:
        _model.add(tf.keras.layers.Reshape(input_shape + (1, )))   
    _model.add(tf.keras.layers.LSTM(50, activation='tanh'))
    _model.add(tf.keras.layers.Dropout(dropout_rate))
    _model.add(tf.keras.layers.Dense(50, activation='tanh'))
    _model.add(tf.keras.layers.Dense(1, activation='linear'))
    NN = KerasNN(_model, tf.keras.optimizers.Adam(lr=lr), tf.keras.losses.MSE)
    NN.train(train_inputs, train_labels, epochs=epochs, verbose=1)
    return NN

def get_trained_model(train_inputs, train_labels, model='KRR', **kwargs):
    models_dict = {'KRR': KRR_model, 'NN': NN_model, 'custodi': custodi_model, 'RNN': RNN_model}
    if not model in models_dict.keys():
        raise ValueError(f"Unrecognized tokenization method {method}, allowed methods are {', '.join(models_dict.keys())}")
    return models_dict[model] (train_inputs, train_labels, **kwargs)

def custodi_tokenize(data, train_inputs, train_labels, degree=2, alpha=0.01, max_iter=10000):
    model = CuSToDi(degree, alpha, max_iter)
    model.train(train_inputs, train_labels)
    data.vectorized_inputs = model.encode(data.vectorized_inputs)
    return data

def aug_custodi_tokenize(data, train_inputs, train_labels, alpha=0.01, max_iter=10000):
    model = CuSToDi(1, alpha, max_iter)
    model.train(train_inputs, train_labels)
    d = gen_word_tokenization_dict(data.vectorized_inputs)
    counter = 0
    for k, v in d.items():
        vec = np.zeros(len(d))
        try:
            vec[v] = model.dictionary[k]
        except KeyError:
            vec[v] = np.mean(list(model.dictionary.values()))
        d[k] = vec
    data.tokenization_func = d
    data.vectorized_inputs = data.tokenize('vectorized_inputs', keep_shape=False)
    return data

def ECFP4_tokenize(data, train_inputs, train_labels):
    data.vectorized_inputs = []
    for i, smiles in enumerate(data.inputs):
        mol = BaseMol()
        mol.from_str(''.join(smiles))
        try:
            data.vectorized_inputs.append(calculate_ecfp4(mol, 1024))
        except Exception as err:
            print("Errors encountered with", ''.join(smiles))
            data.vectorized_labels.pop(i)
    return data

def random_tokenize(data, train_inputs, train_labels, shape=(1024, )):
    shape = (len(data.vectorized_inputs), ) + shape
    data.vectorized_inputs = np.random.rand(*shape)
    return data

def gen_word_tokenization_dict(vecs):
    '''Method to generate standard word tokenization dictionary'''
    char_set = set()
    for char in flatten(vecs):
        char_set.add(char)
    return dict([(char, i) for i, char in enumerate(list(char_set))])

def one_hot_tokenize(data, train_inputs, train_labels):
    d = gen_word_tokenization_dict(data.vectorized_inputs)
    for k, v in d.items():
        vec = np.zeros(len(d))
        vec[v] = 1
        d[k] = vec
    data.tokenization_func = d
    data.vectorized_inputs = data.tokenize('vectorized_inputs', keep_shape=False)
    return data

def word_tokenize(data, train_inputs, train_labels):
    data.set_word_tokenization_func()
    data.vectorized_inputs = data.tokenize('vectorized_inputs', keep_shape=False)
    data.noramlize_vectors(normalize='inputs', method='z_score')
    return data

def get_cms():
    # readig original dataset
    qm9 = pd.read_csv('/home/shaharpit/Torina/Testing/CoSToDi_PAPER/QM9/database.csv')
    qm9 = qm9[qm9['SMILES'] != 'C[C@H](C=C=C(O)=O)C[NH3]']
    qm9 = qm9[qm9['SMILES'] != 'C[C@@H]([NH3])c1noc(=O)n1']
    qm9 = qm9[qm9['SMILES'] != 'NC[C@H]([NH3])c1nnnn1']
    qm9 = qm9.set_index('gdb entry')

    # reading cm dataset
    cmdf = pd.read_csv('/home/shaharpit/Torina/Testing/CoSToDi_PAPER/QM9/red_cm.csv')
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
    return cms

def cm_tokenize(data, train_inputs, train_labels):
    cms = get_cms()
    data.vectorized_inputs = cms
    return data

def none_tokenize(data, train_inputs, train_labels):
    data.vectorized_inputs = data.inputs
    return data


def tokenize_data(data, train_inputs, train_labels, tokenization_method='word', **kwargs):
    tokenization_methods_dict = {'custodi': custodi_tokenize, 'ECFP4': ECFP4_tokenize, 'word': word_tokenize, 'one_hot': one_hot_tokenize, 'aug_custodi': aug_custodi_tokenize,
                                    'random': random_tokenize, 'None': none_tokenize, 'cm': cm_tokenize}
    if not tokenization_method in tokenization_methods_dict.keys():
        raise ValueError(f"Unrecognized tokenization method {tokenization_method}, allowed methods are {', '.join(tokenization_methods_dict.keys())}")
    data = tokenization_methods_dict[tokenization_method] (data, train_inputs, train_labels, **kwargs)
    return data
    
def predict(data, idxs, model):
    inputs = np.array(data_from_idxs(data, idxs, 'inputs'))
    if isinstance(model, KernalRidge):
        if len(inputs[0].shape) > 1:
            inputs = [flatten(x) for x in inputs]
    return model.predict(inputs)

# =======================
#       Run Funcs
# =======================

def data_prep(label, sample_idxs, tokenization_method, train_idxs, normalization_method='z_score', **tokenization_kwargs):
    df = pd.read_csv('/home/shaharpit/Torina/Testing/CoSToDi_PAPER/QM9/database.csv')
    
    # removing non readable molecules (cause ECFP4 problems)
    df = df[df['SMILES'] != 'C[C@H](C=C=C(O)=O)C[NH3]']
    df = df[df['SMILES'] != 'C[C@@H]([NH3])c1noc(=O)n1']
    df = df[df['SMILES'] != 'NC[C@H]([NH3])c1nnnn1']
    # sampling
    if not sample_idxs== None:
        df = df.iloc[sample_idxs, :]
    data = SMILES()
    data.load_inputs_from_text('\n'.join(df['SMILES'].values), sep='chars')
    data.vectorized_labels = df[label].values
    data.vectorized_inputs = data.pad_data("_", " ", pad="inputs")
    if not normalization_method == None:
        data.noramlize_vectors(normalize='labels', method=normalization_method)
    data = tokenize_data(data, data_from_idxs(data, train_idxs, 'inputs'), data_from_idxs(data, train_idxs, 'labels'), tokenization_method, **tokenization_kwargs)
    return data

def run_fit(data, model, train_idxs, val_idxs, test_idxs, image_path, prefix, **train_kwargs):
    model = get_trained_model(data_from_idxs(data, train_idxs, 'inputs'), data_from_idxs(data, train_idxs, 'labels'), model, **train_kwargs)
    train_pred = predict(data, train_idxs, model)
    test_pred = predict(data, test_idxs, model)
    val_pred = predict(data, val_idxs, model)
    print("calculating descriptors...")
    descrps = calc_descps(train_pred, data_from_idxs(data, train_idxs, 'labels'), 'train_', data._label_norm_params)
    descrps.update(calc_descps(test_pred, data_from_idxs(data, test_idxs, 'labels'), 'test_', data._label_norm_params))
    descrps.update(calc_descps(val_pred, data_from_idxs(data, val_idxs, 'labels'), 'val_', data._label_norm_params))
    #model.plot_fit(data_from_idxs(data, train_idxs, 'inputs'), data_from_idxs(data, train_idxs, 'labels'), show=False, alpha=0.2) # taining set
    #plt.gcf()
    #plt.savefig(os.path.join(image_path, prefix + '_train.png'))
    #plt.close()
    #model.plot_fit(data_from_idxs(data, test_idxs, 'inputs'), data_from_idxs(data, test_idxs, 'labels'), show=False, alpha=0.2) # testing set
    #plt.gcf()
    #plt.savefig(os.path.join(image_path, prefix + '_test.png'))
    #plt.close()
    return descrps

def custodi_run_fit(data, model, train_idxs, test_idxs, image_path, prefix, **train_kwargs):
    model = get_trained_model([data.inputs[i] for i in train_idxs], data_from_idxs(data, train_idxs, 'labels'), model, **train_kwargs)
    train_pred = predict(data, train_idxs, model)
    test_pred = predict(data, test_idxs, model)
    print("calculating descriptors...")
    descrps = calc_descps(train_pred, data_from_idxs(data, train_idxs, 'labels'), 'train_', data._label_norm_params)
    descrps.update(calc_descps(test_pred, data_from_idxs(data, test_idxs, 'labels'), 'test_', data._label_norm_params))
    #model.plot_fit([data.inputs[i] for i in train_idxs], data_from_idxs(data, train_idxs, 'labels'), show=False, alpha=0.2) # taining set
    #plt.gcf()
    #plt.savefig(os.path.join(image_path, prefix + '_train.png'))
    #plt.close()
    #model.plot_fit([data.inputs[i] for i in test_idxs], data_from_idxs(data, test_idxs, 'labels'), show=False, alpha=0.2) # testing set
    #plt.gcf()
    #plt.savefig(os.path.join(image_path, prefix + '_test.png'))
    #plt.close()
    return descrps

def run_calc(sample_idxs, train_idxs, val_idxs, test_idxs, label, results_df, normalization_methods, tokenization_kw_dicts, train_kw_dicts, runner_id, train_size, recover_from):
    tot_dicts = []
    for l in [l for l in cartesian_prod(tokenization_kw_dicts, train_kw_dicts)]:
        dct = {}
        for d in l:
            dct.update(d) if not type(d) == list else dct.update(d[0])
        tot_dicts.append(dct)
    
    tokenization_keys = set()
    for d in tokenization_kw_dicts:
        for k in d[0].keys():
            if not k == 'tokenization_method':
                tokenization_keys.add(k)
    
    train_kw_dicts = [d[0] if type(d) is list else d for d in train_kw_dicts]
    train_keys = set()
    for d in train_kw_dicts:
        for k in d.keys():
            if not k == 'model':
                train_keys.add(k)

    changing_kwargs = []
    for d in tot_dicts:
        for k, v in d.items():
            if len(v) > 1 and not k == 'tokenization_method' and not k == 'model':
                changing_kwargs.append(k)

    base_image_dir = '/home/shaharpit/Torina/Testing/CoSToDi_PAPER/QM9/plots' + runner_id
    #if not os.path.isdir(base_image_dir):
    #    os.mkdir(base_image_dir)
    
    kw_dicts = [d for kw_dict in tot_dicts for d in kw_cartesian_prod(kw_dict)]
    kw_dicts = []
    for kw_dict in tot_dicts:
        for d in kw_cartesian_prod(kw_dict):
            for norm_method in normalization_methods:
                d['norm_method'] = norm_method
                kw_dicts.append(d)
    if not recover_from == 0:
        print("Recovers from idx = {}".format(recover_from))
        kw_dicts = kw_dicts[recover_from:]

    for d in kw_dicts:
        print("Running calculation with:", 
            " ".join([k + ": " + str(v)  + ',' for k, v in d.items() if k in changing_kwargs] + [f"tokenization method: {d['tokenization_method']}, model: {d['model']}, normalization: {d['norm_method']}"]))
        
        tokenization_kwds = dict([(k, d[k]) for k in d.keys() if k in tokenization_keys])
        # correcting for custodi model case - no need for tokenization!
        if d['model'] == 'Costodi':
            d['tokenization_method'] = 'None'
            tokenization_kwds = {}
        train_kwds = dict([(k, d[k]) for k in d.keys() if k in train_keys])
        data = data_prep(label, sample_idxs, d['tokenization_method'], train_idxs, normalization_method=d['norm_method'], **tokenization_kwds)
        image_path = os.path.join(base_image_dir, '_'.join([f'label_{label}'] + [k + "_" + str(v) for k, v in d.items() if k in changing_kwargs]))
        #if not os.path.isdir(image_path):
        #    os.mkdir(image_path)
        image_prefix = 'model_{}_tokenization_method_{}_train_size_{}'.format(d['model'], d['tokenization_method'], len(train_idxs))
        #if d['model'] == 'custodi':
        #    descrps = custodi_run_fit(data, d['model'], train_idxs, test_idxs, image_path, image_prefix, **train_kwds) 
        #else:
        descrps = run_fit(data, d['model'], train_idxs, val_idxs, test_idxs, image_path, image_prefix, **train_kwds)
        descrps['train_size'] = len(train_idxs)
        descrps['test_size'] = len(test_idxs)
        descrps['val_size'] = len(val_idxs)
        descrps['label'] = label
        descrps['norm_method'] = d['norm_method']
        descrps.update(d)
        results_df = results_df.append(descrps, ignore_index=True)
        name = "_".join(['./results/results', label, str(train_size), runner_id, '.csv']) if recover_from == 0 else "_".join(['./results/results', label, str(train_size), runner_id, '.csv'])
        results_df.to_csv(name)
    return results_df
    
def get_options_for_section(section, train_size):
    import comp_params
    if section.lower() == 'small': 
        d = comp_params.nn_small
        if train_size <= 0.01:
            d['model_ds'].append(comp_params.krr_small['model_ds'])
    elif section.lower() == 'medium':
        d = comp_params.nn_medium
    elif section.lower() == 'large':
        d = comp_params.krr_large
        if train_size > 0.01:
            d['tokenization_ds'] = d['tokenization_ds'][2:] 
    elif section.lower() == 'custodi':
        d = comp_params.custodi
    elif section.lower() == 'rnn':
        d = comp_params.rnn
    return d['model_ds'], d['tokenization_ds']

def main():
    # set runner number
    # labels = ['rotational constant A [1.0]','rotational constant B [1.0]','rotational constant C [1.0]','dipole moment [Debye]','isotropic polarizability [Bohr ** 3]','homo [Hartree]','lumo [Hartree]','gap [Hartree]','electronic spatial extent [Bohr ** 2]','zpve [Hartree]','energy U0 [Hartree]','energy U [Hartree]','enthalpy H [Hartree]','free energy G [Hartree]','heat capacity Cv [1.0]']
    # train_sizes = [0.01, 0.1, 0.9]
    
    #tokenization_kw_dicts = [[{'tokenization_method': ['ECFP4']}],
    #                        [{'tokenization_method': ['word']}],
    #                        [{'tokenization_method': ['one_hot']}],
    #                        [{'tokenization_method': ['custodi'], 'degree': [1, 2], 'alpha': [0.01, 0.05]}],
    #                        [{'tokenization_method': ['aug_custodi'], 'degree': [1, 2], 'alpha': [0.01, 0.05]}],
    #                        [{'tokenization_method': ['cm']}],
    #                        [{'tokenization_method': ['random']}]]
    #train_kw_dicts = [{'model': ['KRR'], 'model_alpha': [0.01, 0.1], 'kernel': ['rbf']},
    #                    {'model': ['NN'], 'lr': [0.01, 0.1], 'dropout_rate': [0, 0.1]},
    #                    {'model': ['RNN'], 'lr': [0.01, 0.1], 'dropout_rate': [0, 0.1]}]
    
    print(sys.argv)
    label = sys.argv[1] # label for the calculation
    train_set_size = float(sys.argv[2]) # train set size 
    section = str(sys.argv[3]) # mem usage of the process
    runner_id = str(sys.argv[4]) # runner id - to separate different repeats.
    try:
      recover_from = int(sys.argv[5]) # set idx of kewords dict to recover from (writes results to the same csv file)
      if recover_from == -1:
          print("Computation is done !!!")
          sys.exit(0)
    except Exception:
      recover_from = 0
    val_size = 0.1

    train_kw_dicts, tokenization_kw_dicts = get_options_for_section(section, train_set_size)
    normalization_methods = ['z_score']

    results_df = pd.DataFrame()

    print(f"Running for {label}")
    
    train_idxs, val_idxs, test_idxs = gen_train_val_test_idxs(train_set_size, val_size, 111375 - 3, [], [])
    results_df = run_calc(None, train_idxs, val_idxs, test_idxs, label, results_df, normalization_methods, tokenization_kw_dicts, train_kw_dicts, runner_id, train_set_size, recover_from)
    print("Computation is done !!!")

if __name__ == '__main__':
    main()