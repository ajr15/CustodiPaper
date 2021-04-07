import sys; sys.path.append('/home/shachar/Documents')
import numpy as np
import os
from copy import copy 
import warnings; warnings.simplefilter(action = 'ignore', category=FutureWarning)
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

from Torina.Data.Objects import SMILES
from Torina.Data.Base import flatten, generate_data_using_comp_protocol
from Torina.Data.utils import *
from Torina.Model.utils import KerasArchitectures
from Torina.Model.Objects import *

def get_trained_KRR(train_inputs, train_labels, alpha=0.01, kernel='gaussian'):
    model = KernalRidge(alpha, kernel=kernel)
    model.train(train_inputs, train_labels)
    return model

def get_trained_pred_nn(input_shape, train_inputs, train_labels, lr=0.001, nn_depth=3, epochs=500):
    '''Train a NN for property prediction. Returns trained KerasNN'''
    input_size = np.prod(input_shape)
    
    model_layers = [tf.keras.layers.Flatten()]
    for i in range(nn_depth - 1):
        model_layers = model_layers + [tf.keras.layers.Dropout(0.1), tf.keras.layers.Dense(round(input_size / nn_depth) * (i + 1), activation='relu')]
    model_layers = model_layers + [tf.keras.layers.Dense(1, activation='tanh'), tf.keras.layers.Dense(1, activation='linear')]
    
    predictive_model = KerasArchitectures.GeneralANN(input_shape=input_shape, layers=model_layers)
    mape = tf.keras.losses.MeanAbsolutePercentageError(reduction='sum_over_batch_size')
    mse = tf.keras.losses.MSE
    NN = KerasNN(predictive_model, tf.keras.optimizers.Adam(lr=lr), mse)
    NN.train(train_inputs, train_labels, epochs=epochs, verbose=1)
    return NN

def pca_reduce(inputs, encoding_size=20):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=encoding_size)
    return pca.fit_transform(inputs)

def autoencoder_reduce(inputs, encoding_size=20, n_layers=3, lr=0.01, epochs=150):
    ae = KerasArchitectures.AutoEncoders()
    # getting AE size params
    input_shape = np.array(inputs[0]).shape
    size_step = int(np.floor((np.prod(input_shape) - encoding_size) / n_layers))
    # setting encoder and decoder
    encoding_layers = [tf.keras.layers.Flatten()] + [tf.keras.layers.Dense(np.prod(input_shape) - i * size_step, activation='relu') for i in range(n_layers - 1)]
    encoding_layers += [tf.keras.layers.Dense(encoding_size, activation='relu')]
    ae.set_encoder(input_shape=input_shape, layers=encoding_layers)
    decoding_layers = [tf.keras.layers.Dense(encoding_size + (i + 1) * size_step, activation='relu') for i in range(n_layers - 1)]
    decoding_layers += [tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid'), 
                        tf.keras.layers.Dense(np.prod(input_shape), activation='linear'), 
                        tf.keras.layers.Reshape(input_shape)]
    ae.set_decoder(layers=decoding_layers)
    # building autoencoder model - compiling and training
    AE = KerasAE(ae, tf.keras.optimizers.Adadelta(lr=lr), tf.keras.losses.binary_crossentropy)
    AE.train(np.array(inputs), epochs=epochs, verbose=1)
    return AE.encode(np.array(inputs))

def reduce_inputs(inputs, method, **kwargs):
    reduce_methods_dict = {
        'pca': pca_reduce,
        'ae': autoencoder_reduce
    }

    if not method.lower() in reduce_methods_dict.keys():
        raise ValueError(f"{method} is not an allowed method. Allowed methods are {', '.join(list(reduce_methods_dict.keys()))}")
    return reduce_methods_dict[method.lower()](inputs, **kwargs)

def costodi_descrps(costodi_inputs, train_idxs, test_idxs, labels):
    '''Method to check the fit of costodi on train and test sets. takes inputs that were tokenized by costodi. returns descrps dicts'''
    pred = [sum(vec) for vec in costodi_inputs]
    descrps = calc_descps([pred[i] for i in train_idxs], np.array([labels[i] for i in train_idxs]), 'costodi_train_')
    descrps.update(calc_descps([pred[i] for i in test_idxs], np.array([labels[i] for i in test_idxs]), 'costodi_test_'))
    return descrps

def select_idxs(max_idx, sample_size):
    idxs = []
    while True:
        i = np.random.randint(0, max_idx)
        if not i in idxs:
            idxs.append(i)
        if len(idxs) == sample_size:
            break
    return idxs

def base_data_prep(label, sample_size):
    df = pd.read_csv('./database.csv')
    if not sample_size == None:
        idxs = select_idxs(len(df) - 1, sample_size)
        df = df.iloc[idxs, :]
    data = SMILES()
    data.load_inputs_from_text('\n'.join(df['SMILES'].values), sep='chars')
    data.vectorized_labels = df[label].values
    data.vectorized_inputs = data.pad_data("_", " ", pad="inputs")
    return data

def data_prep(labeled_data, train_idxs, test_idxs, tokenization_method, compressor=None, normalization_method='z_score', compressor_kwargs={}):
    '''making a data object and costodi fitting parameters'''
    data = SMILES()
    data.vectorized_inputs = copy(labeled_data.vectorized_inputs)
    data.vectorized_labels = copy(labeled_data.vectorized_labels)
    data.noramlize_vectors(normalize='labels', method=normalization_method)
    # check for sample quality
    val = data.vectorized_labels[0]
    if all([v == val for v in data.vectorized_labels]):
        raise RuntimeError("Bad Data Sampling! All samples equal to {}".format(val))
    plt.figure()
    plt.hist(data.vectorized_labels, bins=10)
    plt.savefig(os.path.join('./plots', 'label_hist.png'))
    print("setting tokenization function...")
    if tokenization_method == 'custodi':
        data.set_custodi_tokenization_func(use_idxs=train_idxs)
    elif tokenization_method == 'word':
        data.set_word_tokenization_func()
    else:
        raise ValueError("Unrecognized tokenization method. Allowed methods \'custodi\', \'word\'")
    print('tokenizing...')
    data.vectorized_inputs = data.tokenize('vectorized_inputs', keep_shape=False)
    if tokenization_method == 'custodi':
        costodi_fit_descrps = costodi_descrps(data.vectorized_inputs, train_idxs, test_idxs, data.vectorized_labels)
    else:
        costodi_fit_descrps = {}
        data.noramlize_vectors(normalize='inputs', method=normalization_method)
    if not compressor == 'None':
        data.vectorized_inputs = reduce_inputs(data.vectorized_inputs, compressor, **compressor_kwargs)
    return data, costodi_fit_descrps

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
    mare_func = lambda p, t: np.abs((p - t) / t)
    descps[prefix + 'mare'] = _safe_calc_diff_with_func(pred, true, mare_func)
    # corrects shape of descrps values in case a numpy array is returned
    for k, v in descps.items():
        try:
            descps[k] = v[0]
        except IndexError:
            continue
    return descps

def run_ml(labeled_data, train_idxs, test_idxs, image_path, prefix, lr=0.001, nn_depth=3, epochs=500):
    print("running ML...")
    input_shape = np.array(labeled_data.vectorized_inputs[0]).shape
    # NN = get_trained_pred_nn(input_shape, np.array([labeled_data.vectorized_inputs[i] for i in train_idxs]), 
                                                    # np.array([labeled_data.vectorized_labels[i] for i in train_idxs]), 
                                                    # lr, nn_depth, epochs)
    NN = get_trained_KRR(np.array([labeled_data.vectorized_inputs[i] for i in train_idxs]), 
                            np.array([labeled_data.vectorized_labels[i] for i in train_idxs]), 
                            alpha=0.01, kernel='laplacian')
    train_pred = NN.predict(np.array([labeled_data.vectorized_inputs[i] for i in train_idxs]))
    test_pred = NN.predict(np.array([labeled_data.vectorized_inputs[i] for i in test_idxs]))
    print("calculating descriptors...")
    NN_descrps = calc_descps(train_pred, np.array([labeled_data.vectorized_labels[i] for i in train_idxs]), 'NN_train_', labeled_data._norm_params)
    NN_descrps.update(calc_descps(test_pred, np.array([labeled_data.vectorized_labels[i] for i in test_idxs]), 'NN_test_', labeled_data._norm_params))
    NN.plot_fit(np.array([labeled_data.vectorized_inputs[i] for i in train_idxs]), [labeled_data.vectorized_labels[i] for i in train_idxs], show=False, alpha=0.5) # taining set
    plt.gcf()
    plt.savefig(os.path.join(image_path, prefix + '_train.png'))
    NN.plot_fit(np.array([labeled_data.vectorized_inputs[i] for i in test_idxs]), [labeled_data.vectorized_labels[i] for i in test_idxs], show=False, alpha=0.5) # testing set
    plt.gcf()
    plt.savefig(os.path.join(image_path, prefix + '_test.png'))
    return NN_descrps

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

def run_calc(labeled_data, train_idxs, test_idxs, results_df, label_name,
                tokenization_method=[], 
                compressor=[], 
                normalization_method=[],
                compressor_kwargs=[],
                lr=[],
                nn_depth=[],
                epochs=[]):
    '''Method to run all computation for a set of parameters writen in keywords. returns a dictionary with fitting parameters. exports fitting plots'''
    kwargs_dict = {'tokenization_method': tokenization_method, 
                'compressor': compressor, 
                'normalization_method': normalization_method,
                'compressor_kwargs': compressor_kwargs,
                'lr': lr,
                'nn_depth': nn_depth,
                'epochs': epochs}
    # changing keywords for image export (changing keywords will apear in dir name)
    changing_kwargs = []
    for k, v in kwargs_dict.items():
        if len(v) > 1 and not k == 'tokenization_method':
            changing_kwargs.append(k)
    
    base_image_dir = './plots'
    if not os.path.isdir(base_image_dir):
        os.mkdir(image_dir)
    
    for d in kw_cartesian_prod(kwargs_dict):
        print("Running calculation with:", 
            " ".join([k + ": " + v  + ',' for k, v in d.items() if k in changing_kwargs] + ["tokenization method: {}".format(d['tokenization_method'])]))
        data, descrps = data_prep(labeled_data, train_idxs, test_idxs, d['tokenization_method'], 
                                            d['compressor'], d['normalization_method'], d['compressor_kwargs'])
        descrps.update(d)
        image_path = os.path.join(base_image_dir, '_'.join([f'label_{label_name}'] + [k + "_" + str(v) for k, v in d.items() if k in changing_kwargs]))
        if not os.path.isdir(image_path):
            os.mkdir(image_path)
        image_prefix = 'tokenization_method_{}_train_size_{}'.format(d['tokenization_method'], len(train_idxs))
        NN_descrps = run_ml(data, train_idxs, test_idxs, image_path, image_prefix, d['lr'], d['nn_depth'], d['epochs'])
        descrps.update(NN_descrps)
        results_df = results_df.append(descrps, ignore_index=True)
        results_df.to_csv('./results.csv')
    return results_df

def gen_train_test_idxs(train_size, data_size, train_idxs):
    print("setting train idxs...")
    while True:
        if len(train_idxs) == round(train_size * data_size):
            break
        i = np.random.randint(0, data_size)
        if not i in train_idxs:
            train_idxs.append(i)
    print("setting test idxs...")
    test_idxs = [i for i in range(data_size) if not i in train_idxs]
    return train_idxs, test_idxs

def main(sample_size):
    image_dir = './plots'
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)
    # labels = ['dipole moment' ,'isotropic polarizability','homo' ,'electronic spatial extent' ,'heat capacity']
    labels = ['isotropic polarizability']
    results = pd.DataFrame()
    #train_sizes = [0.001, 0.005, 0.01, 0.05, 0.1]
    train_sizes = [0.1]
    train_idxs = []
    for label in labels:
        labeled_data = base_data_prep(label, sample_size)
        for train_size in train_sizes:
            train_idxs, test_idxs = gen_train_test_idxs(train_size, len(labeled_data.inputs), train_idxs)
            print("Running ML..")
            results = run_calc(labeled_data, train_idxs, test_idxs, results, label,
                        tokenization_method=['custodi', 'word'], 
                        compressor=["None"],
                        normalization_method=['z_score'],
                        # compressor_kwargs=[{'encoding_size': 7, 'epochs': 20, 'n_layers': 3, 'lr': 0.001}],
                        compressor_kwargs=[{'encoding_size': 10}],
                        lr=[5e-4],
                        nn_depth=[3],
                        epochs=[1000])

if __name__ == '__main__':
    main(15000)
    # data = base_data_prep('dipole moment', 15000)
    # train_idxs, test_idxs = gen_train_test_idxs(0.1, len(data.inputs), [])
    # data = data_prep(data, train_idxs, test_idxs, 'custodi', 'AE', normalization_method='unit_scale', compressor_kwargs={'epochs': 10, 'n_layers': 3, 'lr': 0.001})
    
