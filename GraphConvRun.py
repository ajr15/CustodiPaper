import settings
import deepchem as dc
from rdkit import Chem
import pandas as pd
from time import time
import sys; sys.path.append(settings.torina_parent_dir)
from Torina.Model.Model import Model
from commons import *
import sys

# wrapper around DeepChemModel to make a good comparision with other results
class DeepChemModel (Model):

    def __init__(self, dcmodel):
        self.model = dcmodel

    def train(self, data):
        dataset = dc.data.NumpyDataset(data.vectorized_inputs, y=data.vectorized_labels, ids=range(len(data.vectorized_labels)))
        self.model.fit(dataset)

    def predict(self, X):
        return self.model.predict_on_batch(X)

def data_prep(target_label, train_size, sample='all'):
    data = qm9_data(target_label, normalization_method='z_score', pad_smiles=False)
    if not sample == 'all':
        data = data.sample(sample)
    featurizer = dc.feat.ConvMolFeaturizer()
    # build list of SMILES strings
    mols = []
    idxs = []
    for i, s in enumerate(data.vectorized_inputs):
        mol = Chem.MolFromSmiles(''.join(s), sanitize=True)
        if mol is None:
            continue
        mols.append(mol)
        idxs.append(i)
    data = data.data_from_idxs(idxs)
    data.vectorized_inputs = featurizer.featurize(mols)

    if train_size < 0.8:
        return data.split_to_groups([train_size, 0.1], add_fill_group=True, random_seed=0)
    else:
        return data.split_to_groups([train_size, 0.05], add_fill_group=True, random_seed=0)

def run_fit(target_label, train_size,  sample='all'):
    train, val, test = data_prep(target_label, train_size, sample=sample)
    # stting up deep chem model
    dcmodel = DeepChemModel(dc.models.GraphConvModel(1, mode='regression'))
    # training
    dcmodel.train(train)
    # estimating fits
    d = dcmodel.estimate_fit(train, 
                        estimators=['r_squared', 'rmse','mae', 'mare'], 
                        prefix='train_')
    d.update(dcmodel.estimate_fit(val, 
                        estimators=['r_squared', 'rmse','mae', 'mare'], 
                        prefix='train_'))
    d.update(dcmodel.estimate_fit(test, 
                        estimators=['r_squared', 'rmse','mae', 'mare'], 
                        prefix='train_'))
    d.update({'train_size': train_size, 'label': target_label})
    return d

def main_run():
    counter = 1
    c = pd.DataFrame()
    ti = time()
    for label in ["dipole moment [Debye]", "gap [Hartree]"]:
        for train_size in [0.1]:
            print("Running for {} and {}% train size (~{} samples)".format(label, round(train_size * 100), round(1.1e5 * train_size)))
            c = c.append(run_fit(label, train_size, sample=100))
            counter += 1
    # compute(c, scheduler='distributed')
    c.to_csv('./Results/GCResults.csv')
    tf = time()
    print("Computation is Done ! Computation time:", tf - ti, "seconds")

def test():
    print("1")
    tasks, datasets, transformers = dc.molnet.load_lipo()
    print("2")
    n_tasks = len(tasks)
    model = dc.models.GraphConvModel(n_tasks, mode='classification')
    print("3")
    model.fit(train_dataset, nb_epoch=50)
    print("4")

def main():
    main_run()

if __name__ == '__main__':
    main()