import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ===================
#   TORINA SETTINGS
# ===================

torina_parent_dir = '/home/shachar/Documents/'

# ================
#   QM9 SETTINGS
# ================

qm9_csv_file = '/home/shachar/Documents/CustodiPaper/DataFiles/QM9.csv'
qm9_cm_file = '/home/shachar/Documents/CustodiPaper/DataFiles/Qm9Cm.csv'

# we commented-out some of the labels for performance, totally 9 properties
qm9_labels = [  # "SMILES",
                # "# of atoms",
                # "gdb entry",
                # "rotational constant A [1.0]",
                # "rotational constant B [1.0]",
                # "rotational constant C [1.0]",
                "dipole moment [Debye]",
                "isotropic polarizability [Bohr ** 3]",
                "homo [Hartree]",
                "lumo [Hartree]",
                "gap [Hartree]",
                "electronic spatial extent [Bohr ** 2]",
                "zpve [Hartree]",
                "energy U0 [Hartree]",
                # "energy U [Hartree]",
                # "enthalpy H [Hartree]",
                # "free energy G [Hartree]",
                "heat capacity Cv [1.0]"]

qm9_train_sizes = [0.001, 0.01, 0.1, 0.9] # [0.1%, 1%, 10%, 90%]

# ====================
#   Model parameters
# ====================

import sys; sys.path.append(torina_parent_dir)
from commons import CustomNN
from Torina.Model.Custodi import Custodi

models_dict = {"CUSTODI": Custodi,
                "NN": CustomNN}

# init parameters for the different models
model_params = {
                "CUSTODI": #{'degree':   [1, 2, 3, 4],
                           #'alpha':    [0.01, 1, 100, 1e4],
                           {'degree':   [1],
                            'alpha':    [0.01],
                            'max_iter': [10000]},
                "NN":      {"init": {'lr': [0.01, 0.1], 
                                    'dropout_rate': [0, 0.1]},
                            "train": {'epochs': 10,
                                        'batch_size': 128}}

                }
