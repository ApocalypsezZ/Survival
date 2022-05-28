from pysurvival.utils.metrics import concordance_index
from pysurvival.utils.display import integrated_brier_score
from config import *
from utils import *
from models import *
from pysurvival.utils import save_model

# Load data
X_all_train, X_all_test, X_select_train, X_select_test, T_train, T_test, E_train, E_test = load_data(config['path_ok'])

# Load model
csf_all.fit(X_all_train, T_train, E_train, max_features='sqrt',
            max_depth=1000, min_node_size=12, alpha=0.05, minprop=0.1)
rsf_all.fit(X_all_train, T_train, E_train,
            max_features="sqrt", max_depth=1000, min_node_size=10)
esf_all.fit(X_all_train, T_train, E_train,
            max_features="sqrt", max_depth=1000, min_node_size=20,
            num_random_splits=1000)
coxph.fit(X_all_train, T_train, E_train, lr=0.5, l2_reg=1e-2, init_method='zeros')
nonlinear_coxph.fit(X_all_train, T_train, E_train, lr=1e-3, init_method='xav_uniform')
gomp_model.fit(X_all_train, T_train, E_train, lr=1e-2, init_method='zeros',
               optimizer='adam', l2_reg=1e-3, num_epochs=2000)

# Cross Validation / Model Performances
for i in [csf_all, rsf_all, esf_all, coxph, nonlinear_coxph, gomp_model]:
    print(i)
    print("C-index of {} training set:{}".format(i, concordance_index(i, X_all_train, T_train, E_train)))
    print("C-index of {} test set:{}".format(i, concordance_index(i, X_all_test, T_test, E_test)))
    print("Integrated Brier Score of {} training set:{}".format(i, integrated_brier_score(i, X_all_train, T_train, E_train, t_max=100, figure_size=(20, 6.5))))
    print("Integrated Brier Score of {} test set:{}".format(i, integrated_brier_score(i, X_all_test, T_test, E_test, t_max=100, figure_size=(20, 6.5))))

# save model
save_model(csf_all, config["save_path"] + "csf_all.zip")
save_model(rsf_all, config["save_path"] + "rsf_all.zip")
save_model(esf_all, config["save_path"] + "esf_all.zip")
save_model(coxph, config["save_path"] + "coxph.zip")
save_model(nonlinear_coxph, config["save_path"] + "nonlinear_coxph.zip")
save_model(gomp_model, config["save_path"] + "gomp_model.zip")
