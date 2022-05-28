from pysurvival.models.survival_forest import ConditionalSurvivalForestModel
from pysurvival.models.survival_forest import RandomSurvivalForestModel
from pysurvival.models.survival_forest import ExtraSurvivalTreesModel
from pysurvival.models.semi_parametric import CoxPHModel
from pysurvival.models.semi_parametric import NonLinearCoxPHModel
from pysurvival.models.parametric import GompertzModel

csf_all = ConditionalSurvivalForestModel(num_trees=100)
rsf_all = RandomSurvivalForestModel(num_trees=100)
esf_all = ExtraSurvivalTreesModel(num_trees=100)
csf_select = ConditionalSurvivalForestModel(num_trees=100)
rsf_select = RandomSurvivalForestModel(num_trees=120)
esf_select = ExtraSurvivalTreesModel(num_trees=100)
coxph = CoxPHModel()
structure = [{'activation': 'BentIdentity', 'num_units': 150}, ]
nonlinear_coxph = NonLinearCoxPHModel(structure=structure)
gomp_model = GompertzModel()
