import logging
from copy import deepcopy
from easydict import EasyDict
from naslib.evaluators.zc_ensemble_evaluator import ZCEnsembleTransferEvaluator
from naslib.predictors.ensemble import Ensemble
from naslib.search_spaces import get_search_space
from naslib.utils.get_dataset_api import get_dataset_api, get_zc_benchmark_api
from naslib.utils.logging import setup_logger
from naslib.utils import utils


# args = EasyDict({'config_file': '/home/alex/NASLib/configs/finetune_transfer_model_only_zc/24/13/nasbench201-9000/cifar10/config_9000.yaml',
#                  'datapath': None})
config = utils.get_config_from_args()

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)

utils.log_args(config)
# print(config)
search_space = get_search_space(config.search_space, config.dataset)
dataset_api = None #get_dataset_api(config.search_space, config.dataset)
zc_api = get_zc_benchmark_api(config.search_space, config.dataset)
search_space.instantiate_model = False
search_space.sample_without_replacement = True
search_space.labeled_archs = [eval(arch) for arch in zc_api.keys()]

utils.set_seed(config.seed)
train_loader, _, _, _, _ = utils.get_train_val_loaders(config)



ensemble = Ensemble(num_ensemble=1,
                    ss_type=search_space.get_type(),
                    predictor_type='adapt',
                    zc=config.zc_ensemble,
                    zc_only=config.zc_only,
                    config=config)


# target search space 
config.search_space = config.test_search_space
config.dataset = config.test_dataset
targ_loader, _, _, _, _ = utils.get_train_val_loaders(config)

targ_search_space = get_search_space(config.test_search_space, config.test_dataset)
targ_search_space.instantiate_model = False
targ_search_space.sample_without_replacement = True
targ_zc_api = get_zc_benchmark_api(config.test_search_space, config.test_dataset)
targ_search_space.labeled_archs = [eval(arch) for arch in targ_zc_api.keys()]



# get zc scores on train dataset
# get accuracies on train dataset

# get zc scores on test
# "" on test

# call fit with all 4 scores
# self.ensemble[0].fit()
# ensemble, zc_predictors = evaluator.fit(ensemble, train_loader)

evaluator = ZCEnsembleTransferEvaluator(
    n_train=config.train_size,
    n_test=config.test_size,
    zc_names=config.zc_names,
    zc_api=zc_api,
    # targ_search_space=targ_search_space,
    # targ_zc_api=targ_zc_api
)

evaluator.adapt_search_space(search_space, config.dataset, dataset_api, config)
xtrain, ytrain, train_models, train_info = evaluator.get_train_samples(ensemble, train_loader)

target_evaluator = ZCEnsembleTransferEvaluator(
    n_train=config.test_train_size,
    n_test=config.test_size,
    zc_names=config.zc_names,
    zc_api=targ_zc_api,
    # targ_search_space=targ_search_space,
    # targ_zc_api=targ_zc_api
)
target_evaluator.adapt_search_space(targ_search_space, config.test_dataset, dataset_api, config)
xtarg, ytarg, targ_models, targ_info = target_evaluator.get_train_samples(ensemble, targ_loader)
evaluator.fit_adaptation(ensemble, xtrain, ytrain, train_info, xtarg, ytarg, targ_info)

target_evaluator.test(ensemble, None, train_loader)
# get a new evaluator to train the adapted model
# target_ensemble = Ensemble(num_ensemble=1,
#                     ss_type=search_space.get_type(),
#                     predictor_type='adapt',
#                     zc=config.zc_ensemble,
#                     zc_only=config.zc_only,
#                     config=config)
# target_evaluator = ZCEnsembleTransferEvaluator(
#     n_train=config.test_train_size,
#     n_test=config.test_size,
#     zc_names=config.zc_names,
#     zc_api=zc_api
# )
# target_evaluator.adapt_search_space(search_space, config.test_dataset, dataset_api, config)
# target_ensemble, target_zc_predictors = target_evaluator.fit(target_ensemble, train_loader)
# target_evaluator.test(target_ensemble, target_zc_predictors, train_loader)