# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + colab={"base_uri": "https://localhost:8080/"} id="Y7JH7N8ZoORX" outputId="3f36464d-f074-47b9-aaa9-e61c8969ec89"
# gpu_info = !nvidia-smi
# gpu_info = '\n'.join(gpu_info)
# if gpu_info.find('failed') >= 0:
#     print('Not connected to a GPU')
# else:
#     print(gpu_info)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# + id="lhBKW8YaoOOg"
need_to_install = False

# + [markdown] id="NmROrJOj9Hkt"
# # Install and Import

# + id="HkqDgz7s9LJw"
# if need_to_install:
  # !pip install wandb==0.13.3
  # !pip install transformers
  # !pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'

# + id="nS-7Qab59LHK"
import sys
code_path = 'code/'

sys.path.insert(0, code_path)

# + id="o7UyFizG9LEn"
import os
import warnings
warnings.filterwarnings("ignore")

import wandb
import torch
import pandas as pd

import matplotlib.pyplot as plt

from GISLR_utils.utils import get_logger, class2dict
from GISLR_utils.pipeline import train_loop, full_train_loop


# + [markdown] id="PK3WyBt-9xcY"
# # Config

# + id="W90Jxds_oOLp"
# ====================================================
# Config
# ====================================================
class CFG:
    ####################
    # MAIN
    ####################
    '''
    wandb = True
    wandb_project = 'GISLR_IMG_OPTUNA'
    competition = 'G_ISLR_Kaggle'
    wb_group = None
    exp_name = 'exp1'
    base_path = '/home/sphuang/Downloads/sign_language/input/'

    seed = 333

    train = True
    LOOP = False
    full_train = True
    debug = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ####################
    # DATA
    ####################
    dataset = 'img_80_mixup'

    num_workers = 12
    train_bs = 64
    valid_bs = 256
    n_fold = 8
    trn_fold = [0]
    fold_group = False

    ####################
    # TRAIN
    ####################

    early_stopping = 40  # None for dont use
    use_early_break = None # None for dont use

    FULL_TRAIN = False
    apex = True

    eval_after = 0
    eval_every = 1
    eval_always_after = 1

    finetune = False
    finetune_path = 'PATH/TO/CKPT'
    finetune_fold = 0
    finetune_sched_opt = True
    finetune_epoch = 2
    finetune_change_seed = True

    # Scheduler step 1

    scheduler = 'onecycle'
    onecycle_start = 0.1
    onecycle_m = 1.
    num_cycles = 0.5
    num_warmup_steps = 333

    # Loop step 1

    epochs = 180

    # LR, optimizer step 1

    eps = 1e-8
    betas = (0.9, 0.999)
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    optimizer = "RAdam"

    data_dir = base_path + '/asl-signs/'
    BREAK_EPOCH = 100000
    fill_nan_value = 0.

    new_size= (160, 80, 3)
    encoder = 'rexnet_100'

    COLAB = False
    '''
    wandb=True
    wandb_project='GISLR_IMG_OPTUNA'
    competition='G_ISLR_Kaggle'
    wb_group=None
    exp_name='exp3'
    # base_path='/home/jeff/project/poc-project/sign_translate/holistic/part_time/m1_m2/'
    base_path='/home/jovyan/project/poc-project/sign_translate/holistic/part_time/m1_m2/'
    seed=1223
    train=True
    LOOP=False
    full_train=True
    debug=False
    device='cuda'
    use_external=True
    external_multiplier=2
    dnt_use_ext_after=999
    hidden_dim=256
    input_size=345
    dataset='img_80_mixup'
    use_meta=False
    num_workers=12
    train_bs=64
    valid_bs=256
    n_fold=8
    trn_fold=[0]
    fold_group=False
    aug_prob=0.24189489872688524
    invert_prob=0.22482221835067367
    drop_rate=0.105507794097595
    # num_classes=250
    # num_classes=391
    num_classes=282
    early_stopping=None
    use_early_break=None
    FULL_TRAIN=False
    apex=True
    eval_after=0
    eval_every=1
    eval_always_after=1
    finetune=False
    # finetune=True
    finetune_path=''
    finetune_fold=0
    finetune_sched_opt=True
    finetune_epoch=182
    finetune_change_seed=True
    k=3
    alpha=0.3
    gamma=2
    onecycle_start=0.1
    onecycle_m=1.0
    num_cycles=0.5
    num_warmup_steps=333
    epochs=300
    use_restart=False
    rest_thr_=0.9
    rest_epoch=3
    iter4eval=4
    lr=0.002705471149867012
    min_lr=1e-06
    eps=1e-08
    betas=(0.9, 0.999)
    weight_decay=0.01
    gradient_accumulation_steps=1
    optimizer="RAdam"
    data_dir = base_path + '/asl-signs/'
    BREAK_EPOCH=100000
    fill_nan_value=0.0
    new_size=(160, 80, 3)
    encoder='rexnet_100'
    COLAB=False
    use_loss_wgt=False
    pw_bad=0.8219710950094926
    pw_com=1.3979208248868304
    deep_supervision=False
    zero_prob=0.0
    deal_with_len=False
    moreN=False
    drop_bad_inds=False
    restart_from=11111110
    interp_nearest_random=0.3761024354307254
    tree_rot_prob=0.3738408998077264
    interpol_prob=0.3738408998077264
    normalize=True
    rotate_prob=0.16521647559289365
    replace_prob=0.23897547462598495
    mixup_prob=0.4874750221165106
    mixup_alpha=1.2591818589682875
    shift_prob=0.20489588604493592
    scale_prob=0.31396992296421367
    img_masking=0.98
    freq_m=80
    time_m=18
    motion=False
    use_swa=False
    swa_start=0
    label_smooth=0.5126793416811322
    train_fold = list(range(18))
    valid_fold = list(range(18))

if False:
    os.makedirs(CFG.base_path + 'results/', exist_ok=True)
    os.makedirs(CFG.base_path + 'results/' + CFG.exp_name, exist_ok=True)
    os.makedirs(CFG.base_path + 'results/' + CFG.exp_name + '/checkpoints', exist_ok=True)
    CFG.save_path = CFG.base_path + 'results/' + CFG.exp_name + '/checkpoints/'
    with open(CFG.base_path + 'results/' + CFG.exp_name + '/CFG.txt', 'w') as f:
        for key, value in CFG.__dict__.items():
            f.write('%s:%s\n' % (key, value))


# + [markdown] id="zczl4yoR_ewA"
# # Load and Prepare Data

# + colab={"base_uri": "https://localhost:8080/", "height": 0} id="tDQu7Mhl_g75" outputId="86061634-ec46-4098-8c79-9a602d7e1d24"
import json
def read_dict(file_path):
    path = os.path.expanduser(file_path)
    with open(path, "r") as f:
        dic = json.load(f)
    return dic

train = pd.read_csv(CFG.base_path + 'asl-signs/train.csv')
# label_index = read_dict(f"{CFG.base_path}/asl_signs/sign_to_prediction_index_map.json")
# index_label = dict([(label_index[key], key) for key in label_index])
# train["label"] = train["sign"].map(lambda sign: label_index[sign])
print(train.shape)
# display(train.head())


import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

# if CFG.fold_group:
#     print(f'FOLD SPLIT USING GROUPS')
#     split = StratifiedGroupKFold(CFG.n_fold, random_state=42, shuffle=True) #rs = 42

#     for k, (_, test_idx) in enumerate(split.split(train, train.sign, groups=train.participant_id)):
#         train.loc[test_idx, 'fold'] = k
# else:
#     print(f'FOLD SPLIT ONLY ON SIGN')
#     split = StratifiedKFold(CFG.n_fold, random_state=42, shuffle=True) #rs = 42

#     for k, (_, test_idx) in enumerate(split.split(train, train.sign)):
#         train.loc[test_idx, 'fold'] = k

train.fold = train.fold.astype(int)
# display(train.groupby('fold').size())


# + [markdown] id="gKNYHmanBMBi"
# # Training

# +
import optuna

def objective(trial):
    """
    Function to optimize model params
    :param trial:(optuna instance) iteration
    :return:(float) metric of model iteration
    """

    # CFG.new_size = (128, 120, 3)
    param = {
        'seed': np.random.randint(20, 10000), #6374,
        # 'aug_prob': trial.suggest_float('aug_prob', 0.15, 0.25), # do_random_affine prob
        # 'invert_prob': trial.suggest_float('invert_prob', 0.25, 0.32), # it flips all points (hands, lips, pose)
        # 'scale_prob': trial.suggest_float('scale_prob', 0.17, 0.3), # prob to rescale some parts (e.g. one hand or hand and lips)
        'lr': 2e-2, # trial.suggest_float('lr', 2e-3, 2.8e-3),  # LR
        # 'train_bs': trial.suggest_categorical('train_bs', [32, 64]),  # BS
        'drop_rate': trial.suggest_float('drop_rate', 0.2, 0.7, step=0.1), # model dropout
        # 'epochs': trial.suggest_int('epochs', 100, 210), # epochs
        # 'img_masking': 0.98, # prob to use torchaudio masks
        'model': trial.suggest_categorical('model', ['img_v0', 'img_v0_b1', 'effb0_timm', 'v2_b1', 'timm',
                                                     'img_v0_b2', 'img_v0_b3', 'img_v0_b4', 'img_v0_b5']) , #'img_v0', # model name
        # 'dataset': 'img_80_mixup', # img_80_onehand for one hand new_size=(T, 64, 3), img_80_mixup for 2 hands new_size=(T, 80, 3)
        # 'freq_m': trial.suggest_int('freq_m', 66, 80), # mask for time axis  (yes name freq but mask time :) )
        # 'time_m': trial.suggest_int('time_m', 3, 12), # mask some points. the max possible masked points
        # 'use_loss_wgt': True, # use wieghted loss
        # 'pw_bad': 1.45, # power to bad predicted classes
        # 'pw_com': 0.81, # power to classes which has common classes (e.g. dad, grandpa, grandma)
        # "label_smooth": trial.suggest_float('label_smooth', 0.50, 0.65), # loss label smoothing
        # 'shift_prob': trial.suggest_float('shift_prob', 0.12, 0.29),  # shift random parts to random value (one or few)
        # 'mixup_prob': trial.suggest_float('mixup_prob', 0.3, 0.42),  # mixup prob
        # 'zero_prob': 0., # pixel dropout prob. It didn't work
        # 'rotate_prob': trial.suggest_float('rotate_prob', 0.18, 0.26),  # rotate one or few parts
        # 'replace_prob': trial.suggest_float('replace_prob', 0.08, 0.17),  # replace one or two parts from another element with same class
        # 'deep_supervision': False, # DSV
        # 'interpol_prob': trial.suggest_float('interpol_prob', 0.15, 0.4), # interploation as in Carno' code
        # 'normalize': True, # mean std normalization
        # 'tree_rot_prob': trial.suggest_float('tree_rot_prob', 0.25, 0.55), # finger tree augmentation from Carno's Code
        # 'interp_nearest_random': trial.suggest_float('interp_nearest_random', 0.35, 0.5),
        # 'lookahed_k':trial.suggest_int('lookahed_k', 2, 7),
        # 'lookahed_alpha':trial.suggest_float('lookahed_alpha', 0.3, 0.6),
        'loss': trial.suggest_categorical('loss', ['ce', 'focal']),
    }
    print(param)
    # CFG.tree_rot_prob = param['tree_rot_prob']
    # CFG.interpol_prob = param['interpol_prob']
    # CFG.normalize = param['normalize']
    # CFG.rotate_prob = param['rotate_prob']
    # CFG.zero_prob = param['zero_prob']
    # CFG.replace_prob = param['replace_prob']

    # CFG.deep_supervision = param['deep_supervision']
    # CFG.mixup_prob = param['mixup_prob']
    # CFG.shift_prob = param['shift_prob']
    # CFG.scale_prob = param["scale_prob"]
    # CFG.use_loss_wgt = param['use_loss_wgt'] #False
    # CFG.pw_bad = param['pw_bad']
    # CFG.pw_com = param['pw_com']
    # CFG.img_masking = param["img_masking"]
    # CFG.freq_m = param["freq_m"]
    # CFG.time_m = param["time_m"]
    CFG.scheduler = 'onecycle'
    CFG.new_size = (160, 80, 3)
    # CFG.loss = 'ce'
    CFG.loss = param['loss']
    CFG.alpha = 0.3
    CFG.model = param['model']
    #CFG.encoder = 'rexnet_100'  # if model == 'timm'
    # CFG.aug_prob = param['aug_prob']
    # CFG.invert_prob = param['invert_prob']
    # CFG.train_bs = param['train_bs']
    CFG.drop_rate = param['drop_rate']
    CFG.lr = param['lr']
    # CFG.epochs = param['epochs']
    CFG.num_cycles = 0.5 # param['num_cycles']
    # CFG.dataset = param['dataset']
    CFG.seed = param['seed']
    CFG.optimizer = 'Lookahead_RAdam'  # param['optimizer']
    CFG.motion = False
    CFG.use_swa = False
    CFG.swa_start = 0
    # CFG.label_smooth = param['label_smooth']

    CFG.trn_fold = [0]
    fold_ = CFG.trn_fold[0]

    CFG.exp_name = f'EXP_NAME_f{fold_}_{trial.number}_m{CFG.model}_d{CFG.drop_rate}_s{CFG.seed}_bs{CFG.train_bs}_lr{CFG.lr:8f}_ep{CFG.epochs}'

    os.makedirs(CFG.base_path + 'results/', exist_ok=True)
    os.makedirs(CFG.base_path + 'results/' + CFG.exp_name, exist_ok=True)
    os.makedirs(CFG.base_path + 'results/' + CFG.exp_name + '/checkpoints', exist_ok=True)
    CFG.save_path = CFG.base_path + 'results/' + CFG.exp_name + '/checkpoints/'
    with open(CFG.base_path + 'results/' + CFG.exp_name + '/CFG.txt', 'w') as f:
        for key, value in CFG.__dict__.items():
            f.write('%s:%s\n' % (key, value))

    wandb.init(project='TRAIN_2ND_CNN_MODELS',
            name=CFG.exp_name,
            config=class2dict(CFG),
            group=CFG.wb_group,
            job_type="train",
            dir=CFG.base_path)

    LOGGER = get_logger(CFG.base_path + 'results/' + CFG.exp_name + f'/train_f{fold_}')
    acc, topk = train_loop(CFG, train, fold_, CFG.train_fold, CFG.valid_fold, LOGGER)
    print(f'FOR PARAMS: {param}')
    print(f'Accuracy: {acc}')
    print(f'TOPK: {topk}')
    print()
    return acc

print('Starting train parameters optimization process.\n'
          f'With main metric Accuracy')
optuna.logging.disable_default_handler()
direct = 'maximize'
study = optuna.create_study(direction=direct)
study.optimize(objective, n_trials=128)

model_params = study.best_trial.params
print('Best params:')
print(model_params, '\n')

