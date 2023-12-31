import os
import hydrodataset as hds
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import train_and_evaluate

def run_spp_lstm(
        project_name,
        train_period = None,
        test_period = None,
        gage_id = None,
):
    if project_name is None:
        project_name = "test_spp_lstm/ex2"
    if train_period is None:
        train_period = ["2017-01-03", "2017-01-03"]
    if test_period is None:
        test_period = ["2017-01-03", "2017-01-03"]
    if gage_id is None:
        gage_id = ["4150377B"],
    config_data = default_config_file()
    args = cmd(
        sub=project_name,
        source="GPM_GFS",
        source_path=os.path.join(hds.ROOT_DIR, "gpm_gfs_data"),
        source_region="US",
        download=0,
        ctx=[0],
        model_name="SPPLSTM",
        model_hyperparam={
            "seq_length": 168,
            "forecast_length": 24,
            "n_output": 1,
            "n_hidden_states": 80,
        },
        gage_id=[            
            "4150377B",
            "4150233B",
            "4150248B",
        ],
        # batch_size有一些限制，不能超过一个流域用于训练的item个数，比如1个流域只有6个item,batch_size需小于6
        batch_size=64,
        var_t=["tp"],
        var_out=["waterlevel"],
        dataset="GPM_GFS_Dataset",
        sampler="WuSampler",
        scaler="GPM_GFS_Scaler",
        train_epoch=10,
        save_epoch=1,
        te=10,
        # train_period=["2017-01-10", "2017-03-21"],
        # test_period=["2017-03-21", "2017-04-10"],
        # valid_period=["2017-04-11", "2017-04-28"],
        # train_period=["2017-01-10", "2017-01-15"],
        # test_period=["2017-03-21", "2017-03-21"],
        # valid_period=["2017-04-11", "2017-04-11"],
        train_period=["2017-01-10", "2017-01-15"],
        test_period=["2017-01-10", "2017-01-10"],
        valid_period=["2017-01-10", "2017-01-10"],
        loss_func="RMSESum",
        opt="Adam",
        # explainer="shap",
        lr_scheduler={1: 5e-4, 2: 1e-4, 3: 1e-5},
        which_first_tensor="sequence",
        # is_tensorboard=True,
    )
    update_cfg(config_data, args)
    train_and_evaluate(config_data)
