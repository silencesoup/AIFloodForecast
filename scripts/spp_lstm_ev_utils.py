import os
import pytest
import hydrodataset as hds
from torchhydro.configs.config import cmd, default_config_file, update_cfg
from torchhydro.trainers.trainer import set_random_seed
from torchhydro.trainers.deep_hydro import DeepHydro
from torchhydro.datasets.data_dict import data_sources_dict
from lstm.load_model import load_model
from lstm.lstm import normalization_test, split_data_test, split_windows_test
from datetime import datetime, timedelta
import xarray as xr
import hashlib
import random
import pandas as pd
import json
import numpy as np
import torch
from torch.autograd import Variable

# 转换函数
def convert_times(times):
    # 将字符串时间转换为datetime对象
    times = [datetime.strptime(time, "%Y-%m-%dT%H:%M:%S") for time in times]

    # 对时间列表中的每个时间应用所需的调整
    adjusted_times = []
    for i, time in enumerate(times):
        if i == 0:  # 对第一个时间减去8小时
            new_time = time - timedelta(hours=168)
        elif i == 1:  # 对第二个时间加上1小时
            new_time = time + timedelta(hours=23)
        # 将调整后的时间添加到列表中
        # 添加纳秒级的零
        adjusted_times.append(new_time.strftime("%Y-%m-%dT%H:%M:%S"))

    # print(adjusted_times)
    return adjusted_times


def create_json_from_ndarray(ndarray, start_time, filename):
    """
    将numpy数组转换为JSON格式，并保存到指定文件。

    :param ndarray: 包含z值的numpy数组。
    :param start_time: 开始时间，格式为'YYYY-MM-DD HH:MM:SS'。
    :param filename: 保存JSON数据的文件名。
    """
    # 将开始时间转换为datetime对象
    current_time = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")

    # 创建一个空列表来存储JSON对象
    data = []

    # 遍历数组中的每个元素
    for z in np.nditer(ndarray):
        # 创建一个字典来表示JSON对象
        json_object = {"z": str(z), "tm": current_time.strftime("%Y-%m-%d %H:%M:%S")}
        # 添加到列表中
        data.append(json_object)

        # 时间递增一小时
        current_time += timedelta(hours=1)

    # 将列表转换为JSON字符串并保存到文件
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def test_spp_lstm(
    project_name,
    train_period=None,
    test_period=None,
    gage_id=None,
    time_now=None,
    output_file_name=None,
):
    if project_name is None:
        project_name = "output/output"
    if train_period is None:
        train_period = ["2023-10-01T19:00:00", "2023-10-14T21:00:00"]
    if test_period is None:
        test_period = ["2023-10-01T15:00:00", "2023-10-14T20:00:00"]
    if gage_id is None:
        gage_id = ["4150377B"]
    if output_file_name is None:
        output_file_name = "output"
    config_data = default_config_file()
    random_seed = config_data["training_cfgs"]["random_seed"]
    set_random_seed(random_seed)
    data_cfgs = config_data["data_cfgs"]
    data_cfgs["data_source_name"] = "GPM_GFS"
    data_cfgs["download"] = False
    data_cfgs["data_path"] = os.path.join(hds.ROOT_DIR, "gpm_gfs_data")
    data_cfgs["model_path"] = "model"
    data_cfgs["shap_path"] = project_name
    data_cfgs["shap_name"] = output_file_name[0]
    config_data["data_cfgs"] = data_cfgs
    data_source_name = data_cfgs["data_source_name"]
    data_source = data_sources_dict[data_source_name](
        data_cfgs["data_path"], data_cfgs["download"]
    )

    if gage_id == ["4150377B"]:
        print("Yes!")

    # 将输入参数转换为字符串，并用这个字符串作为种子
    seed_str = "".join(test_period)
    seed = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest(), 16) % (
        10**10
    )  # 使用 SHA-256 散列并取前10位作为种子

    # 使用这个种子初始化随机数生成器
    random.seed(seed)

    # 定义时间范围
    min_possible_date = datetime(2022, 11, 1)
    max_possible_date = datetime(2023, 10, 1) - timedelta(days=8)  # 确保最大日期和最小日期之间的间隔为8天

    # 生成随机日期
    random_start_date = min_possible_date + timedelta(
        days=random.randint(0, (max_possible_date - min_possible_date).days)
    )

    # 创建列表，其中包含随机日期和随机日期加8天
    train_period = [
        random_start_date.strftime("%Y-%m-%dT%H:%M:%S"),
        (random_start_date + timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%S"),
    ]

    args = cmd(
        sub=project_name,
        source="GPM_GFS",
        source_path=os.path.join(hds.ROOT_DIR, "gpm_gfs_data"),
        # test_path = "/home/xushuolong1/AIFloodForecast/results/test_spp_lstm/swh/opt_Adam_lr_0.001_bsize_128",
        # source_path = "/home/xushuolong1/AIFloodForecast/gpm_gfs_data",
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
        gage_id=gage_id,
        # "4150233B",
        # "4150248B",
        # batch_size有一些限制，不能超过一个流域用于训练的item个数，比如1个流域只有6个item,batch_size需小于6
        batch_size=64,
        var_t=["tp"],
        var_out=["waterlevel"],
        dataset="GPM_GFS_Dataset",
        sampler="WuSampler",
        scaler="GPM_GFS_Scaler",
        train_epoch=2,
        save_epoch=1,
        te=1,
        # train_period=["2017-01-10", "2017-03-21"],
        # test_period=["2017-03-21", "2017-04-10"],
        # valid_period=["2017-04-11", "2017-04-28"],
        # train_period=["2017-01-10", "2017-01-15"],
        # test_period=["2017-03-21", "2017-03-21"],
        # valid_period=["2017-04-11", "2017-04-11"],
        # train_period=train_period,
        # test_period=["2023-09-01T15:00:00", "2023-09-12T20:00:00"],
        # valid_period=["2023-09-01T15:00:00", "2023-09-12T20:00:00"],
        train_period=["2023-10-01T00:00:00", "2023-10-13T21:00:00"],
        # test_period=["2023-09-01T15:00:00", "2023-09-01T16:00:00"],
        test_period=["2018-08-08T08:00:00","2018-08-08T08:00:00"],
        valid_period=["2023-10-01T00:00:00", "2023-10-13T21:00:00"],
        loss_func="RMSESum",
        opt="Adam",
        explainer="shap",
        lr_scheduler={1: 5e-4, 2: 1e-4, 3: 1e-5},
        which_first_tensor="sequence",
        # is_tensorboard=True,
    )
    # print(args)
    args.test_period = convert_times(args.test_period)
    print(args.test_period)
    update_cfg(config_data, args)
    model = DeepHydro(data_source, config_data)
    eval_log, preds_xr, obss_xr = model.model_evaluate()
    # print(preds_xr)

    if not os.path.exists(project_name):
        os.makedirs(project_name)

    # preds_xr.to_netcdf(project_name + "/" + output_file_name[0] + ".nc")


def lstm_gage(
    project_name="output/output",
    gage_id=None,
    test_period=None,
    output_file_name=None,
):
    if gage_id == ["4150377B"]:  # 开山
        model_name = "ks_model.pth"
    elif gage_id == ["4150248B"]:  # 嘀嗒
        model_name = "ddq_model.pth"
    elif gage_id == ["41521242"]:  # 石屋
        model_name = "swz_model.pth"
    elif gage_id == ["5111349B"]:  # 彭家
        model_name = "pjy_model.pth"
    else:
        raise ValueError("Invalid Gage Id")

    model = load_model("model/" + model_name)
    model.eval()
    seq_length = 168
    split_ratio = 0.9

    water_file_path = os.path.join("param/" + output_file_name[0] + "-z.json")
    # water_file_path = os.path.join("param/" + "test-z.json")
    # Read the generated JSON file
    with open(water_file_path, "r") as file:
        generated_data = json.load(file)

    # Convert to DataFrame with specified column names
    data = pd.DataFrame(generated_data)
    # data.rename(columns={"tm": "time", "z": "waterlevel"}, inplace=True)
    data.rename(columns={"tm": "time"}, inplace=True)
    # Converting the 'waterlevel' to numeric as it's currently a string
    data["waterlevel"] = pd.to_numeric(data["waterlevel"])
    data = data[["time", "waterlevel"]]
    data.set_index(["time"], inplace=True)
    data.fillna(method="ffill")

    data, mm_x = normalization_test(data)
    x = split_windows_test(data, seq_length)
    x_data=Variable(torch.Tensor(np.array(x)))
    # x_data, x_train, x_test = split_data_test(x, split_ratio)

    predict = model(x_data)
    data_predict = predict.data.cpu().numpy()
    data_predict = mm_x.inverse_transform(data_predict)

    # print(data_predict)
    # print(test_period[0])

    create_json_from_ndarray(
        data_predict, test_period[0], project_name + "/" + output_file_name[0] + ".json"
    )
