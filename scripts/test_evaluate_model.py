"""
Author: Wenyu Ouyang
Date: 2022-04-27 10:54:32
LastEditTime: 2023-04-23 11:37:16
LastEditors: Wenyu Ouyang
Description: Generate commands to run scripts in Linux Screen
FilePath: /HydroSPDB/scripts/train_model.py
Copyright (c) 2021-2022 Wenyu Ouyang. All rights reserved.
"""
import argparse
import os
from pathlib import Path
import sys
from spp_lstm_utils import run_spp_lstm
from spp_lstm_ev_utils import test_spp_lstm, lstm_gage

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" ###指定此处为-1即可

# from scripts.torchhydro.configs.config import default_config_file
# from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(Path(os.path.abspath(__file__)).parent))

# def train_and_test(args):
#     project_name = args.project_name
#     train_period = args.train_period
#     test_period = args.test_period
#     gage_id = args.gage_id
#     run_spp_lstm(
#         project_name = project_name,
#         train_period=train_period,
#         test_period=test_period,
#         gage_id = gage_id,
#     )


def test_model(args, time_now):
    project_name = args.project_name
    train_period = args.train_period
    test_period = args.test_period
    gage_id = args.gage_id
    output_file_name = args.output_file_name
    test_spp_lstm(
        project_name=project_name,
        train_period=train_period,
        test_period=test_period,
        gage_id=gage_id,
        time_now=time_now,
        output_file_name=output_file_name,
    )


def test_lstm_gage(args):
    project_name = args.project_name
    gage_id = args.gage_id
    test_period = args.test_period
    output_file_name = args.output_file_name
    lstm_gage(
        project_name=project_name,
        gage_id=gage_id,
        test_period=test_period,
        output_file_name=output_file_name,
    )


if __name__ == "__main__":
    print("Here are your commands:")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_name",
        dest="project_name",
        help="project path,such as 'test_spp_lstm/ex1'",
        type=str,
        default="test_spp_lstm/swh",
    )
    parser.add_argument(
        "--train_period",
        dest="train_period",
        help="training period, such as ['2017-01-02T19:00:00', '2017-01-02T21:00:00']",
        nargs="+",
        default=["2023-10-01T00:00:00", "2023-10-13T21:00:00"],
    )
    parser.add_argument(
        "--output_file_name",
        dest="output_file_name",
        help="output_file_name",
        nargs="+",
        default=["output/output"],
    )
    parser.add_argument(
        "--test_period",
        dest="test_period",
        help="testing period, such as ['2017-01-02T19:00:00', '2017-01-02T21:00:00']",
        nargs="+",
        # default=["2023-10-06T15:00:00", "2023-10-06T20:00:00"],
        default=["2018-08-08T08:00:00", "2018-08-08T08:00:00"],
    )
    parser.add_argument(
        "--gage_id",
        dest="gage_id",
        help="stcd id, such as ['05584500']",
        nargs="+",
        default=["4150377B"],
    )

    print(parser.parse_args().test_period)
    # parser.parse_args().test_period = convert_times(parser.parse_args().test_period)
    args = parser.parse_args()
    print(f"Your command arguments:{str(args)}")
    test_model(args, parser.parse_args().test_period[0])
    test_lstm_gage(args)
