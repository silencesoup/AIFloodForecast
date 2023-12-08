import os
import hydrodataset as hds
from hydrodataset import HydroDataset, CACHE_DIR
from hydrodataset.camels import map_string_vars
import numpy as np
from netCDF4 import Dataset as ncdataset
import collections
import pandas as pd
import xarray as xr
import json

GPM_GFS_NO_DATASET_ERROR_LOG = (
    "We cannot read this dataset now. Please check if you choose correctly:\n"
)


class GPM_GFS(HydroDataset):
    def __init__(
        self,
        data_path=os.path.join("gpm_gfs_data"),
        download=False,
        region: str = "US",
    ):
        super().__init__(data_path)
        self.region = region
        self.data_source_description = self.set_data_source_describe()
        if download:
            raise NotImplementedError(
                "We don't provide methods for downloading data at present\n"
            )
        self.sites = self.read_site_info()

    def get_name(self):
        return "GPM_GFS_" + self.region

    def set_data_source_describe(self) -> collections.OrderedDict:
        """
        the files in the dataset and their location in file system

        Returns
        -------
        collections.OrderedDict
            the description for GPM and GFS dataset
        """
        gpm_gfs_db = self.data_source_dir
        if self.region == "US":
            return self._set_data_source_GpmGfsUS_describe(gpm_gfs_db)

        else:
            raise NotImplementedError(GPM_GFS_NO_DATASET_ERROR_LOG)

    def _set_data_source_GpmGfsUS_describe(self, gpm_gfs_db):
        # water_level of basins
        camels_water_level = gpm_gfs_db.joinpath("water_level")

        # gpm
        gpm_data = gpm_gfs_db.joinpath("gpm")

        # gfs
        gfs_data = gpm_gfs_db.joinpath("gfs")

        # basin id
        gauge_id_file = gpm_gfs_db.joinpath("camels_name.txt")

        return collections.OrderedDict(
            GPM_GFS_DIR=gpm_gfs_db,
            CAMELS_WATER_LEVEL=camels_water_level,
            GPM_DATA=gpm_data,
            GFS_DATA=gfs_data,
            CAMELS_GAUGE_FILE=gauge_id_file,
        )

    def read_site_info(self) -> pd.DataFrame:
        """
        Read the basic information of gages in a CAMELS dataset

        Returns
        -------
        pd.DataFrame
            basic info of gages
        """
        camels_gauge_file = self.data_source_description["CAMELS_GAUGE_FILE"]
        if self.region == "US":
            data = pd.read_csv(
                camels_gauge_file, sep=";", dtype={"gauge_id": str, "huc_02": str}
            )
        else:
            raise NotImplementedError(GPM_GFS_NO_DATASET_ERROR_LOG)
        return data

    def read_object_ids(self, **kwargs) -> np.array:
        """
        read station ids

        Parameters
        ----------
        **kwargs
            optional params if needed

        Returns
        -------
        np.array
            gage/station ids
        """
        if self.region in ["US"]:
            return self.sites["gauge_id"].values
        else:
            raise NotImplementedError(GPM_GFS_NO_DATASET_ERROR_LOG)

    def waterlevel_xrdataset(
        self,
    ):
        """
        convert txt file of water level to a total netcdf file with corresponding time
        """
        # open the waterlevel and gpm files respectively
        waterlevel_path = os.path.join(hds.ROOT_DIR, "gpm_gfs_data", "water_level")
        gpm_path = os.path.join(hds.ROOT_DIR, "gpm_gfs_data", "gpm")
        waterlevel_path_list = os.listdir(waterlevel_path)
        gpm_path_list = os.listdir(gpm_path)

        basin_id_list = []
        waterlevel_array_list = []
        time_fin = []

        for basin_id in gpm_path_list:
            basin_id_list.append(basin_id)
            waterlevel_file = os.path.join(waterlevel_path, str(basin_id) + ".txt")
            df = pd.read_csv(
                waterlevel_file,
                sep="\s+",
                header=None,
                engine="python",
                usecols=[0, 1, 2],
            )
            df.columns = ["date", "time", "water_level"]
            df["datetime"] = df.apply(
                lambda row: " ".join(
                    row[:2],
                ),
                axis=1,
            )
            df = df.drop(columns=[df.columns[0], df.columns[1]])
            df = df[df.columns[::-1]]
            df["datetime"] = df["datetime"].str.slice(0, 19)

            id = waterlevel_path_list.index(basin_id)
            gpm_file = os.path.join(gpm_path, str(waterlevel_path_list[id][:-4]))
            gpm_file_list = os.listdir(gpm_file)
            time_list = []
            for time in gpm_file_list:
                datetime = time[0:19]
                time_list.append(datetime)
            time_df = pd.DataFrame(time_list, columns=["datetime"])

            df_fin = pd.merge(df, time_df, how="right", on="datetime")
            df_fin["datetime"] = pd.to_datetime(df_fin["datetime"])
            df_fin = df_fin.sort_values("datetime")

            waterlevel_array_list.append(df_fin[["water_level"]].values)
            time_fin.append(df_fin["datetime"].values)

        waterlevel_merged_array = np.concatenate(waterlevel_array_list, axis=1)
        ds = xr.Dataset(
            {
                "waterlevel": (["time", "basin"], waterlevel_merged_array),
            },
            coords={"time": time_fin[0], "basin": basin_id_list},
        )

        ds.to_netcdf(os.path.join(hds.ROOT_DIR, "gpm_gfs_data", "water_level_total.nc"))

    def gpm_xrdataset(self):
        gpm_path = os.path.join(hds.ROOT_DIR, "gpm_gfs_data", "gpm")
        gpm_path_list = os.listdir(gpm_path)

        gpm_whole_path = os.path.join(hds.ROOT_DIR, "gpm_gfs_data", "gpm_whole")
        gpm_whole_path_list = os.listdir(gpm_whole_path)
        gpm_whole_path_list_tmp = []
        for path in gpm_whole_path_list:
            gpm_whole_path_list_tmp.append(path[:-3])
        gpm_path_list = list(set(gpm_path_list) - set(gpm_whole_path_list_tmp))

        if len(gpm_whole_path_list_tmp) != 0:
            for basin in gpm_path_list:
                total_data = []
                gpm_list = os.listdir(os.path.join(gpm_path, str(basin)))

                for gpm in gpm_list:
                    single_data_path = os.path.join(gpm_path, str(basin), gpm)
                    single_data = xr.open_dataset(single_data_path)
                    total_data.append(single_data)

                da = xr.concat(total_data, dim="time")

                da_sorted = da.sortby("time")

                da_sorted.to_netcdf(
                    os.path.join(hds.ROOT_DIR, "gpm_gfs_data", "gpm_whole", str(basin))
                    + "nc"
                )

    def read_waterlevel_xrdataset(
        self, gage_id_lst=None, t_range: list = None, var_list=None, **kwargs
    ):
        if var_list is None or len(var_list) == 0:
            return None

        folder = os.path.exists(
            os.path.join(hds.ROOT_DIR, "gpm_gfs_data", "water_level_total.nc")
        )
        if not folder:
            self.waterlevel_xrdataset()

        waterlevel = xr.open_dataset(
            os.path.join("/ftproot", "cache", "gpm_gfs_data", "water_level_total.nc")
        )
        all_vars = waterlevel.data_vars
        if any(var not in waterlevel.variables for var in var_list):
            raise ValueError(f"var_lst must all be in {all_vars}")
        return waterlevel[["waterlevel"]].sel(
            time=slice(t_range[0], t_range[1]), basin=gage_id_lst
        )

    def read_gpm_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        **kwargs,
    ):
        if var_lst is None:
            return None

        gpm_dict = {}
        for basin in gage_id_lst:
            gpm = xr.open_dataset(
                os.path.join(
                    "/ftproot", "cache", "gpm_gfs_data_24h", str(basin) + ".nc"
                )
            )

            gpm = gpm[var_lst].sel(time=slice(t_range[0], t_range[1]))
            gpm_dict[basin] = gpm

        return gpm_dict

    def tp_json_2_nc(self, name):
        # Load the JSON file
        file_path = os.path.join("param/", name + "-rainfall.json")
        with open(file_path, 'r') as file:
            json_data = json.load(file)

        # Extract 'real' and 'forecast' data
        real_data = json_data['real']
        forecast_data = json_data['forecast']

        # Extracting data from 'real'
        real_times = [pd.to_datetime(item['time']) for item in real_data]
        real_lons = [float(item['lgtd']) for item in real_data]
        real_lats = [float(item['lttd']) for item in real_data]
        real_rainfall = [float(item['rainfall']) for item in real_data]

        # Extracting time and rainfall data from 'forecast'
        forecast_times = [pd.to_datetime(item['time']) for item in forecast_data]
        forecast_rainfall = [float(item['rainfall']) for item in forecast_data]

        # Combining 'real' and 'forecast' times and rainfall
        combined_times = real_times + forecast_times
        combined_lons = real_lons + [float(item['lgtd']) for item in forecast_data]
        combined_lats = real_lats + [float(item['lttd']) for item in forecast_data]
        combined_rainfall = real_rainfall + forecast_rainfall

        # Creating unique lists for combined lon, lat, and time
        unique_combined_times = sorted(set(combined_times))
        sorted_combined_lons = sorted(set(combined_lons))
        sorted_combined_lats = sorted(set(combined_lats))

        # Converting lon and lat to float32
        sorted_combined_lons = np.array(sorted_combined_lons, dtype=np.float32)
        sorted_combined_lats = np.array(sorted_combined_lats, dtype=np.float32)

        # For 'forecast', using the first 'tm' value and subtracting 1 hour for 'time_now'
        forecast_time = pd.to_datetime(forecast_data[0]['time'])

        # Creating a 4D array for 'tp' with sorted dimensions
        tp_sorted = np.full((1, len(unique_combined_times), len(sorted_combined_lons), len(sorted_combined_lats)), np.nan)

        # Filling the sorted array with combined rainfall data
        for time, lon, lat, rainfall in zip(combined_times, combined_lons, combined_lats, combined_rainfall):
            time_idx = unique_combined_times.index(time)
            lon_idx = np.where(sorted_combined_lons == np.float32(lon))[0][0]
            lat_idx = np.where(sorted_combined_lats == np.float32(lat))[0][0]
            tp_sorted[0, time_idx, lon_idx, lat_idx] = rainfall

        # Creating the xarray Dataset with sorted dimensions
        ds_sorted = xr.Dataset(
            {
                "tp": (["time_now", "time", "lon", "lat"], tp_sorted)
            },
            coords={
                "time": unique_combined_times,
                "lat": sorted_combined_lats,
                "lon": sorted_combined_lons,
                "time_now": [forecast_time],
            }
        )

        # Saving the sorted Dataset to a new NetCDF file
        nc_file_path = os.path.join("param/", name + "-rainfall.nc")
        ds_sorted.to_netcdf(nc_file_path)

    def read_test_gpm_xrdataset(
        self,
        gage_id_lst: list = None,
        t_range: list = None,
        var_lst: list = None,
        name: str = None,
        **kwargs,
    ):
        if var_lst is None:
            return None
        folder = os.path.exists(
            os.path.join(os.path.join("param/", name + "-rainfall.nc"))
        )
        # if not folder:
        self.tp_json_2_nc(name)
        gpm_dict = {}
        for basin in gage_id_lst:
            gpm = xr.open_dataset(os.path.join("param/", name + "-rainfall.nc"))

            gpm = gpm[var_lst].sel(time=slice(t_range[0], t_range[1]))
            gpm_dict[basin] = gpm

        return gpm_dict

    def waterlevel_json_2_nc(self, name):
        # 加载JSON文件
        json_file_path = os.path.join("param/", name + "-z.json")
        with open(json_file_path, "r") as file:
            data = json.load(file)

        # 将JSON数据转换为Pandas DataFrame
        df = pd.DataFrame(data)

        # 转换 'waterlevel' 列为数值类型（假设它应该是浮点数）
        df["waterlevel"] = pd.to_numeric(df["waterlevel"], errors="coerce")

        # 转换 'tm' 列为日期时间类型
        df["tm"] = pd.to_datetime(df["tm"], errors="coerce", format="%Y-%m-%d %H:%M:%S")

        # 检查'stcd'列的唯一性
        if df["stcd"].nunique() == 1:
            # 创建xarray数据集
            ds_new = xr.Dataset(
                {"waterlevel": (("time", "basin"), df[["waterlevel"]].values)},
                coords={"time": df["tm"].values, "basin": [df["stcd"].iloc[0]]},
            )
        else:
            raise NotImplementedError(
                "there are more than one stcd in the corresponding -z.json"
            )
        netcdf_file_path = os.path.join("param/", name + "-z.nc")
        # 保存xarray数据集为NetCDF格式
        ds_new.to_netcdf(netcdf_file_path)

    def read_test_waterlevel_xrdataset(
        self,
        gage_id_lst=None,
        t_range: list = None,
        var_list=None,
        name: str = None,
        **kwargs,
    ):
        if var_list is None or len(var_list) == 0:
            return None

        folder = os.path.exists((os.path.join("param/", name + "-z.nc")))
        # if not folder:
        self.waterlevel_json_2_nc(name)
        waterlevel = xr.open_dataset(os.path.join("param/", name + "-z.nc"))
        return waterlevel[["waterlevel"]].sel(
            time=slice(t_range[0], t_range[1]), basin=gage_id_lst
        )

    def read_shap_gpm_xrdataset(
        self,
        gage_id_lst: list = None,
        var_lst: list = None,
        name: str = None,
        **kwargs,
    ):
        if var_lst is None:
            return None

        gpm_dict = {}
        for basin in gage_id_lst:
            gpm = xr.open_dataset(os.path.join("param/", name + "-rainfall.nc"))

            gpm = gpm[var_lst]
            gpm_dict[basin] = gpm

        return gpm_dict

    def read_shap_waterlevel_xrdataset(
        self,
        gage_id_lst=None,
        var_list=None,
        name: str = None,
        **kwargs,
    ):
        if var_list is None or len(var_list) == 0:
            return None

        waterlevel = xr.open_dataset(os.path.join("param/", name + "-z.nc"))
        return waterlevel[["waterlevel"]].sel(basin=gage_id_lst)

    def read_gpm(self):
        gpm_data = []
        data_path_all = os.path.join(hds.ROOT_DIR, "gpm_gfs_data", "gpm")
        data_path_basin_list = os.listdir(data_path_all)
        for i in range(len(data_path_basin_list)):
            data_path_basin = os.path.join(data_path_all, data_path_basin_list[i])
            data_path_time_list = os.listdir(data_path_basin)
            for j in range(len(data_path_time_list)):
                dict = {
                    "ID": "",
                    "data": "",
                    "length": "",
                    "width": "",
                    "precip": "",
                }
                data_path = os.path.join(data_path_basin, data_path_time_list[j])
                dst = ncdataset(data_path)
                precip = dst.variables["precipitationCal"][:]
                dict.update(
                    {
                        "ID": data_path_basin_list[i],
                        "data": data_path_time_list[j],
                        "length": len(precip[0]),
                        "width": len(precip[0][0]),
                        "precip": precip[0],
                    }
                )
                gpm_data.append(dict)

        return gpm_data

    def read_attr_xrdataset(self, gage_id_lst=None, var_lst=None, **kwargs):
        if var_lst is None or len(var_lst) == 0:
            return None
        # attr = xr.open_dataset(os.path.join(hds.ROOT_DIR, "camelsus_attributes.nc"))
        attr = xr.open_dataset(
            os.path.join("/ftproot", "cache", "camelsus_attributes.nc")
        )
        if "all_number" in list(kwargs.keys()) and kwargs["all_number"]:
            attr_num = map_string_vars(attr)
            return attr_num[var_lst].sel(basin=gage_id_lst)
        return attr[var_lst].sel(basin=gage_id_lst)

    def read_mean_prcp(self, gage_id_lst) -> np.array:
        if self.region in ["US", "AUS", "BR", "GB"]:
            if self.region == "US":
                return self.read_attr_xrdataset(gage_id_lst, ["p_mean"])
            return self.read_constant_cols(
                gage_id_lst, ["p_mean"], is_return_dict=False
            )
        elif self.region == "CL":
            # there are different p_mean values for different forcings, here we chose p_mean_cr2met now
            return self.read_constant_cols(
                gage_id_lst, ["p_mean_cr2met"], is_return_dict=False
            )
        else:
            raise NotImplementedError(GPM_GFS_NO_DATASET_ERROR_LOG)

    def read_area(self, gage_id_lst) -> np.array:
        if self.region == "US":
            return self.read_attr_xrdataset(gage_id_lst, ["area_gages2"])
        else:
            raise NotImplementedError(GPM_GFS_NO_DATASET_ERROR_LOG)
