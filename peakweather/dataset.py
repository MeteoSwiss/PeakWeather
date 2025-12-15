import os
from dataclasses import dataclass
from typing import List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .io import download_url, extract_zip, import_xarray
from .utils import (
    df_add_missing_columns,
    sliding_window_view,
    timestamps_from_xr,
    to_pandas_freq,
    xr_to_np,
)

FrameArray = Union[pd.DataFrame, np.ndarray]


@dataclass
class Windows:
    """Windows is a data class containing sliding window data, either as numpy arrays or
    Xarray datasets. 

    Attributes:
    ----------
    x : Union[pd.DataFrame, np.ndarray, xr.Dataset]
        Input data of the look-back window.
    mask_x : Union[pd.DataFrame, np.ndarray, xr.Dataset]
        Boolean mask for x indicating valid or missing values.
    y : Union[pd.DataFrame, np.ndarray, xr.Dataset]
        Target data of the horizon window.
    mask_y : Union[pd.DataFrame, np.ndarray, xr.Dataset]
        Boolean mask for y indicating valid or missing values.
    index_x : Optional[Union[pd.DatetimeIndex, np.ndarray]], default=None
        Timestamps or indices corresponding to the values in x.
    index_y : Optional[Union[pd.DatetimeIndex, np.ndarray]], default=None
        Timestamps or indices corresponding to the values in y.
    nwp : Optional[Union[np.ndarray, xr.Dataset]], default=None
        Numerical Weather Prediction (ICON-CH1-EPS) data associated with the horizon window y.
    nwp_to_y : Optional[Sequence[int]], default=None
        Mapping of NWP parameters to the target parameters in y.
        Specifies which NWP variables correspond to which y variables.
    """
    x: Union[pd.DataFrame, np.ndarray, "xr.Dataset"]
    mask_x: Union[pd.DataFrame, np.ndarray, "xr.Dataset"]
    y: Union[pd.DataFrame, np.ndarray, "xr.Dataset"]
    mask_y: Union[pd.DataFrame, np.ndarray, "xr.Dataset"]
    index_x: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None
    index_y: Optional[Union[pd.DatetimeIndex, np.ndarray]] = None
    nwp: Optional[Union[np.ndarray, "xr.Dataset"]] = None
    nwp_to_y: Optional[Sequence[int]] = None


class PeakWeatherDataset:
    """PeakWeather is a high-quality meteorological dataset derived from `SwissMetNet
    <https://www.meteoswiss.admin.ch/weather/measurement-systems/land-based-stations/automatic-measurement-network.html>`_, 
    the automated measurement network operated by MeteoSwiss. It offers a robust resource 
    for research and applications in spatiotemporal modeling.
    
    PeakWeather includes high-frequency meteorological observations recorded every 
    10 minutes, collected from 302 ground stations distributed across Switzerland, 
    covering the period from January 1, 2017 to March 31, 2025. The dataset also 
    provides high-resolution topographic features at 50-meter resolution and ensemble 
    forecasts from the ICON-CH1-EPS operational numerical weather prediction (NWP) model.
    The dataset is described in more details in
    `"PeakWeather: MeteoSwiss Weather Station Measurements for Spatiotemporal Deep Learning"
    <https://arxiv.org/abs/2506.13652>`_ (Zambon et al., 2025).

    This class loads and reads the PeakWeather dataset, providing utilities for 
    accessing, preprocessing, and integrating the data into machine learning workflows.

    Dataset size:
        + Time steps: 433728
        + Stations: 302
        + Channels: 8
        + Sampling interval: 10 minutes

    Channels:
        + ``wind_direction``: Wind direction (degree). Ten minutes mean.
        + ``wind_speed``: Wind speed scalar (meter/second). Ten minutes mean.
        + ``wind_gust``: Gust peak (meter/second). Maximum recorded over ten minutes.
        + ``pressure``: Atmospheric pressure at barometric altitude (QFE) (hectopascal).
          Instant value.
        + ``precipitation``: Precipitation (millimeter). Ten minutes total.
        + ``sunshine``: Sunshine duration (minute). Ten minutes total.
        + ``temperature``: Air temperature 2 m above ground (degree Celsius).
          Instant value.
        + ``humidity``: Relative air humidity 2 m above ground (per cent).
          Instant value.

    Static attributes:
        + :obj:`stations_table`: Information associated with the stations, including
          name, type, latitude, longitude, height, and topographical descriptors.
        + :obj:`installation_table`: Information about stations' installation.
        + :obj:`parameters_table`: Description of the quantities measured.

    Args:
        root (str, optional): The root directory where the dataset is stored.
            If :obj:`None`, the dataset is stored in the current working directory.
            (default: :obj:`None`)
        pad_missing_values (bool, optional): If :obj:`True`, pad missing
            parameter values with NaN values. (default: :obj:`True`)
        years (int or list of int, optional): The years to include in the dataset.
            If :obj:`None`, all available years are included. (default: :obj:`None`)
        extended_topo_vars (str or list of str, optional): The topography variables
            to include in the dataset. If :obj:`None`, no topography variables are
            included. (default: :obj:`"none"`)
        extended_nwp_pars (str or list of str, optional): The NWP (ICON-CH1-EPS) parameters
            to include in the dataset. If :obj:`None`, no NWP parameters are
            included. (default: :obj:`"none"`)
        imputation_method (str, optional): The method to use for imputing missing
            values. Options are "locf" (last observation carried forward), "zero"
            (fill with zero), or :obj:`None` (no imputation). (default: :obj:`"zero"`)
        interpolation_method (str, optional): The method to use for interpolating
            topography variables. Options are "linear", "nearest", "quadratic",
            "cubic", "barycentric", "krogh", "akima", or "makima". (default: :obj:`"nearest"`)
        freq (str, optional): The frequency to resample the dataset to. If :obj:`None`,
            no resampling is applied. (default: :obj:`None`)
        compute_uv (bool): Whether the u-v components of the wind should be computed and 
        included in the dataset. (default: True)
        station_type (str, optional): The type of stations to consider, either
            meteorological stations or rain gauges. If not defined, all stations
            will be included.
            (default: :obj:`None`)
        aggregation_methods (dict, optional): If given allows to apply a different
            aggregation than the default one to the specified parameters. The
            dictionary must map the parameter string name to one of :obj:`"mean"`,
            :obj:`"max"`, :obj:`"sum"`, :obj:`"last"`.
            (default: :obj:`None`)
    """
    __version__ = "0.2.0"

    base_url = ("https://huggingface.co/datasets/MeteoSwiss/PeakWeather/"
                "resolve/main/data/")

    available_parameters = {
        "temperature": "tre200s0",
        "humidity": "ure200s0",
        "precipitation": "rre150z0",
        "sunshine": "sre000z0",
        "pressure": "prestas0",
        "wind_speed": "fkl010z0",
        "wind_gust": "fkl010z1",
        "wind_direction": "dkl010z0",
    }
    available_years = {2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025}
    available_topography = {
        "DEM",
        "SN_DERIVATIVE_2000M_SIGRATIO1",
        "SN_DERIVATIVE_10000M_SIGRATIO1",
        "WE_DERIVATIVE_2000M_SIGRATIO1",
        "WE_DERIVATIVE_10000M_SIGRATIO1",
        "STD_2000M",
        "STD_10000M",
        "TPI_2000M",
        "TPI_10000M",
        "ASPECT_2000M_SIGRATIO1",
        "ASPECT_10000M_SIGRATIO1",
        "SLOPE_2000M_SIGRATIO1",
        "SLOPE_10000M_SIGRATIO1",
    }
    available_icon = {
        "wind_u",
        "wind_v",
        "humidity",
        "precipitation",
        "pressure",
        "sunshine",
        "temperature",
        "wind_gust"
    }

    test_set_start = pd.Timestamp("2024-10-01", tz="UTC")

    def __init__(self,
                 root: str = None,
                 pad_missing_values: bool = True,
                 years: Optional[Union[int, Sequence[int]]] = None,
                 parameters: Optional[Union[str, Sequence[str]]] = None,
                 extended_topo_vars: Optional[Union[str, Sequence[str]]] = "none",
                 extended_nwp_pars: Optional[Union[str, Sequence[str]]] = "none",
                 imputation_method: Literal["locf", "zero", None] = "zero",
                 interpolation_method: str = "nearest",
                 freq: str = None,
                 compute_uv: bool = True,
                 station_type: Optional[Literal["rain_gauge", "meteo_station"]] = None,
                 aggregation_methods: dict[str, str] = None):
        # set root path
        self.root = root
        self.pad_missing_values = pad_missing_values

        self.compute_uv = compute_uv
        self.station_type = station_type
        if self.station_type == 'rain_gauge' and self.compute_uv:
            raise ValueError(
                "Cannot compute wind components on a dataset with rain gauges only "
                "(station_type='rain_gauge and compute_uv=True')"
            )

        # Select parameters
        view_params = set(self.available_parameters.keys())
        if self.station_type == 'rain_gauge':
            view_params = {"temperature", "precipitation"}
        if parameters is not None:
            params = {parameters} if isinstance(parameters, str) else set(parameters)
            if compute_uv:
                params = params.union({"wind_speed", "wind_direction"})
            if "wind_u" in parameters or "wind_v" in parameters:
                assert compute_uv, "Parameters `wind_u` and `wind_v` require setting `compute_uv=True`."
            view_params = view_params.intersection(params)
            if not len(view_params):
                raise ValueError(f"Incorrect choice for 'parameters' ({parameters}). "
                                 f"Must be a subset of {self.available_parameters}.")
        self.parameter_map = {
            k: self.available_parameters[k] for k in sorted(view_params)
        }

        # Select years
        view_years = self.available_years
        if years is not None:
            years = {years} if isinstance(years, int) else set(years)
            view_years = view_years.intersection(years)
            if not len(view_years):
                raise ValueError(f"Incorrect choice for 'year' ({years}). "
                                 f"Must be a subset of {self.available_years}.")
        self.years = sorted(view_years)

        # Select topography variables
        view_topos = set()
        if isinstance(extended_topo_vars, str) and extended_topo_vars == "none":
            view_topos = set()
        elif isinstance(extended_topo_vars, str) and extended_topo_vars == "all":
            view_topos = self.available_topography
        elif extended_topo_vars is not None:
            if isinstance(extended_topo_vars, str):
                extended_topo_vars = {extended_topo_vars}
            else:
                extended_topo_vars = set(extended_topo_vars)
            view_topos = self.available_topography.intersection(extended_topo_vars)
            if not len(view_topos):
                raise ValueError("Incorrect choice for 'extended_topo_vars' "
                                 f"({extended_topo_vars}). Must be a subset of "
                                 f"{self.available_topography}.")
        self.extended_topo_vars = sorted(view_topos)

        # Select NWP (ICON) parameters
        view_nwp = set()
        if isinstance(extended_nwp_pars, str) and extended_nwp_pars == "none":
            view_nwp = set()
        elif isinstance(extended_nwp_pars, str) and extended_nwp_pars == "all":
            view_nwp = self.available_icon
        elif extended_nwp_pars is not None:
            if isinstance(extended_nwp_pars, str):
                extended_nwp_pars = {extended_nwp_pars}
            else:
                extended_nwp_pars = set(extended_nwp_pars)
            view_nwp = self.available_icon.intersection(extended_nwp_pars)
            if not len(view_nwp):
                raise ValueError("Incorrect choice for 'extended_nwp_pars' "
                                 f"({extended_nwp_pars}). Must be a subset of "
                                 f"{self.available_icon}.")
        self.extended_nwp_pars = sorted(view_nwp)

        self.imputation_method = imputation_method
        self.interpolation_method = interpolation_method

        # Set dataset frequency here to resample when loading
        if freq is not None:
            freq = to_pandas_freq(freq)
        self.freq = freq

        # load dataset
        df_observations, df_mask, static_tables = self.load(
            aggregation_methods=aggregation_methods)
        df_stations, df_installation, df_params = static_tables

        # Store time series attributes
        self.observations = df_observations
        self.mask = df_mask

        # Store static attributes
        self.stations_table = df_stations
        self.installation_table = df_installation
        self.parameters_table = df_params

    def __repr__(self):
        return "{}(num_time_steps={}, num_stations={}, num_parameters={})".format(
            self.__class__.__name__,
            self.num_time_steps,
            self.num_stations,
            self.num_parameters,
        )

    @property
    def urls(self) -> Mapping[str, str]:
        """The URLs of the files to download."""
        urls = {
            "stations": self.base_url + "stations.parquet",
            "installation": self.base_url + "installation.parquet",
            "parameters": self.base_url + "parameters.parquet",
            "disclaimer": self.base_url + "disclaimer.txt"
        }
        urls.update({
            str(year): f"{self.base_url}observations/{year}.parquet"
            for year in self.available_years
        })
        urls.update({
            f"topo_{topo}": f"{self.base_url}topography/{topo}.zarr.zip"
            for topo in self.available_topography
        })
        urls.update({
            f"icon_{nwp_var}": f"{self.base_url}icon/{nwp_var}.zarr.zip"
            for nwp_var in self.available_icon
        })
        return urls

    @property
    def root_dir(self) -> str:
        """The root directory where the dataset is stored."""
        if self.root is not None:
            root = os.path.expanduser(os.path.normpath(self.root))
        else:
            root = os.path.abspath(os.path.join(os.curdir, self.__class__.__name__))
        return root

    @property
    def required_file_names(self) -> Mapping[str, str]:
        """The relative filepaths that must be present in order to skip downloading."""
        out = {
            "stations": "stations.parquet",
            "installation": "installation.parquet",
            "parameters": "parameters.parquet",
            "disclaimer": "disclaimer.txt"
        }
        out.update({
            str(year): f"observations/{year}.parquet" for year in self.years
        })
        out.update({
            f"topo_{topo}": f"topography/{topo}.zarr"
            for topo in self.extended_topo_vars
        })
        out.update({
            f"icon_{nwp_var}": f"icon/{nwp_var}.zarr"
            for nwp_var in self.extended_nwp_pars
        })
        return out

    @property
    def required_files_paths(self) -> Mapping[str, str]:
        """The absolute filepaths that must be present in order to skip downloading."""
        return {
            k: os.path.join(self.root_dir, f)
            for k, f in self.required_file_names.items()
        }

    @property
    def num_stations(self) -> int:
        """Number of stations in the dataset."""
        return len(self.stations_table)

    @property
    def num_parameters(self) -> int:
        """Number of parameters in the dataset."""
        return len(self.parameters_table)

    @property
    def num_time_steps(self) -> int:
        """Number of time_steps in the dataset."""
        return len(self.observations)

    @property
    def stations(self) -> pd.Index:
        """IDs of stations in the dataset."""
        return self.stations_table.index

    @property
    def parameters(self) -> pd.Index:
        """Parameters measured by the stations in the dataset."""
        return self.parameters_table.index

    @property
    def missing_values(self) -> pd.Series:
        """Missing values for each parameter, considering
        stations equipped with the necessary sensor.
        """
        valid_columns = self.mask.loc[:, self.mask.any()]
        mean_missing = {
            param: 1 - valid_columns.xs(param, axis=1, level=1).values.mean()
            for param in self.parameters
        }
        out = pd.Series(mean_missing, name="missing_perc", dtype="float32")
        return out.reindex(self.parameters)

    def __len__(self):
        return self.num_time_steps

    def download(self) -> None:
        """Download the dataset if it is not already present."""
        for key, filepath in self.required_files_paths.items():
            # download only required data that are missing
            if not os.path.exists(filepath):
                sub_dir, filename = os.path.split(filepath)
                os.makedirs(sub_dir, exist_ok=True)
                url = self.urls[key]
                if '.zip' in url:
                    filename = filename + '.zip'
                self._maybe_unzip(download_url(self.urls[key], sub_dir, filename))

    def _maybe_unzip(self, filepath) -> None:
        if os.path.exists(filepath) and filepath.endswith('.zip'):
            extract_zip(path=filepath,
                        folder=filepath.split('.zip')[0])
            os.remove(filepath)

    def load_topography(self) -> dict:
        """Load the topography data.

        This method downloads the topography data if it is not already present
        and loads the data into memory. The data is returned as a dictionary
        containing the topography data for each variable.
        """
        if len(self.extended_topo_vars) == 0:
            return {}
        xr = import_xarray()
        topography = {
            k: xr.open_zarr(v) for k, v in self.required_files_paths.items()
            if k.startswith("topo_")
        }
        return topography

    def _add_uv_columns(self, df_observations: pd.DataFrame) -> pd.DataFrame:
        # Select wind speed and direction columns
        speed = df_observations.xs("wind_speed", axis=1, level=1)
        direction = df_observations.xs("wind_direction", axis=1, level=1)
        # Select only the columns that are present in both DataFrames
        cols = speed.columns.intersection(direction.columns).sort_values()
        speed = speed.loc[:, cols]
        direction = direction.loc[:, cols]
        # Compute u and v components
        u, v = PeakWeatherDataset.get_uv_wind(wind_speed=speed.values,
                                              wind_direction=direction.values,
                                              direction_unit="deg")
        # Create new columns for u and v components
        uv = np.stack([u, v], axis=-1).reshape(u.shape[0], -1)
        uv_cols = pd.MultiIndex.from_product([cols, ["wind_u", "wind_v"]])
        df_uv = pd.DataFrame(uv, index=df_observations.index, columns=uv_cols)
        # Concatenate the new columns with the original DataFrame
        df_observations = pd.concat([df_observations, df_uv], axis=1)
        # Sort the columns
        df_observations = df_observations.sort_index(axis=1)
        return df_observations

    def _recompute_dir_from_uv(self,
                               df_observations,
                               keep_uv: bool = True) -> pd.DataFrame:
        # Select wind speed and direction columns
        u = df_observations.xs("wind_u", axis=1, level=1)
        v = df_observations.xs("wind_v", axis=1, level=1)
        # Select only the columns that are present in both DataFrames
        cols = u.columns.intersection(v.columns).sort_values()
        u = u.loc[:, cols]
        v = v.loc[:, cols]
        # Compute wind direction from u and v components
        new_wind_dir = PeakWeatherDataset.get_wind_direction(u=u.values,
                                                             v=v.values)
        # Overwrite the wind direction columns
        wind_dir_cols = pd.MultiIndex.from_product([cols, ["wind_direction"]])
        df_observations.loc[:, wind_dir_cols] = new_wind_dir

        # Optional: drop the u and v columns
        if not keep_uv:
            df_observations = df_observations.drop(columns=["wind_u", "wind_v"],
                                                   level=1)
        return df_observations

    def resample(self,
                 df_observations: pd.DataFrame,
                 df_parameters: pd.DataFrame) -> pd.DataFrame:
        """Resample the observations to the specified frequency.

        This method resamples the observations to the specified frequency
        and returns the resampled DataFrame.
        """
        assert self.freq is not None, \
            "Resampling frequency is not set. Please set the 'freq' parameter."
        if "wind_direction" in self.parameter_map and not self.compute_uv:
            df_observations = self._add_uv_columns(df_observations=df_observations)
            df_parameters = df_parameters.copy()
            wind_aggr = df_parameters.loc["wind_direction", "aggregation"]
            df_parameters.loc["wind_u", "aggregation"] = wind_aggr
            df_parameters.loc["wind_v", "aggregation"] = wind_aggr
        resampled_dfs = []
        resample_methods = df_parameters.groupby("aggregation").groups

        # Resample the data according to the aggregation method
        for method, cols in resample_methods.items():
            # Select the columns to resample
            df_cols = df_observations.loc[:, (slice(None), cols)]
            # Resample the data
            df_res = df_cols.resample(self.freq, closed="right", label="right")
            if method == "mean":
                resampled_dfs.append(df_res.mean())
            elif method == "max":
                resampled_dfs.append(df_res.max())
            elif method == "sum":
                resampled_dfs.append(df_res.sum(min_count=1))
            elif method == "last":
                resampled_dfs.append(df_res.last())
            elif method == "circ_mean":
                # Set a placeholder. The actual computation is carried out separately below.
                df_ph = df_res.last()
                df_ph[:] = np.nan
                resampled_dfs.append(df_ph)
            else:
                raise ValueError(f"Invalid resampling method: {method}.")
        # Concatenate the gust data with the rest of the data
        df_resampled = pd.concat(resampled_dfs, axis=1).sort_index(axis=1)
        assert df_resampled.shape[1] == df_observations.shape[1], \
            "Resampling failed: number of columns do not match."

        # Handle circle mean aggragation (it's currently necessary and supported only for wind_direction)
        if self.station_type != "rain_gauge" and "circ_mean" in resample_methods:
            assert len(resample_methods["circ_mean"]) <= 1
            if len(resample_methods["circ_mean"]) == 1:  assert \
                resample_methods["circ_mean"][0] == "wind_direction"
            if "wind_direction" in self.parameter_map and "wind_direction" in \
                    resample_methods["circ_mean"]:
                # If resampling happened, the accumulated angle needs to be inferred
                df_resampled = self._recompute_dir_from_uv(df_observations=df_resampled,
                                                           keep_uv=self.compute_uv)
        return df_resampled

    def load_raw(self, aggregation_methods: dict[str, str] = None):
        """Load the raw dataset.

        This method downloads the dataset if it is not already present
        and loads the data into memory. The data is returned as a tuple
        containing the observations, static tables, and optional topography data.
        """
        self.download()

        filenames = self.required_files_paths

        # Load stations metadata
        df_stations = pd.read_parquet(filenames["stations"])
        df_stations = df_stations.sort_index()

        # Load installation table
        df_installation = pd.read_parquet(filenames["installation"])

        # Load parameters table
        df_params = pd.read_parquet(filenames["parameters"])
        df_params = df_params.loc[self.parameter_map.keys()]
        if aggregation_methods is not None:
            # overwrite specific aggregation methods, if specified.
            df_params['aggregation'] = pd.Series(aggregation_methods).combine_first(
                df_params['aggregation'])

        # Load observations only for selected years
        observations = []
        for year in self.years:
            data_path = filenames[str(year)]
            data_df = pd.read_parquet(data_path)
            # Select columns
            data_df = data_df.loc[:, (slice(None), self.parameter_map.values())]
            data_df = data_df.asfreq(to_pandas_freq("10min"))
            observations.append(data_df)

        if len(observations) == 1:
            df_observations = observations[0]
        else:
            df_observations = pd.concat(observations)

        # Explicit index type
        df_observations.index = df_observations.index.astype("datetime64[ns, UTC]")
        # Relabel parameters columns
        rename_map = {v: k for k, v in self.parameter_map.items()}
        df_observations = df_observations.rename(columns=rename_map, level=1)

        if self.station_type is not None:
            # Filter away unwanted stations.
            station_mask = df_stations["station_type"] == self.station_type
            df_stations = df_stations[station_mask]
            # Edge case for stations that do not exist in a subset
            # e.g., if we select only year=2017, station PFA is not observed
            observed_stations = df_observations.columns.unique(0)
            common_stations = df_stations.index.intersection(observed_stations)

            df_observations = df_observations.loc[
                              :, (common_stations, slice(None))
                              ]
            df_installation = df_installation.loc[
                df_installation["nat_abbr"].isin(df_stations.index)
            ]

        # Add the uv components for wind (derived from speed and direction)
        if self.compute_uv:
            df_observations = self._add_uv_columns(df_observations=df_observations)
            # Add the u and v wind components to the parameters table
            df_params.loc["wind_u"] = df_params.loc["wind_speed"]
            df_params.loc["wind_u", "description_E"] = "u component (E-W) of the wind."
            df_params.loc["wind_u", "param_short"] += "_u"
            df_params.loc["wind_u", "decimals"] = 10
            df_params.loc["wind_v"] = df_params.loc["wind_speed"]
            df_params.loc["wind_v", "description_E"] = "v component (N-S) of the wind."
            df_params.loc["wind_v", "param_short"] += "_v"
            df_params.loc["wind_v", "decimals"] = 10

        # Optionally resample the data to the specified frequency
        if self.freq is not None:
            df_observations = self.resample(df_observations, df_params)
        else:
            df_observations = df_observations.sort_index(axis=1)

        # Load observations only for selected years
        topography = self.load_topography()

        static_tables = (df_stations, df_installation, df_params)
        return df_observations, static_tables, topography

    def interpolate_topography(self,
                               topographic_params: dict,
                               stations_table: pd.DataFrame) -> Optional[pd.DataFrame]:
        if not len(topographic_params):
            return None

        xr = import_xarray()
        x_coords = xr.DataArray(stations_table.loc[:, "swiss_easting"], dims="points")
        y_coords = xr.DataArray(stations_table.loc[:, "swiss_northing"], dims="points")

        topography_dict = {}
        for topo_name, topo_ds in topographic_params.items():
            if self.interpolation_method == "nearest":
                topo_interp = topo_ds.sel(x=x_coords, y=y_coords, method="nearest")
            else:
                topo_interp = topo_ds.interp(x=x_coords, y=y_coords,
                                             method=self.interpolation_method)
            topography_dict.update({
                var: topo_interp[var].values
                for var in topo_interp.data_vars
            })
        # Convert to DataFrame
        df_topography = pd.DataFrame(topography_dict, index=stations_table.index)
        return df_topography

    def load(self, aggregation_methods: dict[str, str] = None):
        """Load the dataset.

        This method downloads the dataset if it is not already present
        and loads the data into memory. The data is returned as a tuple
        containing the observations, mask, and static tables.

        The observations are resampled to the specified frequency and
        missing values are imputed using the specified method.

        The topography data is interpolated to the station locations
        using the specified interpolation method.
        
        Args:
            aggregation_methods (dict, optional): If given, applies different
            aggregation strategies for the specified parameters.
                    (default: :obj:`None`)
        """
        df_observations, static_tables, topography = self.load_raw(
            aggregation_methods=aggregation_methods)
        df_stations, df_installation, df_params = static_tables

        # Remove stations that are not in the observations
        stations = df_observations.columns.unique(0)
        df_stations = df_stations.loc[stations]

        if self.pad_missing_values:
            df_observations = df_add_missing_columns(df_observations,
                                                     col0=df_stations.index,
                                                     col1=df_params.index)

        df_mask = ~(df_observations.isna())

        # Impute missing values on the temporal axis
        if self.imputation_method == "locf":
            df_observations = df_observations.ffill()
        elif self.imputation_method == "zero":
            df_observations = df_observations.fillna(0)
        elif self.imputation_method is not None:
            raise ValueError(f"Invalid imputation method: {self.imputation_method}.")

        # Interpolate topography variables
        topo_values = self.interpolate_topography(topography, df_stations)
        if topo_values is not None:
            # Add topography values to stations metadata
            df_stations.update(topo_values)

        static_tables = (df_stations, df_installation, df_params)
        return df_observations, df_mask, static_tables

    @staticmethod
    def get_uv_wind(
            wind_speed: np.ndarray,
            wind_direction: np.ndarray,
            direction_unit: Literal["deg", "rad"] = "deg",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the u,v components of the wind given wind speed and direction.
        The u component is the eastward component while v is the northward component. 

        Args:
            wind_speed (np.ndarray): The wind speed.
            wind_direction (np.ndarray): The wind direction, increasing clockwise where a northerly wind has 0 degrees.
            direction_unit (Literal["deg", "rad], optional): The angle unit of measure. Defaults to "deg".

        Returns:
            Tuple[np.ndarray]: Returns a tuple containing [u,v].
        """
        # Source: ECMWF
        # https://confluence.ecmwf.int/pages/viewpage.action?pageId=133262398
        if direction_unit == "deg":
            wind_direction = np.deg2rad(wind_direction)
        u = -wind_speed * np.sin(wind_direction)
        v = -wind_speed * np.cos(wind_direction)

        return u, v

    @staticmethod
    def get_wind_speed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Given the u and v components, get the wind speed.

        Args:
            u (np.ndarray): The eastward wind component.
            v (np.ndarray): The northward wind component.

        Returns:
            np.ndarray: The wind speed.
        """
        # Source: ECMWF
        # https://confluence.ecmwf.int/pages/viewpage.action?pageId=133262398
        return np.sqrt(u ** 2 + v ** 2)

    @staticmethod
    def get_wind_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Given the u and v components, get the wind direction.

        Args:
            u (np.ndarray): The eastward wind component.
            v (np.ndarray): The northward wind component.

        Returns:
            np.ndarray: The wind direction.
        """
        # Source: ECMWF
        # ϕ=mod(180+180/π atan2(u,v),360)
        # https://confluence.ecmwf.int/pages/viewpage.action?pageId=133262398
        return (180 + (180 / np.pi) * np.arctan2(u, v)) % 360

    def show_parameters_description(self):
        """Show description of the parameters in the dataset."""
        pretty_df = self.parameters_table
        pretty_df = pretty_df.reset_index()
        pretty_df["Description"] = (pretty_df["description_E"] +
                                    " (" + pretty_df["unit_name_E"] + ")")
        pretty_df = pretty_df.loc[:, ["name", "param_short", "Description"]]
        # Rename columns for display
        pretty_df.rename(columns={"name": "Parameter", "param_short": "Original name"})
        print(pretty_df.to_string(index=False))

    def _export_df(self, df: pd.DataFrame,
                   as_numpy: bool = False,
                   copy: bool = True, ) -> FrameArray:
        if as_numpy:
            stations = df.columns.unique(0)
            parameters = df.columns.unique(1)
            if not self.pad_missing_values:
                df = df_add_missing_columns(df, col0=stations, col1=parameters)
            df = df.values.reshape(-1, len(stations), len(parameters))
        elif copy:
            df = df.copy()
        return df

    def get_observations(
            self,
            stations: Optional[Union[str, List[str]]] = None,
            parameters: Optional[Union[str, List[str]]] = None,
            first_date: Optional[Union[str, pd.Timestamp]] = None,
            last_date: Optional[Union[str, pd.Timestamp]] = None,
            split: Optional[Literal['train', 'test']] = None,
            as_numpy: bool = False,
            return_mask: bool = False,
            copy: bool = True,
    ) -> Union[FrameArray, tuple[FrameArray, FrameArray]]:
        """Get observations for the specified stations and parameters.

        The observations are returned as a pandas DataFrame or numpy array,
        depending on the value of `as_numpy`. If `return_mask` is set to
        `True`, a tuple of (observations, mask) is returned.

        The observations are filtered based on the specified stations,
        parameters, and date range. If no filtering is applied, all
        observations are returned. The date range is inclusive of the
        start date and exclusive of the end date.

        Args:
            stations (str or list, optional): Station IDs to filter. If :obj:`None`,
                all stations are used.
                (default: :obj:`None`)
            parameters (str or list, optional): Parameter IDs to filter. If :obj:`None`,
                all parameters are used.
                (default: :obj:`None`)
            first_date (str or pd.Timestamp, optional): Start date for filtering.
                If :obj:`None`, no temporal filtering is applied.
                (default: :obj:`None`)
            last_date (str or pd.Timestamp, optional): End date for filtering.
                If :obj:`None`, no temporal filtering is applied.
                (default: :obj:`None`)
            split (Literal['train', 'test'], optional): Predefined data split to load. 
                If given, `first_date` and `last_date` must be None.
                (default: :obj:`None`)
            as_numpy (bool, optional): If :obj:`True`, return the observations as a
                :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`.
                (default: :obj:`False`)
            return_mask (bool, optional): If :obj:`True`, return the mask as well.
                (default: :obj:`False`)
            copy (bool, optional): If :obj:`True`, return a copy of the data.
                (default: :obj:`False`)

        Returns:
            FrameArray or tuple: The observations as a :class:`~pandas.DataFrame` or
                :class:`~numpy.ndarray` (if `as_numpy` is set to :obj:`True`).
                If `return_mask` is set to :obj:`True`, a tuple of
                :obj:`(observations, mask)` is returned.
        """
        # Get columns filter
        stations_loc = slice(None)
        if stations is not None:
            stations_loc = [stations] if isinstance(stations, str) else stations
        parameters_loc = slice(None)
        if parameters is not None:
            parameters_loc = [parameters] if isinstance(parameters, str) else parameters
        # Get index filter
        if split is not None:
            assert first_date is None and last_date is None, \
                "Can not provide both a split and first_date and/or last_date."
            if split == 'test':
                first_date = self.test_set_start
            elif split == 'train':
                last_date = self.test_set_start
            else:
                raise ValueError(
                    f"Invalid split value: {split}. Expected 'train' or 'test'."
                )
        if first_date is not None:
            first_date = pd.to_datetime(first_date, utc=True)
        if last_date is not None:
            last_date = pd.to_datetime(last_date, utc=True) - pd.Timedelta(minutes=1)
        index_loc = slice(first_date, last_date)

        # Get observations
        observations = self.observations.loc[index_loc, (stations_loc, parameters_loc)]
        observations = self._export_df(observations, as_numpy=as_numpy, copy=copy)
        if not return_mask:
            return observations

        # Get mask
        mask = self.mask.loc[index_loc, (stations_loc, parameters_loc)]
        mask = self._export_df(mask, as_numpy=as_numpy, copy=copy)
        return observations, mask

    def get_observation_windows(
            self,
            window_size: int,
            horizon_size: int,
            stations: Optional[Union[str, List[str]]] = None,
            parameters: Optional[Union[str, List[str]]] = None,
            first_date: Optional[Union[str, pd.Timestamp]] = None,
            last_date: Optional[Union[str, pd.Timestamp]] = None,
            split: Optional[Literal['train', 'test']] = None,
            as_xarray: bool = False) -> Windows:
        r"""Get sliding windows of observations and mask. See `get_windows`
        for more details.
        """
        if split is not None and split not in ['train', 'test']:
            raise ValueError(f"Invalid split value: {split}. "
                             "Expected 'train' or 'test'.")

        observations, mask = self.get_observations(
            stations=stations,
            parameters=parameters,
            first_date=first_date,
            last_date=last_date,
            split=split,
            as_numpy=False,
            return_mask=True,
            copy=True,
        )

        index = observations.index
        observations = self._export_df(observations, as_numpy=True, copy=False)
        mask = self._export_df(mask, as_numpy=True, copy=False)

        x = sliding_window_view(observations[:-horizon_size], window_size)
        mask_x = sliding_window_view(mask[:-horizon_size], window_size)
        index_x = sliding_window_view(index[:-horizon_size], window_size)
        y = sliding_window_view(observations[window_size:], horizon_size)
        mask_y = sliding_window_view(mask[window_size:], horizon_size)
        index_y = sliding_window_view(index[window_size:], horizon_size)

        windows = Windows(x=x,
                          mask_x=mask_x,
                          y=y,
                          mask_y=mask_y,
                          index_x=index_x,
                          index_y=index_y)

        if not as_xarray:
            return windows

        return self.np_windows_as_xr(windows, stations, parameters)

    def np_windows_as_xr(self, windows, stations, parameters):
        windows_coordinates = self._get_windows_coordinates(windows=windows,
                                                            stations=stations,
                                                            parameters=parameters)
        windows_xr = self._pack_windows_as_xarray(windows,
                                                  feature=windows_coordinates['features'],
                                                  reftime=windows_coordinates['reftime'],
                                                  lagtime=windows_coordinates['lagtime'],
                                                  leadtime=windows_coordinates['leadtime'],
                                                  stations=windows_coordinates['nat_abbr'])

        return windows_xr

    def align_windows(self, obs: Windows, nwp: "xr.Dataset", drop_extra_y_pars: bool, as_xarray: bool):

        xr = import_xarray()

        assert isinstance(nwp, xr.Dataset)
        for k in ["x", "mask_x", "y", "mask_y"]:
            assert isinstance(obs.__dict__[k], xr.Dataset)

        nwp_pars = list(nwp.data_vars)
        y_pars_init = list(obs.y.data_vars)
        if not (set(nwp_pars) <= set(y_pars_init)):
            raise ValueError(f"Aligning NWP forecasts with observations is possible only when the NWP parameters are also loaded as observations. "
                             f"The requested NWP parameters are {nwp_pars}, while the observation parameters are {y_pars_init}.")

        if drop_extra_y_pars:
            obs.y = obs.y[nwp_pars]
            obs.mask_y = obs.mask_y[nwp_pars]
        else:
            obs.nwp_to_y = nwp_pars

        #data
        obs.y, nwp_aligned = xr.align(obs.y, nwp, join='inner')
        obs.x, _ = xr.align(obs.x, obs.y, join='inner')
        #masks
        obs.mask_x, _ = xr.align(obs.mask_x, obs.x, join='inner')
        obs.mask_y, _ = xr.align(obs.mask_y, obs.y, join='inner')
        nwp_mask = nwp_aligned.notnull().all(dim="realization")
        assert nwp_mask["reftime"].identical(obs.mask_y["reftime"])
        nwp_pars_ = nwp_pars[0] if len(nwp_pars) == 1 else nwp_pars
        obs.mask_y[nwp_pars_] = obs.mask_y[nwp_pars_] & nwp_mask[nwp_pars_]
        #indices
        obs.index_x = timestamps_from_xr(obs.x, "lag")
        obs.index_y = timestamps_from_xr(obs.y, "lead")

        if as_xarray:
            return obs, nwp_aligned

        nwp_aligned = xr_to_np(a=nwp_aligned, stack_dim=-2, sample_dim=-1)
        obs.x = xr_to_np(obs.x)
        obs.mask_x = xr_to_np(obs.mask_x)
        obs.y = xr_to_np(obs.y)
        obs.mask_y = xr_to_np(obs.mask_y)

        if not drop_extra_y_pars:
            obs.nwp_to_y = [y_pars_init.index(p) for p in obs.nwp_to_y]

        return obs, nwp_aligned

    def get_windows(
            self,
            window_size: int,
            horizon_size: int,
            stations: Optional[Union[str, List[str]]] = None,
            parameters: Optional[Union[str, List[str]]] = None,
            first_date: Optional[Union[str, pd.Timestamp]] = None,
            last_date: Optional[Union[str, pd.Timestamp]] = None,
            split: Optional[Literal['train', 'test', 'nwp_test']] = None,
            nwp_parameters: Optional[Union[str, List[str]]] = None,
            drop_extra_y_pars: bool = True,
            as_xarray: bool = False
    ) -> Windows:
        """Get sliding windows of observations, mask of missing values and,
        when requested, NWP forecasts. The input data is reshaped into sliding
        windows of size (window_size, num_stations, num_channels) and the target 
        data is reshaped into sliding windows of size (horizon_size, num_stations,
        num_channels). The NWP forecasts are of shape (horizon_size, num_stations,
        num_channels, num_ensemble_members).

        Args:
            window_size (int): Size of the input window.
            horizon_size (int): Size of the output horizon.
            stations (str or list, optional): Station IDs to filter. If :obj:`None`,
                all stations are used.
                (default: :obj:`None`)
            parameters (str or list, optional): Parameter IDs to filter. If :obj:`None`,
                all parameters are used.
                (default: :obj:`None`)
            first_date (str or pd.Timestamp, optional): Start date for filtering.
                If :obj:`None`, no temporal filtering is applied.
                (default: :obj:`None`)
            last_date (str or pd.Timestamp, optional): End date for filtering.
                If :obj:`None`, no temporal filtering is applied.
                (default: :obj:`None`)
            split (Literal['train', 'test', 'nwp_test'], optional): Predefined data split to load.
                If given, `first_date` and `last_date` must be None.
                (default: :obj:`None`)
            nwp_parameter (str or list, optional): Defines which NWP parameters to return. If no parameter is
                is specified, none will be returned. If split is 'nwp_test', an NWP parameter must be specified.
                The windows will then then be aligned on the reference foreasting time available from
                the NWP model ICON-CH1-EPS, i.e., every 3 hours starting at midnight UTC. If a list is given, all those parameters
                are loaded. The nwp data is then available in Windows.nwp. If no split is specified but nwp_parameter is not None, the windows will nonetheless be aligned with the first available timestamp of the nwp data.
                (default: :obj:`None`)
            drop_extra_y_pars (bool, optional): If :obj:`False`, the returned parameters in `y` are selected
                according to `parameters`, as per `x`. If :obj:`True`, only the features corresponding 
                to `nwp_parameter` are returned.
                (default: :obj:`True`)
            as_xarray (bool, optional): If :obj:`True`, return the Windows with x, mask_x, y and mask_y as a
                :class:`~xarray.Dataset` instead of a :class:`~np.ndarray`.
                (default: :obj:`False`)

        Returns:
            Windows: a windowed view of observations and NWP forecasts.
        """
        if split is not None and split not in ['train', 'test', 'nwp_test']:
            raise ValueError(f"Invalid split value: {split}. "
                             "Expected 'train', 'test' or 'nwp_test'.")

        need_nwp_data = split == 'nwp_test' or nwp_parameters is not None
        if need_nwp_data:
            assert nwp_parameters is not None, "For split=nwp_test you must specify an ICON parameter to load."

            freq = pd.infer_freq(self.observations.index)
            assert pd.tseries.frequencies.to_offset(freq) == pd.Timedelta("1h"), (
                f'The loaded observations have a frequency of {self.observations.index.freqstr.lower()}, while '
                'the ICON forecasts are only available at 1h frequency. Please reload the dataset with the flag freq="h"'
            )

            assert horizon_size <= 33, "ICON-CH1-EPS data only forecasts up to 33h ahead. Use a smaller horizon size."

        windows = self.get_observation_windows(
            window_size=window_size,
            horizon_size=horizon_size,
            stations=stations,
            parameters=parameters,
            first_date=first_date,
            last_date=last_date,
            split="test" if split == "nwp_test" else split,
            as_xarray=need_nwp_data or as_xarray
        )

        if need_nwp_data:
            nwp_data = self.load_icon_data(icon_parameter=nwp_parameters)
            windows, nwp_aligned = self.align_windows(windows, nwp_data, drop_extra_y_pars, as_xarray)
            windows.nwp = nwp_aligned

        return windows

    def _get_windows_coordinates(self,
                                 windows: Windows,
                                 stations=None,
                                 parameters=None):
        assert windows.index_x is not None and windows.index_y is not None, \
            "Windows must have index_x and index_y to compute coordinates."
        # reftime is the time of the last observation in the input window
        reftime = pd.DatetimeIndex(windows.index_x[:, -1]).tz_localize(None)
        window_size = windows.x.shape[1]
        horizon_size = windows.y.shape[1]
        freq = self.observations.index.freq
        leadtime = pd.timedelta_range(start=1 * freq, periods=horizon_size, freq=freq)
        lagtime = pd.timedelta_range(start=-(window_size - 1) * freq, end=0, freq=freq)

        if isinstance(stations, str):
            stations = [stations]

        if isinstance(parameters, str):
            parameters = [parameters]

        nat_abbr = np.array(self.stations if stations is None else stations)
        features = np.array(self.parameters if parameters is None else parameters)

        return {
            'reftime': reftime,
            'leadtime': leadtime,
            'lagtime': lagtime,
            'nat_abbr': nat_abbr,
            'features': features
        }

    def _pack_windows_as_xarray(self,
                                windows: Windows,
                                feature: Union[str, np.ndarray],
                                reftime: Union[List, np.ndarray, pd.DatetimeIndex],
                                lagtime: Sequence[pd.Timedelta],
                                leadtime: Sequence[pd.Timedelta],
                                stations: Sequence[str]) -> Windows:

        xr = import_xarray()

        if isinstance(reftime, pd.DatetimeIndex):
            reftime = reftime.copy()
            reftime.name = None

        def get_data_vars(data, feature, x_or_y: str):
            if data.ndim == 3:
                assert isinstance(feature, str) or len(feature) == 1, (
                    f"Data has 3 dimensions. Expected feature to be a string or "
                    f"a 1D array/list of length 1. Given: {feature}"
                )
                feature = [feature] if isinstance(feature, str) else feature
                data = np.expand_dims(data, axis=-1)

            assert data.ndim == 4
            assert len(feature) == data.shape[-1]
            second_dim = 'lag' if x_or_y == 'x' else 'lead'
            return {f: (["reftime", second_dim, "nat_abbr"], data[..., i])
                    for i, f in enumerate(feature)}

        common_attrs = {
            "LICENSE": "CC-BY-4.0",
            "source": "MeteoSwiss (SwissMetNet)",
            "time_zone": "UTC",
        }
        x = xr.Dataset(
            get_data_vars(data=windows.x, feature=feature, x_or_y='x'),
            coords={
                "reftime": reftime,
                "lag": lagtime,
                "nat_abbr": stations
            },
            attrs={
                "parameter": ', '.join(feature) if not isinstance(feature, str) else feature,
                "description": "Represents past measurements with respect to each reftime",
                **common_attrs
            }
        )

        mask_x = xr.Dataset(
            get_data_vars(data=windows.mask_x, feature=feature, x_or_y='x'),
            coords={
                "reftime": reftime,
                "lag": lagtime,
                "nat_abbr": stations
            },
            attrs={
                "parameter": ', '.join(f'(mask){v}' for v in feature) if not isinstance(feature, str) else feature,
                "description": "Represents the availability mask of past measurements with respect to each reftime",
                **common_attrs
            }
        )

        y = xr.Dataset(
            get_data_vars(data=windows.y, feature=feature, x_or_y='y'),
            coords={
                "reftime": reftime,
                "lead": leadtime,
                "nat_abbr": stations
            },
            attrs={
                "parameter": ', '.join(feature) if not isinstance(feature, str) else feature,
                "description": "Represents future measurements with respect to each reftime",
                **common_attrs
            }
        )

        mask_y = xr.Dataset(
            get_data_vars(data=windows.mask_y, feature=feature, x_or_y='y'),
            coords={
                "reftime": reftime,
                "lead": leadtime,
                "nat_abbr": stations
            },
            attrs={
                "parameter": ', '.join(f'(mask){v}' for v in feature) if not isinstance(feature, str) else feature,
                "description": "Represents the availability mask of future measurements with respect to each reftime",
                **common_attrs
            }
        )

        return Windows(x=x,
                       mask_x=mask_x,
                       y=y,
                       mask_y=mask_y,
                       index_x=windows.index_x,
                       index_y=windows.index_y)

    def load_icon_data(self, icon_parameter: Union[str, Sequence[str]]) -> "xr.Dataset":
        """Returns an Xarray dataset with the requested parameters.

        Args:
            icon_parameter (str): The ICON parameters. Must be within of self.available_icon.

        Raises:
            ValueError: If the corresponding zarr is not available.

        Returns:
            xr.Dataset: The dataset with the ICON forecasts.
        """
        if icon_parameter == "all":
            icon_parameter == self.available_icon

        if isinstance(icon_parameter, list):
            xr = import_xarray()
            out = [self.load_icon_data(p) for p in icon_parameter]
            m = xr.merge(out, compat="identical", join="outer")
            m.attrs["parameter"] = ", ".join([o.attrs["parameter"] for o in out]) + "."
            periods = [o.attrs["period"] for o in out]
            if len(set(periods)) > 1:
                m.attrs["period"] = ", ".join(periods) + "."
            return m

        if icon_parameter in self.available_icon:
            xr = import_xarray()
            return xr.open_zarr(self.required_files_paths[f'icon_{icon_parameter}'],
                                decode_timedelta=True)
        else:
            raise ValueError(f'Can not load ICON parameter {icon_parameter}. '
                             f'It must be one of {self.available_icon}.')

    def had_values_before(self, cutoff_time: Union[str, pd.Timestamp]) -> pd.Series:
        """Returns a binary masks that informs whether a station measured a parameter
        before the given cutoff time. This information is particularly important when
        the task at hand relies on an inductive or transductive learning procedure.

        Args:
            cutoff_time (str or pd.Timestamp): The timestamp (UTC) representing the
                cutoff time.

        Returns:
            pd.Series: A binary series with a multi-index (station, parameter).
        """
        if isinstance(cutoff_time, str):
            cutoff_time = pd.Timestamp(cutoff_time, tz="UTC")

        df_before = self.mask.loc[:cutoff_time - pd.Timedelta("1ns")]
        return df_before.any(axis=0)
