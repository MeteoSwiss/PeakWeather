import numpy as np
import pandas as pd
import pytest
import xarray as xr

from peakweather import PeakWeatherDataset, Windows

TEST_DATA_ROOT = "./data"


def test_dataset_shapes():
    ds = PeakWeatherDataset(root=TEST_DATA_ROOT,
                            extended_nwp_pars='all',
                            freq='h',
                            years=2017,
                            compute_uv=False,
                            parameters=['temperature', 'wind_speed'])

    assert len(ds.parameters) == 2

    obs = ds.get_observations(stations=['ABO', 'KLO', 'GRO'], as_numpy=True)
    assert obs.shape[-1] == 2
    assert obs.shape[-2] == 3

    obs = ds.get_observations(stations=['ABO', 'KLO', 'GRO'], parameters='wind_speed', as_numpy=True)
    assert obs.shape[-1] == 1
    assert obs.shape[-2] == 3

    with pytest.raises(KeyError):
        ds.get_observations(parameters='pressure', as_numpy=True)


def test_temporal_subset():
    ds_meteo_2017 = PeakWeatherDataset(root=TEST_DATA_ROOT,
                        extended_nwp_pars='all',
                        freq='h',
                        years=2017)

    ds_meteo = PeakWeatherDataset(root=TEST_DATA_ROOT,
                        freq='h',
                        extended_nwp_pars='all')

    pd.testing.assert_frame_equal(ds_meteo_2017.get_observations(last_date='2017-12-31 23:59'),
                                  ds_meteo.get_observations(stations=ds_meteo_2017.stations, last_date='2017-12-31 23:59'))


def test_station_type():
    ds_station = PeakWeatherDataset(root=TEST_DATA_ROOT,
                        extended_nwp_pars='all',
                        freq='h',
                        station_type='meteo_station')

    with pytest.raises(ValueError):
        # compute_uv is set to True by default,
        # which can not be if station type is rain gauge
        ds_gauge = PeakWeatherDataset(root=TEST_DATA_ROOT,
                            extended_nwp_pars='all',
                            station_type='rain_gauge',
                            freq='h')

    ds_gauge = PeakWeatherDataset(root=TEST_DATA_ROOT,
                            extended_nwp_pars='all',
                            station_type='rain_gauge',
                            compute_uv=False,
                            freq='h')

    assert len(ds_gauge.parameters) == 2
    assert 'precipitation' in ds_gauge.parameters
    assert 'temperature' in ds_gauge.parameters

    assert len(ds_gauge.stations) + len(ds_station.stations) == 302

    assert not set(ds_station.stations).intersection(set(ds_gauge.stations)), "Intersection between gauges and stations should be empty."


def test_dataset_computes_uv():
    ds = PeakWeatherDataset(root=TEST_DATA_ROOT,
                        freq='h',
                        years=2017,
                        compute_uv=True)

    assert 'wind_v' in ds.parameters and 'wind_u' in ds.parameters and 'wind_direction' in ds.parameters


def test_splits():
    ds = PeakWeatherDataset(root=TEST_DATA_ROOT,
                        freq='h',
                        compute_uv=True)

    train_split = ds.get_observations(split='train')
    test_split = ds.get_observations(split='test')

    # assert the train and test split cover the whole timespan
    # (we do not enforce a val split, that is up to the user to take from train)
    combined_index = train_split.index.union(test_split.index)
    assert combined_index.equals(ds.observations.index)

    assert isinstance(ds.get_observations(split='test', as_numpy=True), np.ndarray)
    assert isinstance(ds.get_observations(split='test', as_numpy=True, return_mask=True), tuple)


def test_windows_as_xarray():
    ds = PeakWeatherDataset(root=TEST_DATA_ROOT,
                        freq='h',
                        compute_uv=True)

    w_size = 24
    h_size = 12
    train_windows_xarray = ds.get_windows(window_size=w_size,
                                    horizon_size=h_size,
                                    split='train',
                                    as_xarray=True)

    print()

    # assert the first lag is 0 ns (it is now)
    assert train_windows_xarray.x['lag'].data[-1] == np.timedelta64(0,'ns')

    # assert that reftime + lagtime (broadcasted along the window) corresponds to index_x (lag is negative)
    induced_index_x = (train_windows_xarray.x['reftime'].data[:, None] + train_windows_xarray.x['lag'].data[None, :])
    induced_index_x = pd.to_datetime(induced_index_x).tz_localize("UTC").to_numpy()

    assert induced_index_x.shape == train_windows_xarray.index_x.shape
    assert (induced_index_x == train_windows_xarray.index_x).all()

    # assert that reftime + leadtime (broadcasted along the horizon) corresponds to index_y (lag is negative)
    induced_index_y = (train_windows_xarray.y['reftime'].data[:, None] + train_windows_xarray.y['lead'].data[None, :])
    induced_index_y = pd.to_datetime(induced_index_y).tz_localize("UTC").to_numpy()

    assert induced_index_y.shape == train_windows_xarray.index_y.shape
    assert (induced_index_y == train_windows_xarray.index_y).all()

    # check timestamps in train_windows_xarray.x|y are equal to train_windows_xarray.index_x|y via utility timestamps_from_xr
    from peakweather.utils import timestamps_from_xr
    np.testing.assert_array_equal(timestamps_from_xr(train_windows_xarray.x, "lag"), train_windows_xarray.index_x)
    np.testing.assert_array_equal(timestamps_from_xr(train_windows_xarray.y, "lead"), train_windows_xarray.index_y)

    # assert that get_observation_windows and get_windows return the same windows
    train_windows_xarray2 = ds.get_observation_windows(window_size=w_size,
                                                       horizon_size=h_size,
                                                       split='train',
                                                       as_xarray=True)
    for k in ["x", "mask_x", "y", "mask_y"]:
        xr.testing.assert_identical(train_windows_xarray.__dict__[k], train_windows_xarray2.__dict__[k])
    for k in ["index_x", "index_y"]:
        np.testing.assert_array_equal(train_windows_xarray.__dict__[k], train_windows_xarray2.__dict__[k])


def test_windows_split():
    ds = PeakWeatherDataset(root=TEST_DATA_ROOT,
                            freq='h',
                            compute_uv=True)

    w_size = 24
    h_size = 12
    stations = ['KLO', 'GRO', 'ABO']
    train_windows = ds.get_windows(window_size=w_size,
                                   horizon_size=h_size,
                                   split='train')

    assert train_windows.x.shape[1] == w_size
    assert train_windows.x.shape[:2] == train_windows.index_x.shape
    assert train_windows.x.shape[2] == 302
    assert train_windows.y.shape[1] == h_size
    assert train_windows.y.shape[:2] == train_windows.index_y.shape
    assert train_windows.y.shape[2] == 302

    train_windows = ds.get_windows(window_size=w_size,
                                   horizon_size=h_size,
                                   split='train',
                                   stations=stations)

    assert train_windows.x.shape[1] == w_size
    assert train_windows.x.shape[:2] == train_windows.index_x.shape
    assert train_windows.x.shape[2] == 3
    assert train_windows.y.shape[1] == h_size
    assert train_windows.y.shape[:2] == train_windows.index_y.shape
    assert train_windows.y.shape[2] == 3

    train_windows = ds.get_windows(window_size=w_size,
                                   horizon_size=h_size,
                                   split='train',
                                   stations=stations,
                                   parameters=['temperature', 'pressure'])

    assert train_windows.x.shape[1] == w_size
    assert train_windows.x.shape[2] == 3
    assert train_windows.x.shape[-1] == 2
    assert train_windows.y.shape[1] == h_size
    assert train_windows.y.shape[2] == 3
    assert train_windows.y.shape[-1] == 2

    ds = PeakWeatherDataset(root=TEST_DATA_ROOT,
                            freq='h',
                            extended_nwp_pars='all',
                            compute_uv=True)

    icon_windows = ds.get_windows(window_size=w_size,
                                  horizon_size=h_size,
                                  split='nwp_test',
                                  stations=stations,
                                  nwp_parameters='pressure',
                                  as_xarray=False)

    assert isinstance(icon_windows, Windows)

    assert isinstance(icon_windows.x, np.ndarray)
    assert isinstance(icon_windows.mask_x, np.ndarray)
    assert isinstance(icon_windows.index_x, np.ndarray)
    assert isinstance(icon_windows.y, np.ndarray)
    assert isinstance(icon_windows.mask_x, np.ndarray)
    assert isinstance(icon_windows.index_y, np.ndarray)
    assert isinstance(icon_windows.nwp, np.ndarray)

    # (t, w, n, f)
    assert icon_windows.x.shape[1] == w_size
    assert icon_windows.x.shape[2] == len(stations)
    # (t, h, n, f)
    assert icon_windows.y.shape[0] == icon_windows.x.shape[0]
    assert icon_windows.y.shape[1] == h_size
    assert icon_windows.y.shape[2] == len(stations)
    # (s, t, h, n, f)
    assert icon_windows.nwp.shape[0] == 11 # ensemble members
    assert icon_windows.nwp.shape[1] == icon_windows.y.shape[0]
    assert icon_windows.nwp.shape[2] == h_size
    assert icon_windows.nwp.shape[3] == len(stations)

    icon_windows = ds.get_windows(window_size=w_size,
                                  horizon_size=h_size,
                                  split='nwp_test',
                                  stations=stations,
                                  nwp_parameters='pressure',
                                  as_xarray=True)

    assert isinstance(icon_windows, Windows)

    assert isinstance(icon_windows.x, xr.Dataset)
    assert isinstance(icon_windows.mask_x, xr.Dataset)
    assert isinstance(icon_windows.index_x, np.ndarray)
    assert isinstance(icon_windows.y, xr.Dataset)
    assert isinstance(icon_windows.mask_x, xr.Dataset)
    assert isinstance(icon_windows.index_y, np.ndarray)

    assert isinstance(icon_windows.nwp, xr.Dataset)


def test_nwp_split():

    ds = PeakWeatherDataset(root=TEST_DATA_ROOT,
                            freq='h',
                            extended_nwp_pars=['pressure', 'temperature', 'sunshine'],
                            compute_uv=True)

    icon_fcasts = ds.get_windows(window_size=1,
                                 horizon_size=33,
                                 split=None,
                                 parameters=['temperature', 'pressure', 'wind_speed'],
                                 stations=['KLO', 'GRO'],
                                 nwp_parameters='pressure',
                                 as_xarray=False)

    assert icon_fcasts.index_y[0, 0] < ds.test_set_start

    icon_fcasts_xr = ds.get_windows(window_size=1,
                                    horizon_size=33,
                                    split="nwp_test",
                                    parameters=['temperature', 'pressure', 'wind_speed'],
                                    stations=['KLO', 'GRO'],
                                    nwp_parameters='pressure',
                                    as_xarray=True)

    assert icon_fcasts_xr.index_y[0, 0] >= ds.test_set_start


    nwp_parameters = ["sunshine"]
    windows = ds.get_windows(
        window_size=24,  # number of lookback time steps
        horizon_size=6,  # number of lead times to be predicted
        # stations=['ABO', 'KLO'],
        split="nwp_test",
        nwp_parameters=nwp_parameters,
        drop_extra_y_pars=False,
        as_xarray=False
    )

    # [windows, window|horizon_size, stations, parameters]
    assert windows.x.shape[0] == windows.y.shape[0]
    assert windows.x.shape[2:] == windows.y.shape[2:]
    assert windows.x.shape == windows.mask_x.shape
    assert windows.y.shape == windows.mask_y.shape

    assert windows.nwp.shape[0] == 11 # ensemble dimension
    assert windows.y.shape[:3] == windows.nwp.shape[1:4]
    assert windows.nwp.shape[-1] == len(nwp_parameters)


def test_nwp_missing_data():

    dataset = PeakWeatherDataset(root=TEST_DATA_ROOT, freq="1h", extended_nwp_pars="all")

    for p in dataset.available_icon:
        ds = dataset.load_icon_data(p)
        nulls = ds.isnull().sum()[p].item()
        if nulls > 0:
            # null values should only at lead 0
            analysis_vals = len(ds.reftime) * len(ds.nat_abbr) * len(ds.realization)
            assert nulls == analysis_vals
            assert ds.sel(lead=np.timedelta64(0, "ns")).isnull().all()[p].item()


def test_get_multiple_nwp_in_windows():
    dataset = PeakWeatherDataset(root=TEST_DATA_ROOT, freq="1h", extended_nwp_pars="all")

    nwp_vars = ["temperature", "sunshine", "wind_gust", "precipitation", "pressure"]

    windows = dataset.get_windows(window_size=1,
                             horizon_size=33,
                             split=None,
                             nwp_parameters=nwp_vars,
                             drop_extra_y_pars=True,
                             as_xarray=True)

    assert (windows.nwp.reftime == windows.y.reftime).all()
    assert (windows.nwp.reftime == windows.x.reftime).all()

    # assert reftime + lead (shape = [windows,33]) of the NWP is the same as windows.index_y (shape = [windows,33])
    nwp_dates = (windows.nwp.reftime + windows.nwp.lead).values
    y_dates = np.stack(
        [pd.to_datetime(windows.index_y[:, j]).tz_localize(None).to_numpy() for j in range(windows.index_y.shape[1])],
        axis=1
    )
    assert (nwp_dates == y_dates).all()

    # Assert that the data variables in the nwp xr.Dataset corresponds to the requested ones.
    (np.array(list(windows.nwp.data_vars)) == np.array(nwp_vars)).all()
    (np.array(list(windows.nwp.data_vars)) == np.array(list(windows.y.data_vars))).all()


def test_nwp_alignment():

    nwp_pars = ['sunshine', 'humidity', 'precipitation']

    ds = PeakWeatherDataset(root=TEST_DATA_ROOT, freq="h", extended_nwp_pars=nwp_pars)

    windows = ds.get_windows(window_size=1,
                             horizon_size=33,
                             split=None,
                             stations=['KLO', 'GRO'],
                             nwp_parameters=nwp_pars,
                             drop_extra_y_pars=True,
                             as_xarray=False)

    assert windows.y.shape == windows.nwp.shape[1:]
    assert windows.y.shape == windows.mask_y.shape

    margin = 2
    #select region without missing data
    safe_region = windows.mask_y.all((1, 2, 3))
    safe_region[:margin+1] = False
    safe_region[-margin-1:] = False
    misvals = np.where(~safe_region)[0]
    for i in misvals:
        safe_region[i - margin: i + margin + 1] = False
    safe_idx = np.where(safe_region)[0]
    assert windows.mask_y[safe_idx].all()
    for i in range(-margin, margin+1):
        assert not np.isnan(windows.nwp[:, safe_idx+i]).any()
    y = windows.y[safe_idx]
    #compute pred error
    adiff = lambda nwp:  np.abs(y - np.median(nwp, 0))
    diff_ref = adiff(windows.nwp[:, safe_idx])
    err_glob = diff_ref.sum()
    err_var = diff_ref.sum(axis=(0, 1, 2))
    err_stat = diff_ref.sum(axis=(0, 1, 3))
    #compute pred error with shifted y_hat
    for i in range(-margin, margin+1):
        diff_offset = adiff(windows.nwp[:, safe_idx+i])
        assert err_glob <= np.nansum(diff_offset)
        assert (err_var <= diff_offset.sum(axis=(0, 1, 2))).all()
        assert (err_stat <= diff_offset.sum(axis=(0, 1, 3))).all()

    windows_ = ds.get_windows(window_size=1,
                             horizon_size=33,
                             split=None,
                             stations=['KLO', 'GRO'],
                             parameters=nwp_pars,
                             nwp_parameters=nwp_pars,
                             drop_extra_y_pars=True,
                             as_xarray=False)

    assert np.allclose(windows_.y, windows.y)
    assert np.allclose(windows_.nwp, windows.nwp, equal_nan=True)
    assert np.allclose(windows_.mask_y, windows.mask_y)

    windows = ds.get_windows(window_size=1,
                             horizon_size=33,
                             split=None,
                             stations=['KLO', 'GRO'],
                             nwp_parameters=nwp_pars[0],
                             drop_extra_y_pars=False,
                             as_xarray=False)

    safe_mask = windows.mask_y.any(-1, keepdims=True)
    adiff_f = lambda f:  (np.abs(windows.y[..., f] - np.nanmedian(windows.nwp, 0)) * safe_mask).sum()
    err_ref = adiff_f(windows.nwp_to_y)
    for f in range(windows.y.shape[-1]):
        assert err_ref <= adiff_f([f])

    windows_ = ds.get_windows(window_size=1,
                             horizon_size=33,
                             split=None,
                             stations=['KLO', 'GRO'],
                             parameters=nwp_pars,
                             nwp_parameters=nwp_pars[0],
                             drop_extra_y_pars=False,
                             as_xarray=False)

    assert np.allclose(windows_.nwp, windows.nwp, equal_nan=True)

    windows__ = ds.get_windows(window_size=1,
                             horizon_size=33,
                             split=None,
                             stations=['KLO', 'GRO'],
                             nwp_parameters=nwp_pars[0],
                             drop_extra_y_pars=True,
                             as_xarray=False)

    assert np.allclose(windows__.nwp, windows.nwp, equal_nan=True)


def test_extra_nwp_pars():
    ds = PeakWeatherDataset(root=TEST_DATA_ROOT,
                            freq='h',
                            years=2025,
                            compute_uv=True,
                            parameters=['temperature', 'wind_speed', "wind_u"],
                            extended_nwp_pars=['temperature', 'wind_u'])

    with pytest.raises(ValueError):
        ds.get_windows(window_size=3,
                       horizon_size=13,
                       stations=['KLO', 'GRO'],
                       parameters=['temperature', 'wind_speed'],
                       nwp_parameters=['temperature', 'wind_u'])

    w = ds.get_windows(window_size=3,
                       horizon_size=13,
                       stations=['KLO', 'GRO'],
                       parameters=['temperature', 'wind_speed', 'wind_u'],
                       nwp_parameters=['temperature', 'wind_u'])
