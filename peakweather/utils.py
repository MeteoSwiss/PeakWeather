from typing import Optional

import numpy as np
import pandas as pd


def to_pandas_freq(freq: str):
    """Convert a frequency string to a pandas frequency object.

    Args:
        freq (str): The frequency string.

    Returns:
        pd.DateOffset: The pandas frequency object.

    Raises:
        ValueError: If the frequency string is not valid.
    """
    try:
        freq = pd.tseries.frequencies.to_offset(freq)
    except ValueError:
        raise ValueError(f"Value '{freq}' is not a valid frequency.")
    return freq


def df_add_missing_columns(df: pd.DataFrame, col0=None, col1=None) -> pd.DataFrame:
    """Add missing columns to a MultiIndex :class:`~pandas.DataFrame` with NaN values.

    Args:
        df (pd.DataFrame): The input :class:`~pandas.DataFrame`.
        col0 (list, optional): The first level of the :class:`~pandas.MultiIndex`
            columns. If :obj:`None`, will use the existing columns.
        col1 (list, optional): The second level of the :class:`~pandas.MultiIndex`
            columns. If :obj:`None`, will use the existing columns.

    Returns:
        pd.DataFrame: The :class:`~pandas.DataFrame` with missing columns added.
    """
    if col0 is None:
        col0 = df.columns.unique(0)
    if col1 is None:
        col1 = df.columns.unique(1)
    columns = pd.MultiIndex.from_product((col0, col1))
    return df.reindex(columns=columns).astype("float32")


def sliding_window_view(data: np.ndarray, window_size: int) -> np.ndarray:
    r"""Creates a sliding window view of the input data.

    Args:
        data (np.ndarray): The input data with shape ``(num_time_steps, *)``.
        window_size (int): The size of the sliding window.

    Returns:
        np.ndarray: The sliding window view of the input data with shape
            ``(num_windows, window_size, *)``.
    """
    windows = np.lib.stride_tricks.sliding_window_view(
        data, window_shape=window_size, axis=0
    )
    windows = np.moveaxis(windows, -1, 1)
    return windows


def xr_to_np(
    a: "xr.Dataset",
    pars: Optional[list] = None,
    sample_dim: Optional[int] = None,
    stack_dim: int = -1,
) -> np.ndarray:
    r"""Extract variables from an :class:`~xarray.Dataset` and return them as
    a stacked :class:`~numpy.ndarray`.

    Args:
        a (xarray.Dataset): The input dataset containing one or more data variables.
        pars (list[str], optional): The names of the variables to extract. If
            `None`, all data variables in `a` are used.
        sample_dim (int, optional): The dimension containing the samples in `a`,
            if present. If `sample_dim` is an `int`, the `sample_dim` dimension
            is rearranged as leading dimension (samples, a.shape[~sample_dim]);
            `None` indicates no sampling dimension to be moved.
        stack_dim (int): The dimension along which the arrays are stacked.

    Returns:
        np.ndarray: A NumPy array where the selected variables are stacked along
            the last axis. If each variable has shape `(*dims)`, the returned
            array has shape `(*dims, num_vars)`.
    """
    if pars is None:
        pars = list(a.data_vars)
    out = np.stack([a[p].data for p in pars], axis=stack_dim)

    if sample_dim is not None:
        sample_dim = (
            sample_dim if sample_dim > 0 else sample_dim + out.ndim - 1
        )  # dim in a
        stack_dim = stack_dim if stack_dim > 0 else stack_dim + out.ndim  # dim in out
        if sample_dim >= stack_dim:  # dim in out
            sample_dim += 1
        out = np.moveaxis(out, sample_dim, 0)

    return out


def timestamps_from_xr(
    ds: "xr.Dataset", delta: str, tz: Optional[str] = "UTC"
) -> np.ndarray:
    r"""Compute a 2D array of timezone-aware timestamps by combining a reference
    time coordinate with a time-delta coordinate.

    Args:
        ds (xarray.Dataset): The input dataset containing a `reftime` coordinate
            of type `datetime64[ns]` and a time-delta coordinate.
        delta (str): The name of the offset coordinate (e.g., `lag` or `lead`) of
            type `timedelta64[ns]`.
        tz (str): Timezone.

    Returns:
        np.ndarray: A 2D array of shape `(num_reftime, num_deltas)` containing
            `pandas.Timestamp` objects localized to UTC, where each entry is
            `reftime[i] + offset[j]`.
    """
    idx = ds["reftime"].data[:, None] + ds[delta].data[None, :]
    idx = pd.to_datetime(idx).tz_localize(tz).to_numpy()
    return idx
