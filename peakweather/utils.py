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
        data (np.ndarray): The input data with shape
            (num_time_steps, num_stations, num_channels).
        window_size (int): The size of the sliding window.

    Returns:
        np.ndarray: The sliding window view of the input data with shape
            (num_windows, window_size, num_stations, num_channels).
    """
    windows = np.lib.stride_tricks.sliding_window_view(
        data,
        window_shape=window_size,
        axis=0,
    )
    windows = np.transpose(windows, (0, 3, 1, 2))
    return windows
