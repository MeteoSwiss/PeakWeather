peakweather
===========

The main class of the PeakWeather package is :class:`~peakweather.dataset.PeakWeatherDataset`,
which provides an easy-to-use interface to download, access, and manipulate the
PeakWeather dataset.

.. autosummary::
    :nosignatures:

        ~peakweather.dataset.PeakWeatherDataset

The ``peakweather`` package also includes several modules for data input/output,
utilities, and API access. These modules provide functionality for reading and
writing data, performing data transformations, and accessing the dataset's API.

.. toctree::
   :maxdepth: 2
   :caption: API

   dataset
   io
   utils
