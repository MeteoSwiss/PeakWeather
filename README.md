<div align=center>

<img src="https://raw.githubusercontent.com/MeteoSwiss/PeakWeather/main/docs/source/_static/peakweather_logo.png" alt="PeakWeather Logo" width="128"/>

# PeakWeather

[Documentation](https://peakweather.readthedocs.io/) — [Demo notebook](https://peakweather.readthedocs.io/latest/examples/peakweather_demo.html) — [Data](https://huggingface.co/datasets/meteoswiss/PeakWeather) — [Paper](https://arxiv.org/abs/2506.13652) 

</div>

**PeakWeather** is a high-resolution, benchmark-ready **dataset** for spatiotemporal weather modeling.
This repository provides the Python library to load and preprocess the dataset.
For full documentation, visit [peakweather.readthedocs.io](https://peakweather.readthedocs.io/).

### Key Features

- **High-resolution observations** — 10-minute interval data spanning 2017–2025 over 302 SwissMetNet stations across Switzerland
- **Multiple variables** — Temperature, pressure, humidity, wind, radiation, precipitation and more
- **Topographic descriptors** — Elevation, slope, aspect, and surface roughness to describe the Swiss complex terrain
- **NWP baselines** — Ensemble forecasts from ICON-CH1-EPS, the state-of-the-art numerical prediction model operational at MeteoSwiss
- **Ideal for** — Time series forecasting, missing data imputation, virtual sensing, graph structure learning, and more

<div align=center>
  <img src="https://raw.githubusercontent.com/MeteoSwiss/PeakWeather/main/docs/source/_static/stations.png" alt="Stations" width="350"/>
</div>

The dataset is hosted on [Hugging Face](https://huggingface.co/datasets/MeteoSwiss/PeakWeather) and introduced in the paper:

> [**PeakWeather: MeteoSwiss Weather Station Measurements for Spatiotemporal Deep Learning**](https://arxiv.org/abs/2506.13652)  
> *Daniele Zambon, Michele Cattaneo, Ivan Marisca, Jonas Bhend, Daniele Nerini, Cesare Alippi.* Preprint 2025.


## Quickstart

### Install

Install PeakWeather from PyPI:

```shell
pip install peakweather
```

This installs the base package with support for station measurements and NWP predictions.
To also access the topographical descriptors, install the extra dependencies:

```shell
pip install peakweather[topography]
```

Alternatively, install directly from GitHub to stay up to date with the latest developments:

```shell
pip install git+https://github.com/MeteoSwiss/PeakWeather.git 
pip install "peakweather[topography] @ git+https://github.com/MeteoSwiss/PeakWeather"  # With extras
```

### Load the dataset

On first use, data is automatically downloaded from Hugging Face:

```python
from peakweather.dataset import PeakWeatherDataset

ds = PeakWeatherDataset(root=None)            # Download to the current directory
ds = PeakWeatherDataset(root="path/to/data")  # Or to the specified path
```

If data is already present at the given path, it is loaded directly.

### Get observations

```python
# Single station, all parameters
ds.get_observations(stations='KLO') 
# Multiple stations, specific parameters
ds.get_observations(stations=['KLO', 'GRO'], parameters=['pressure', 'temperature']) 
```

| datetime                  | ('KLO', 'pressure') | ('KLO', 'temperature') |
|:--------------------------|---------------------:|-----------------------:|
| 2017-01-01 00:00:00+00:00 |                977.8 |                   -3.3 |
| 2017-01-01 00:10:00+00:00 |                977.7 |                   -3.5 |
| 2017-01-01 00:20:00+00:00 |                977.6 |                   -3.5 |
| ...                       |                  ... |                    ... |

For detailed usage, see the [full documentation](https://peakweather.readthedocs.io/), the provided [example](https://github.com/MeteoSwiss/PeakWeather/blob/main/example.py), and the [demo notebook](https://peakweather.readthedocs.io/latest/examples/peakweather_demo.html).


## Citation

If you use PeakWeather in your research, please cite:

```bibtex
@misc{zambon2025peakweather,
  title={PeakWeather: MeteoSwiss Weather Station Measurements for Spatiotemporal Deep Learning}, 
  author={Zambon, Daniele and Cattaneo, Michele and Marisca, Ivan and Bhend, Jonas and Nerini, Daniele and Alippi, Cesare},
  year={2025},
  eprint={2506.13652},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2506.13652}, 
}
```


## Authors

- [Daniele Zambon](https://dzambon.github.io)
- [Michele Cattaneo](https://github.com/MicheleCattaneo)
- [Ivan Marisca](https://marshka.github.io)
