"""PeakWeather is a high-quality meteorological dataset derived from SwissMetNet,
the automated measurement network operated by MeteoSwiss. It offers a robust
resource for research and applications in spatiotemporal modeling.

PeakWeather includes high-frequency meteorological observations recorded every
10 minutes, collected from ground stations distributed across Switzerland. The dataset
also provides high-resolution topographic features at 50-meter resolution and ensemble
forecasts from the ICON-CH1-EPS operational numerical weather prediction (NWP)
model. The dataset is described in more details in `"PeakWeather: MeteoSwiss Weather
Station Measurements for Spatiotemporal Deep Learning"
<https://arxiv.org/abs/2506.13652>`_ (Zambon et al., 2025).
"""

from .dataset import PeakWeatherDataset, Windows

__version__ = PeakWeatherDataset.__version__

__all__ = ["PeakWeatherDataset", "Windows"]
