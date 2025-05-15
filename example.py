# %% Load dataset
from peakweather import PeakWeatherDataset

dataset = PeakWeatherDataset(
    root="data",  # Path to the dataset
    pad_missing_variables=True,  # Pad missing variables with NaN
    years=None,  # Years to include in the dataset (None for all)
    parameters=None,  # Parameters to include in the dataset (None for all)
    extended_topo_vars="none",  # Optional extended topographic variables
    imputation_method="zero",  # Method for imputing missing values
    freq="h",  # Frequency of the data (e.g., "h" for hourly)
    compute_uv=True,  # Compute u and v components of wind
    station_type="meteo_station"  # Which station type to load (None for all)
)
print(dataset)

# %% Show dataset information
print(f"Number of time steps: {dataset.num_time_steps}")

print(f"Number of stations: {dataset.num_stations}")
print(dataset.stations_table.head(10))

print(f"Number of parameters: {dataset.num_parameters}")
print(f"Parameters")
dataset.show_parameters_description()

# %% Show data
print(f"Observations shape: {dataset.observations.shape}")
print(dataset.observations.head(10))

# Get observations and mask for each parameter
for param in dataset.parameters_table.index:
    p_data, p_mask = dataset.get_observations(parameters=param,
                                              as_numpy=True,
                                              return_mask=True)
    print(f"Parameter {param} availability: {p_mask.mean():.2%}")

# %% Get observations for a specific station and parameters
print(f"Get wind speed and direction for station KLO")
klo_data = dataset.get_observations(stations="KLO",
                                    parameters=["wind_speed", "wind_direction"],
                                    as_numpy=True)
print(f"KLO data shape: {klo_data.shape}")
print(f"KLO maximum wind speed: {klo_data[..., 0].max():.2f} m/s")

# %% Get windows
window_size = 12
lead_times = 3
print(f"Get observations as windows with a sliding window of size {window_size} "
      f"and lead time {lead_times}")
windows = dataset.get_windows(window_size=window_size, horizon_size=lead_times)
print(f"Windows shape: {windows.x.shape}")
print(f"Percentage of missing values in input: {1 - windows.mask_x.mean():.2%}")
print(f"Target shape: {windows.y.shape}")
print(f"Percentage of missing values in target: {1 - windows.mask_y.mean():.2%}")

# %% Get windows for specific stations, parameters, and time range
print(f"Get observations as windows for specific stations, parameters, and time range")
print(f"Stations: {dataset.stations[:10]}")
sub_windows = dataset.get_windows(window_size=window_size,
                                  horizon_size=lead_times,
                                  stations=dataset.stations[:10],
                                  parameters=["wind_speed", "wind_direction"],
                                  first_date="2020-01-01",
                                  last_date="2022-01-01")
print(f"Windows shape: {sub_windows.x.shape}")
