# Imports
# --------------------

import tensorflow as tf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt



def plot_series(time, series, title, format="-", start=0, end=None):
    """
    Visualizes time series data
    Args:
      time (array of int) - contains the time steps
      series (array of int) - contains the measurements for each time step
      format - line style when plotting the graph
      label - tag for the line
      start - first time step to plot
      end - last time step to plot
    """    
    plt.figure(figsize=(10, 6))     # Setup dimensions of the graph figure    
    if type(series) is tuple:
      for series_num in series:        
        plt.plot(time[start:end], series_num[start:end], format)    # Plot the time series data
    else:
      plt.plot(time[start:end], series[start:end], format)          # Plot the time series data
    plt.xlabel("Time")          # Label the x-axis    
    plt.ylabel("Value")         # Label the y-axis    
    plt.title(title)
    plt.grid(True)              # Overlay a grid on the graph    
    plt.show()                  # Draw the graph on screen
# =============================================================================================
def trend(time, slope=0):
    """
    Generates synthetic data that follows a straight line given a slope value.
    Args:
      time (array of int) - contains the time steps
      slope (float) - determines the direction and steepness of the line
    Returns:
      series (array of float) - measurements that follow a straight line
    """    
    series = slope * time       # Compute the linear series given the slope
    return series
# =============================================================================================
def seasonal_pattern(season_time):
    """
    Just an arbitrary pattern, you can change it if you wish    
    Args:
      season_time (array of float) - contains the measurements per time step
    Returns:
      data_pattern (array of float) -  contains revised measurement values according 
                                  to the defined pattern
    """
    # Generate the values using an arbitrary pattern
    data_pattern = np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))    
    return data_pattern
# =============================================================================================
def seasonality(time, period, amplitude=1, phase=0):
    """
    Repeats the same pattern at each period
    Args:
      time (array of int) - contains the time steps
      period (int) - number of time steps before the pattern repeats
      amplitude (int) - peak measured value in a period
      phase (int) - number of time steps to shift the measured values
    Returns:
      data_pattern (array of float) - seasonal data scaled by the defined amplitude
    """        
    season_time = ((time + phase) % period) / period            # Define the measured values per period    
    data_pattern = amplitude * seasonal_pattern(season_time)    # Generates the seasonal data scaled by the defined amplitude
    return data_pattern
# =============================================================================================
def noise(time, noise_level=1, seed=None):
    """
    Generates a normally distributed noisy signal
    Args:
      time (array of int) - contains the time steps
      noise_level (float) - scaling factor for the generated signal
      seed (int) - number generator seed for repeatability
    Returns:
      noise (array of float) - the noisy signal
    """    
    rnd = np.random.RandomState(seed)       # Initialize the random number generator
    # Generate a random number for each time step and scale by the noise level
    noise = rnd.randn(len(time)) * noise_level    
    return noise
# =============================================================================================

def generate_synthetic_timeseries(baseline=10, amplitude=40, slope=0.05, noise_level=5):
    time = np.arange(4 * 365 + 1, dtype="float32")
    # Create the series
    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
    # Update with noise
    series += noise(time, noise_level, seed=42)
    return series, time

# =============================================================================================
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    """Generates dataset windows

    Args:
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to include in the feature
      batch_size (int) - the batch size
      shuffle_buffer(int) - buffer size to use for the shuffle method

    Returns:
      dataset (TF Dataset) - TF Dataset containing time windows
    """
  
    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)
    
    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    
    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # Create tuples with features and labels 
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    # Shuffle the windows
    dataset = dataset.shuffle(shuffle_buffer)
    
    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)
    
    return dataset

## ---------------------------------------------------------------------------------

def model_forecast(model, series, window_size, batch_size):
    """Uses an input model to generate predictions on data windows

    Args:
      model (TF Keras Model) - model that accepts data windows
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to include in the window
      batch_size (int) - the batch size

    Returns:
      forecast (numpy array) - array containing predictions
    """

    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda w: w.batch(window_size))
    
    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)
    
    # Get predictions on the entire dataset
    forecast = model.predict(dataset)
    
    return forecast