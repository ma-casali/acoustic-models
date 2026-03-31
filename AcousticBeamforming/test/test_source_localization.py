from src.BeamformingModel import BeamformingArray, BeamformingModel
from src.BeamformingModel import ElementDirectivity
import numpy as np
import pytest
import matplotlib.pyplot as plt

def test_source_localization(sample_data):

    # Create the Array
    d = 343/200/4 # element spacing of 1/4 wavelength at 2 kHz
    n_elements_per_side = 6
    y, z = np.meshgrid(np.linspace(-2*d, 2*d, n_elements_per_side), np.linspace(-2*d, 2*d, n_elements_per_side))
    Y = np.ravel(y)
    Z = np.ravel(z)
    bf_array = BeamformingArray(X=np.zeros(n_elements_per_side**2),
                                Y=Y,
                                Z=Z, 
                                element_directivity=ElementDirectivity.OMNI)
    
    arrival_az = 15
    arrival_de = 80
    sample_signals = sample_data(bf_array, arrival_az, arrival_de, frequency=400, snr_db=400)
    steer_az = np.radians(np.arange(-90, 90, 5))
    steer_de = np.radians(np.arange(0, 90, 5))
    AZ, DE = np.meshgrid(steer_az, steer_de)

    bf_model = BeamformingModel(bf_array)

    filtered_power = bf_model.apply_spatial_filter(sample_signals, AZ, DE, frequency= 400, nperseg=4096)
    
    # assert filtered_power.shape == (len(steer_az), len(steer_de), int(np.ceil(sample_signals.shape[0] / 4096))), f"Expected filtered power shape {(len(steer_az), len(steer_de), np.ceil(sample_signals.shape[0] // 4096))}, but got {filtered_power.shape}"
    assert not np.all(filtered_power.flatten() == 0), "Filtered power is all zeros, which is unexpected"
    assert np.unique(filtered_power.flatten()).shape[0] > 1, "Filtered power has only one unique value, which is unexpected"

    # The filtered signal should have a strong peak at the arrival direction of the source, which is (30, 15) in this case
    # We can check this by finding the direction that maximizes the power of the filtered signal
    max_power_index = np.unravel_index(np.argmax(filtered_power), filtered_power.shape)[:2]
    print(max_power_index)
    estimated_de = steer_de[max_power_index[0]]
    estimated_az = steer_az[max_power_index[1]]

    # assert np.isclose(np.degrees(estimated_az), 15, atol=1), f"Estimated azimuth {np.degrees(estimated_az)} is not close to true azimuth 15"
    # assert np.isclose(np.degrees(estimated_de), 15, atol=1), f"Estimated elevation {np.degrees(estimated_de)} is not close to true elevation 15"
    # Show the beampattern for visual inspection
    bf_model.plot_spatially_filtered_result(filtered_power, AZ, DE)
    plt.show()

# @pytest.mark.parametrize("size", [1000, 10000, 44100])
# def test_spatial_filter_memory(size):

#     d = 343/2e3/4 # element spacing of 1/4 wavelength at 2 kHz
#     y, z = np.meshgrid(np.linspace(-2*d, 2*d, 4), np.linspace(-2*d, 2*d, 4))
#     Y = np.ravel(y)
#     Z = np.ravel(z)
#     bf_array = BeamformingArray(X=np.zeros(16),
#                                 Y=Y,
#                                 Z=Z, 
#                                 element_directivity=ElementDirectivity.OMNI)
    
#     bf_model = BeamformingModel(bf_array)

#     # Use a smaller steering grid for the test to avoid hanging the test runner
#     steer_az = np.linspace(-90, 90, 100) 
#     steer_de = np.linspace(-90, 90, 100)
#     az, de = np.meshgrid(steer_az, steer_de)
#     fake_signals = np.random.randn(size, 16)
    
#     result = bf_model.apply_spatial_filter(fake_signals, az, de, 400)
#     assert result.shape == (100, 100, 1)