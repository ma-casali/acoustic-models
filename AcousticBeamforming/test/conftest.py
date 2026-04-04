import pytest
import scipy
import numpy as np
from src.BeamformingModel import BeamformingArray
from src.BeamformingModel import BeamformingModel

DATA_SCENARIOS = {
    "400Hz_Tone_HiSNR": {
        "frequency": 400,
        "snr_db": 500,
        "dropout": False,
        "near_field": False,
    },
    "400Hz_Tone_Dropout": {
        "frequency": 400,
        "snr_db": 500,
        "dropout": True,
        "near_field": False,
    },
    "400Hz_Tone_LoSNR": {
        "frequency": 400,
        "snr_db": 0,
        "dropout": False,
        "near_field": False,
    },
    "400Hz_Tone_NearField": {
        "frequency": 400,
        "snr_db": 500,
        "dropout": False,
        "near_field": True,
    }
}

@pytest.fixture
def sample_data():
    """Fixture to provide sample signals to elements for testing."""
   
    def _generate(array: BeamformingArray, arrival_az=15, arrival_de=15, frequency=400, snr_db=500):

        # For a noise source at a random position in the far-field, 
        # generate the signals received by each element in the array.
        bf_model = BeamformingModel(array)
        arrival_de = np.radians(np.array([[arrival_de]]))
        arrival_az = np.radians(np.array([[arrival_az]]))

        manifold_vector_main = bf_model.compute_manifold_vector(arrival_az, arrival_de, frequency)
        manifold_vector_main = manifold_vector_main.flatten()
        simple_tone = np.exp(1j * 2 * np.pi * frequency * np.arange(0, 1, 1/44100)) # shape (num_samples,)
        tone_array = np.real(manifold_vector_main[np.newaxis, :] * simple_tone[:, np.newaxis]) # shape (num_elements, num_samples)

        # Add random noise to the tone array for a given SNR
        # This will add noise randomly in time and space
        signal_power = np.mean(tone_array**2)
        noise_power = signal_power / (10**(snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), tone_array.shape)
        tone_array += noise

        return tone_array
    
    return _generate
