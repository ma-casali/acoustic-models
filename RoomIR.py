import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy import signal
import scipy

import numpy as np
from scipy import signal

def check_snr(ir_sq, fs):
    # 1. Find the peak energy
    peak_val = np.max(ir_sq)
    
    # 2. Estimate noise energy from the last 100ms of the IR
    # (Assuming your IR is long enough, e.g., 1.0s or more)
    noise_window = int(0.1 * fs)
    if len(ir_sq) < noise_window:
        return 0 # Signal too short
        
    noise_floor = np.mean(ir_sq[-noise_window:])
    
    # 3. Calculate SNR in dB
    snr_db = 10 * np.log10(peak_val / (noise_floor + 1e-12))
    
    return snr_db

def lundeby_algorithm(schroeder_db, fs):
    """
    Finds the intersection point between the decay and noise floor.
    ir_sq: Squared impulse response (energy)
    fs: Sampling frequency
    """
    # 1. Estimate initial noise floor from the last half of the signal
    n_samples = len(schroeder_db)
    noise_db = np.mean(schroeder_db[int(0.5 * n_samples):])
    noise_std = np.std(schroeder_db[int(0.5 * n_samples):])
    noise_db += 3 * noise_std
    
    # 3. Find the peak and the first -10dB point to get an initial slope
    peak_idx = np.argmax(schroeder_db[:1000]) # only look at first portion
    idx_10dB = np.where(schroeder_db <= (schroeder_db[peak_idx] - 1))[0][0]
    
    # Estimate initial slope (dB per sample)
    t_init = np.arange(peak_idx, idx_10dB)
    slope, intercept = np.polyfit(t_init, schroeder_db[t_init], 1)
    
    # get the initial estimate of the noise floor in dB

    
    for _ in range(5):  # Usually converges in 2-3 iterations
        # Intersection point (sample index)
        intersection = int((noise_db - intercept) / slope)
        
        # Safety: don't let intersection exceed signal length
        intersection = min(intersection, n_samples - 1)
        
        # Re-estimate noise floor using samples after the intersection point
        noise_start = int(intersection + (0.1 * fs)) # 100ms after intersection
        if noise_start < n_samples:
            noise_db = np.mean(schroeder_db[noise_start:]) + 3 * np.std(schroeder_db[noise_start:])
            
        idx_start = np.where(schroeder_db <= (schroeder_db[peak_idx] - 1))[0][0]
        idx_end = int(intersection - (intersection - idx_start) * 0.2) # use top 80% of decay

        if idx_end <= idx_start + 10: 
            break
        
        t_range = np.arange(idx_start, idx_end)
        slope, intercept = np.polyfit(t_range, schroeder_db[t_range], 1)

    return intersection, noise_db

def calculate_t60(ir, fs, band_vals, plot=True):

    # filter the IR to the desired frequency band
    sos = signal.butter(4, [band_vals[0]/(fs/2), band_vals[1]/(fs/2)], btype='bandpass', output='sos')
    ir_filtered = signal.sosfiltfilt(sos, ir)
    ir_filtered /= np.max(np.abs(ir_filtered))  # Normalize

    # 1. Square the IR to get energy
    ir_energy = ir_filtered**2

    snr = check_snr(ir_energy, fs)
    if snr < 15:
        print(f"Warning: Low SNR ({snr:.2f} dB) in band {band_vals[0]}-{band_vals[1]}. T60 non-estimable.")
        return np.nan, snr

    # 2. Schroeder Backwards Integration
    # We flip, cumulatively sum, and flip back
    schroeder_envelope = np.cumsum(ir_energy[::-1])[::-1]
    
    # 3. Convert to dB scale
    # Normalize to 0dB at the start of the decay
    schroeder_db = 20 * np.log10(schroeder_envelope / np.max(schroeder_envelope) + 1e-10)
    intersection_idx, noise_db = lundeby_algorithm(schroeder_db, fs)

    if schroeder_db[intersection_idx] > -5:
        print(f"Warning: Decay does not reach -5dB before noise floor in band {band_vals[0]}-{band_vals[1]}. T60 estimate may be inaccurate.")
        idx_begin_dB = np.where(schroeder_db <= 0)[0][0]
        idx_end_dB = np.where(schroeder_db <= -2)[0][0]
    else:
        idx_begin_dB = np.where(schroeder_db <= -2)[0][0]
        idx_end_dB = np.where(schroeder_db <= np.maximum(schroeder_db[intersection_idx], -35))[0][0]
    
    t_vals = np.arange(len(ir)) / fs
    
    # Linear regression on the -5 to -35 section
    slope, intercept = np.polyfit(t_vals[idx_begin_dB:idx_end_dB], schroeder_db[idx_begin_dB:idx_end_dB], 1)
    
    # T60 is the time it takes to drop 60dB
    # Since slope is dB/sec, T60 = -60 / slope
    t60 = -60 / slope
        
    return t60, snr

# JBL Speaker Frequency Response Data
jbl_freq =   [24.56864110136327, 27.601606062312495, 29.46084815750984, 30.63540308235808, 32.50245330019395, 34.05608382585834, 37.33923634156771, 42.019904310478324, 44.239120203596926, 45.73496062883436, 51.19245656759748, 59.5819496474486, 65.65906092810029, 73.3672566557296, 96.21876403294199, 108.30069031008134, 129.55640986408153, 143.78289999439374, 158.8167669500562, 172.657331225298, 183.5655848378578, 190.2079252080974, 199.01852173297328, 212.53114613438674, 229.7000527426764, 243.03409234572368, 280.1667446250832, 303.9139608193118, 317.21362602869874, 335.1854634109452, 374.547088932783, 405.7458548763522, 433.18567059347663, 480.72326605146884, 506.63663746767617, 533.7959919730234, 605.083257856504, 651.7496793449253, 715.411849454059, 743.8173360023808, 770.4662735363049, 796.0176842899701, 881.26667176362, 947.5359874705105, 970.755855706364, 1003.3275769402824, 1044.4758563549015, 1089.97808412575, 1153.5770459703351, 1287.1433177452163, 1403.672091712467, 1468.9681391493705, 1577.2506157282469, 1748.3598507207978, 2080.6947030349634, 2243.771507086488, 2371.7866309675105, 2463.7140718278406, 2637.4430838804024, 2816.6041948834604, 3253.58101957263, 3601.9080223149103, 3985.523879314004, 4351.807372983643, 5208.207334644194, 5779.390635175829, 6632.978419266601, 8232.001905538296, 9688.7791455147, 10715.956255924257, 11780.051387077376, 14452.828064707506, 16448.951869040742, 20067.941052462513]
jbl_spl_dB = [47.128849083161406, 55.20903148680176, 55.83028363187629, 61.634566068295314, 54.76261496635367, 63.29181690222316, 67.70131885930239, 72.62187879605885, 75.29387038929488, 76.66912828027688, 81.2890403563234, 85.75711007847599, 86.92325935638124, 87.20872298170178, 86.99354067446927, 86.9991805333282, 87.16881013439254, 87.45514143030678, 87.22651022887223, 86.80872683801555, 87.1006979927887, 86.39398029423674, 86.90894279158553, 86.76490947303472, 86.31979445847713, 85.51155930046468, 86.52282937739815, 86.27380791701214, 86.78486589668935, 85.7605807608507, 86.42998862387444, 85.96882170333379, 86.0269556331103, 85.10852630970055, 85.56969323024121, 84.89898386132697, 86.25298382276382, 86.10504598654147, 85.26600852245339, 86.07944970402792, 84.79095887241385, 86.54061662456857, 86.88725102674354, 86.75146057883269, 86.10244297476044, 86.5866031660336, 85.76708829030329, 86.57185276594103, 86.68464994311938, 85.24691976939243, 86.58009563658099, 88.0165243044174, 87.81609239727744, 89.81910996278657, 88.13105682278311, 88.4169542834005, 87.18616354626613, 87.89591809189596, 87.53800397200317, 88.2347434587278, 87.25470952316681, 88.00307541021537, 87.35232246495575, 88.36229103599868, 88.50589051925265, 88.28984054142646, 87.41522858299751, 87.10547018105393, 88.76836087384069, 88.21435319977633, 87.29462237047606, 88.64254863775716, 90.56617434394462, 87.18920039334401]
jbl_spl_dB = [spl - max(jbl_spl_dB) for spl in jbl_spl_dB]

# SM-57 Frequency Response Data
sm57_freq = [36.09330978118842, 48.20906024773066, 68.49803811443297, 95.5827752556458, 129.22853300357212, 167.2102527321169, 212.436793477822, 288.5635764574754, 374.7662208255668, 436.70843863530786, 493.07723504604263, 568.886960648911, 716.096690507162, 876.7662205794703, 1104.177200525823, 1361.4459758326045, 1692.3488721342019, 2052.2553987011165, 2507.9644828701375, 3046.1446565777123, 3661.5520937526667, 4366.58336205323, 4944.992569595826, 5525.9183126341295, 6221.599505543331, 6829.870874038535, 7246.925790543502, 8154.498935480979, 8783.57061374405, 9297.176473270663, 10036.829436274671, 10538.131735503164, 11027.592211668925, 11803.314803650423, 12994.815147017947, 13886.914676426904, 14450.114468044812, 14824.940985082467, 16068.39387688515, 16662.614197849336]
sm57_spl_dB = [-12.118704080729398, -9.87695848455342, -7.203682393555811, -4.411348145525361, -2.4125874125874116, -0.9444985394352479, -0.2611312737895002, -0.28370363813401767, -0.8475701513676182, -1.0321324245374868, -0.8528812959192695, -0.49969018323448644, -0.052226254757899326, 0.39966362751173, 0.5731610161989913, 0.7201026821279992, 0.8024254226785885, 1.2078427901212727, 1.6553067185978598, 2.5803310613437205, 3.885102239532621, 5.2673276090997625, 6.36009560060193, 6.869965477560417, 6.7841019739753925, 5.298751881030363, 3.464636629193592, 3.020713463751438, 4.234309993803667, 4.818535894485262, 3.6784101973975396, 2.266088342037712, 1.754448083562007, 2.6546870850668327, 1.6584048862529883, -0.11374701248118946, -1.9518456227316978, -3.256174205541294, -5.8179162609542345, -7.744091351686288]
sm57_spl_dB = [spl - max(sm57_spl_dB) for spl in sm57_spl_dB]

x_res_func = CubicSpline(jbl_freq, jbl_spl_dB, bc_type='not-a-knot')
y_res_func = CubicSpline(sm57_freq, sm57_spl_dB, bc_type='not-a-knot')

# --- 2. Signal Parameters ---
fs = 44100
chirp_len = 5.0
n_samples = int(fs * chirp_len)
t = np.linspace(0, chirp_len, n_samples, endpoint=False)
single_chirp = 0.9 * signal.chirp(t, f0=20, f1=20000, t1=chirp_len, method='logarithmic')

# save chirp signal
scipy.io.wavfile.write('Audio/SingleChirp.wav', fs, (single_chirp * 32767).astype(np.int16))

# --- 3. Process Recording ---
fs_rec, room_raw_1= scipy.io.wavfile.read('Audio/RoomRecording_NT1_noPanels.wav')
_, room_raw_2 = scipy.io.wavfile.read('Audio/RoomRecording_SM57_noPanels.wav')
if fs_rec != fs:
    raise ValueError("Sample rate mismatch!")

# Normalize and convert to float
room_raw_1 = room_raw_1.astype(float)
room_raw_2 = room_raw_2.astype(float)
room_raw = room_raw_1 + room_raw_2 # add in-phase cardioid mics together to get a more omnidirectional response
room_raw /= np.max(np.abs(room_raw))  # Normalize to -1.0 to 1.0 range
print(len(room_raw) / fs, "seconds of recording loaded.")

# Find the start of the sequence using cross-correlation
# We use the first chirp of the sequence for alignment
corr = signal.correlate(room_raw, single_chirp, mode='valid')
delay_arr = np.linspace(0, len(corr), len(corr))
delay_idx = delay_arr[np.argmax(np.abs(corr[:len(single_chirp)]))].astype(int)
print(delay_idx / fs, "seconds delay detected.")

# Extract and Average "Middle" Chirps (Chirps 2 through 6)
# This avoids the startup transient and ending decay of the 7-repeat block
captured_periods = []
for i in range(1, 33): 
    start = delay_idx + (i * n_samples)
    end = start + n_samples
    captured_periods.append(room_raw[start:end])

avg_recording = np.mean(captured_periods, axis=0)

# --- 4. Complex Deconvolution ---
# Perform FFTs on the raw, UNWINDOWED signals
ref_fft = np.fft.rfft(single_chirp)
rec_fft = np.fft.rfft(avg_recording)

# Deconvolve by complex division (Transfer Function)
# We add a small epsilon to avoid division by zero
epsilon = 1e-10
tf_complex = rec_fft / (ref_fft + epsilon)

# --- 5. Clean the Impulse Response (IR) ---
# Convert back to time domain
ir_time = np.real(np.fft.irfft(tf_complex))

# WINDOW THE IR: This is crucial. We only want the room's energy.
# A Tukey window with a small alpha creates a smooth fade-in/out
ir_len = len(ir_time)
window = signal.windows.tukey(ir_len, alpha=0.05)
ir_time_clean = ir_time * window

# --- 6. Final Frequency Analysis & Correction ---
freqs = np.fft.rfftfreq(ir_len, 1/fs)
room_mag = np.abs(np.fft.rfft(ir_time_clean))

# Normalize
room_mag /= np.max(room_mag)

mask = (freqs >= 20) & (freqs <= 20000)

# Get the IR after correction
ir_time = np.real(np.fft.irfft(tf_complex, n=ir_len))
ir_time /= (np.max(np.abs(ir_time)) + epsilon)
t = np.arange(len(ir_time)) / fs

# fig, ax = plt.subplots(2, 1, figsize=(10, 6))
# ax[0].semilogx(freqs[mask], 20 * np.log10(room_mag[mask] + epsilon), label='Corrected Room Response')
# ax[0].set_title("Room Frequency Response (Deconvolved & Corrected)")
# ax[0].set_xlabel("Frequency (Hz)")
# ax[0].set_ylabel("Magnitude (dB)")
# ax[0].grid(True, which="both", ls="-", alpha=0.5)
# ax[0].legend()

# ax[1].plot(t, 20*np.log10(np.abs(ir_time)))
# # ax[1].plot(decay_line_time + (peak_idx / fs), 20*np.log10(decay_line_level), 'r--', label='T60 Decay Line')
# ax[1].set_title("Corrected Room Impulse Response")
# ax[1].set_xlabel("Time (s)")
# ax[1].set_ylabel("Amplitude")
# ax[1].grid(True)

# Usage:
# t60_value = calculate_t60(ir_time_clean, fs)
third_octave_bands = [(20*2**(i/3) , 20*2**((i+1)/3)) for i in range(0, 30)]
third_octave_band_centers = [20*2**((i+0.5)/3) for i in range(0, 30)]
t60_band_values = []
snr_values = []
schroeder_db_arr = np.zeros((len(third_octave_bands), len(ir_time)))
for i, band in enumerate(third_octave_bands):
    t60_value, snr = calculate_t60(ir_time, fs, band_vals=band, plot=False)
    t60_band_values.append(t60_value)
    snr_values.append(snr)
full_t60, snr = calculate_t60(ir_time, fs, band_vals=(20, 20000), plot=False)

f_s = 2000 * np.sqrt(0.6 / 27.18)  # Sabine formula rearranged for f_s
T_m = 0.25 * (27.18 / 100 ) ** (1/3) 
print(f"Estimated diffusion range: [f_s -> 4f_s] = [{f_s:.2f} --> {4*f_s:.2f}] Hz")

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(third_octave_band_centers, t60_band_values, marker='o', label='T60 per 1/3 Octave Band')
ax.axhline(full_t60, color='r', linestyle='--', label=f'Full Band T60: {full_t60:.2f} s')
ax.axhline(T_m, color='g', linestyle='--')
ax.set_title("T60 vs Frequency (without Acoustic Panels)")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("T60 (seconds)")
ax.grid(True, which="both", ls="-", alpha=0.5)
ax.set_ylim(0.01, 2)
ax.legend(loc='upper right')

ax2 = ax.twinx()
color = 'tab:orange'
ax2.semilogx(third_octave_band_centers, snr_values, marker='x', label='SNR per 1/3 Octave Band', color = color)
ax2.set_ylabel("SNR (dB)", color=color)
ax2.tick_params(axis='y', labelcolor=color)
plt.show()