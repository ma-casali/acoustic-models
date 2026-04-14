import numpy as np
import scipy

# y = []
# for n in range(13):
#     for m in range(1, 3):
#         y.append(scipy.special.jn_zeros(n, m)[-1])

# x = np.sort(y)[:20]

# for i in x:
#     print(np.round(i/x[0], 5))


# get upper frequency for N sensors, lower frequency, and aperture size P

N = 16 # number of sensors
f_L = 300 # lower frequency
P = 5 # aperture size in half wavelengths for the ith frequency (basically the grouping of sensors)

f_U = f_L * 10 ** ((N - (P + 1)) * np.log10(P/(P - 1)))

print(f"Upper frequency: {f_U:.2f} Hz")