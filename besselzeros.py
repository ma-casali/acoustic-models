import numpy as np
import scipy

y = []
for n in range(13):
    for m in range(1, 3):
        y.append(scipy.special.jn_zeros(n, m)[-1])

x = np.sort(y)[:20]

for i in x:
    print(np.round(i/x[0], 5))
