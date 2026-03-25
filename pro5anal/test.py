import numpy as np
data = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
print(data)

import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42) 
data = np.random.normal(loc=0.0, scale=1.0, size=1000)


plt.figure(figsize=(10, 6))
plt.hist(data, bins=20, alpha=0.7, color='skyblue', edgecolor='black')


plt.title('good')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.5)


plt.show()

