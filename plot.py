import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV file
data1 = pd.read_csv('build/bin/H1.csv')
data2 = pd.read_csv('build/bin/fid.csv')

# Plot the data
plt.plot(data1['x'], data1['y'], label='H1')
plt.plot(data2['x'], data2['y'], label='fid')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
# plt.title('Plot of y = x^2')
plt.legend()
plt.grid(True)
plt.show()
