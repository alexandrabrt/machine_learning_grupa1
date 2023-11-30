import pandas as pd
import matplotlib.pyplot as plt

data = {"An": [2010, 2011, 2012, 2013, 2014],
        "Vanzari": [500, 600, 750, 800, 900]}

df = pd.DataFrame(data)

df.plot(x='An', y='Vanzari', marker='o', linestyle='--', color='b', label='Vanzari')
df.plot.bar(x='An', y='Vanzari', color='g', alpha=0.7, label='Vanzari (bara)')

plt.legend()
plt.show()
