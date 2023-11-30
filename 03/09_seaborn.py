import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

data = pd.DataFrame({'X': [1, 2, 3, 4, 5],
                     'Y': [2, 4, 1, 3, 5]})

sns.scatterplot(x='X', y='Y', data=data)
plt.show()
