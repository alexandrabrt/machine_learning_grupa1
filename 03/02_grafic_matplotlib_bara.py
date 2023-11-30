import pandas as pd
import matplotlib.pyplot as plt

data = {"An": [2010, 2011, 2012, 2013, 2014],
        "Vanzari": [500, 600, 750, 800, 900]}

df = pd.DataFrame(data)

df.plot(x='An', y='Vanzari', kind='bar', color='g', legend=False)

plt.xlabel("An")
plt.ylabel("Vanzari")
plt.title("Grafic bara - vanzari in functie de an")
plt.grid(axis='y')

plt.show()
