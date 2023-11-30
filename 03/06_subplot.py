import pandas as pd
import matplotlib.pyplot as plt

data = {'An': [2010, 2011, 2012, 2013, 2014],
        "Vanzari": [500, 600, 750, 800, 900]}


df = pd.DataFrame(data)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

df.plot(x='An', y='Vanzari', ax=axs[0, 0], marker='o', linestyle='-', color='b', label="Vanzari (linie)")
axs[0, 0].set_title("Grafic linie - vanzari")

df.plot.bar(x='An', y='Vanzari', ax=axs[0, 1], color='g', alpha=0.7, label='Vanzari (bara)')
axs[0, 1].set_title('Grafic bara - vanzari')

axs[1, 0].set_visible(False)
axs[1, 1].set_visible(False)

plt.tight_layout()
plt.show()
