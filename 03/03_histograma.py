import pandas as pd
import matplotlib.pyplot as plt

data = {"Varsta": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]}

df = pd.DataFrame(data)

df['Varsta'].hist(bins=5, color='skyblue', edgecolor='black')

plt.xlabel('Varsta')
plt.ylabel('Frecventa')
plt.title("Histograma - Distributia Varstei")

plt.show()
