import pandas as pd
import matplotlib.pyplot as plt

data = {'Nota_matematica': [75, 80, 65, 90, 85],
        "Nota_engleza": [85, 70, 95, 80, 75]}

df = pd.DataFrame(data)

df.plot.scatter(x='Nota_matematica', y='Nota_engleza', color='green', marker='o')


plt.xlabel('Nota matematica')
plt.ylabel('Nota engleza')
plt.title("Relatia intre notele la matematica si engleza")

plt.show()
