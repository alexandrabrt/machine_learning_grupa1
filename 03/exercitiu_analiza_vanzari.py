from faker import Faker
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

fake = Faker()

data = {'Data': ['2023-11-15', '2023-11-16'],
        'Produs': [1, 2], 'Cantitate': [2, 4], 'Pret_unitar': [2.0, 4.5]}
# for _ in range(100):
#     data['Data'].append(fake.date_between(start_date='-30d', end_date='today'))
#     data['Produs'].append(random.choice(['Produs_A', 'Produs_B', 'Produs_C']))
#     data['Cantitate'].append(random.randint(10, 50))
#     data['Pret_unitar'].append(round(random.uniform(20, 100), 2))
#
# df = pd.DataFrame(data)
# df.to_csv('vanzari.csv', index=False)
values = pd.read_csv('vanzari.csv')
df = pd.DataFrame(values)

df['Venituri'] = df['Cantitate'] * df['Pret_unitar']

total_vanzari_zilnice = df.groupby('Data')['Venituri'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(x='Data', y='Venituri', data=total_vanzari_zilnice)
plt.title('Evolutie vanzari zilnice')
plt.xlabel('Data')
plt.ylabel('Venituri')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Pret_unitar'], bins=20, kde=True, color='skyblue')
plt.title('Distributie preturi produse')
plt.xlabel('Pret unitar')
plt.ylabel('Numar produse')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Pret_unitar', y='Cantitate', data=df, hue='Produs', palette='viridis')
plt.title('Relatia dintre pret si Cantitate vanduta')
plt.xlabel('Pret unitar')
plt.ylabel('Cantitate vandura')
plt.legend(title='Produs')
plt.show()


df = pd.DataFrame(data)
df['Data'] = pd.to_datetime(df['Data'])

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Matrice de corelatie')
plt.show()
