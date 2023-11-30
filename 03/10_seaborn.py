import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
print(iris)

plt.figure(figsize=(12, 6))
sns.barplot(x='species', y='sepal_length', data=iris)
plt.title('Medie sepal length pe specie')
plt.xlabel('Specie')
plt.ylabel('Medie Sepal Length')
plt.show()

plt.figure(figsize=(14, 8))
sns.violinplot(x='species', y='sepal_length', data=iris, palette='muted')
plt.title('Distributia Sepal Lenght pe Specie')
plt.xlabel('Specie')
plt.ylabel('Sepal length')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris, palette='Dark2')
plt.title('Relatie intre Sepal length si sepal width')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

sns.pairplot(iris, hue='species', palette='husl')
plt.suptitle('Pair plot pentru caracterisitic iris', y=1.02)
plt.show()
