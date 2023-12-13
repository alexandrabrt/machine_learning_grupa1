import matplotlib.pyplot as plt
from scipy import stats

# x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]
# y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
x = [89, 43, 36, 36, 95, 10, 66, 34, 38, 20, 26, 29, 48, 64, 6, 5, 36, 66, 72, 40]
y = [21, 46, 3, 35, 67, 95, 53, 72, 58, 10, 26, 34, 90, 33, 38, 20, 56, 2, 47, 15]

slope, intercept, r, p, str_err = stats.linregress(x, y)


def myfunc(x):
    return slope * x + intercept


my_model = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, my_model)
plt.show()
