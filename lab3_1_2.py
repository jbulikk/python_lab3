import pandas as pd
import matplotlib.pyplot as plt
import timeit
from functools import lru_cache
import json
import os
import pickle
import csv
import numpy as np

#zadanie 1.
df = pd.read_csv('test.csv')

print("\n\nDuplicates:")
duplicated = df[df.duplicated()]
if duplicated.empty:
    print("No duplicates found!")
else:
    print(duplicated)

#zadanie 2 - korelacja
korelacja = df['limit_bal'].corr(df['age'])
print(f"Korelacja: {korelacja}")

#zadanie 3 - dodawanie kolumny
df['suma_transakcji'] = df['bill_amt1'] + df['bill_amt2'] + df['bill_amt3'] + df['bill_amt4'] + df['bill_amt5'] + df['bill_amt6']
print(df)

#zadanie 4 - 10 najstarszych + tabela
oldest_cli = df.sort_values(by='age', ascending=False).head(10)
tabela = oldest_cli[['limit_bal', 'age', 'suma_transakcji']]
print(tabela)

#zadanie 5 - histogram
x1 = df['limit_bal']
x2 = df['age']

fig = plt.figure()

plt.subplot(1,3,1)
plt.hist(x1,label=['1'], bins = 30, alpha = 1)
plt.ylabel('num of people')
plt.xlabel('limit_bal')
plt.legend(borderpad = 0.5, fontsize=20)

plt.subplot(1,3,2)
plt.hist(x2,label=['2'], bins = 30, alpha = 1)
plt.ylabel('num of people')
plt.xlabel('age')
plt.legend(borderpad = 0.5, fontsize=20)

ax = plt.subplot(1, 3, 3, projection='3d')
x = df["age"]
y = df["limit_bal"]
bins = [10, 10]
hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, x.max()], [0, y.max()]])
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0
dx = x.max()/bins[0]
dy = y.max()/bins[1]
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
ax.set_xlabel('Age')
ax.set_ylabel('Balance limit')
ax.set_zlabel('Number of people')

# plt.show()

#DEKORATORY
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

class Tree:
    def __init__(self, root):
        self.root = root

    def _find_min_value_recursive(self, node):
        if not node.children:
            return node.value
        else:
            values = [self._find_min_value_recursive(child) for child in node.children]
            values.append(node.value)
            return min(values)

    @property
    def min_value(self):
        return self._find_min_value_recursive(self.root)


root_node = TreeNode(10)
root_node.children = [TreeNode(5), TreeNode(15)]
root_node.children[0].children = [TreeNode(3), TreeNode(8)]
root_node.children[1].children = [TreeNode(12), TreeNode(20)]

tree = Tree(root_node)
print(tree.min_value)

def fibonacci1(n):
    if n <= 1:
        return n
    else:
        return fibonacci1(n - 1) + fibonacci1(n - 2)

@lru_cache(maxsize=None)
def fibonacci2(n):
    if n <= 1:
        return n
    else:
        return fibonacci2(n - 1) + fibonacci2(n - 2)

# zwykly pomiar czasu
time_recursive = timeit.timeit(lambda: fibonacci1(35), number=1)
print(f"pomiar rekurencji: {time_recursive}")

# pomiar czasu z @lru_cache
time_cached = timeit.timeit(lambda: fibonacci2(35), number=1)
print(f"pomiar z @lru_cache: {time_cached}")

def cache_result(func):
    def wrapper(*args, **kwargs):
        cache_file = kwargs.get('cache_file')

        if cache_file and os.path.exists(cache_file):
            with open(cache_file, 'rb') as file:
                result = pickle.load(file)
            print(f'Wczytano wynik z pliku {cache_file}')
        else:
            result = func(*args, **kwargs)
            if cache_file:
                with open(cache_file, 'wb') as file:
                    pickle.dump(result, file)
                print(f'Zapisano wynik do pliku {cache_file}')
        return result

    return wrapper

@cache_result
def sumowanie(x, y, cache_file=None):
    return x + y

result1 = sumowanie(10, 3, cache_file='cache_result1.pkl')
print(result1)

