import numpy as np
import pandas as pd
from sklearn import tree
from scipy.stats import binom

data = pd.read_csv("dogs.csv")
data = data.drop(["Action"], axis=1)

columns = data.columns.to_list()


def entropy_full(column):
    anzahl = len(data)
    info = data.loc[:, column].tolist()
    x = info.count(0) / anzahl
    y = 1 - x
    score = (x * np.log2(x)) + (y * np.log2(y))
    return score * -1


def entropy(x):
    y = 1 - x
    score = (x * np.log2(x)) + (y * np.log2(y))
    print(score * -1)


growling = [0.8113, 1]
heavy = [0.9183, 0.9709]
smelly = [0.9183, 0.9709]
big = [0.9183, 0.9709]

bb = [heavy, smelly, big, growling]

for b in bb:
    ent = sum(b) / len(b)


def binomial(p, k):
    for n in range(10):
        print(n, ' : ', (1 - binom.cdf(k, n, p))*100)


pmf = binom.pmf(0, 50, 0.08)
cdf = binom.cdf(4, 50, 0.08)
# print(cdf*100)

anzahl = []
werte = []

res = 0
t = 0
w = 0

while res < 0.99:
    t = t + 1
    for m in range(1, t):
        w = binom.pmf(0, m, 0.015)
        res = np.log(0.01)/np.log(w)
        anzahl.append(res)

print("Die Wahrscheinlichkeit", res, "wurde nach", len(anzahl),  "Iterationen erreicht")