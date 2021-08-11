import copy
import math
import time
from typing import List
import random

import numpy as np

def F(x, y):
    return 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(
        -x ** 2 - y ** 2) - 1 / 3 ** np.exp(-(x + 1) ** 2 - y ** 2)

class SearchStrategy(object):
    def __init__(self):
        self.mutateFactor = 0.1
        self.crossFactor = 0.8
        self.tournament = 0.5
        self.popNum = 200
        self.variableNum = 2
        self.pop = []
        self.newPop = []
        self.fitness = []
        self.lower = -3
        self.upper = 3
        self.bitNum = 24
        self.bitRange = pow(2, self.bitNum) - 1
        self.select = 0.8
        self.pop = [[random.random() * (self.upper - self.lower) + self.lower
                     for i in range(self.variableNum)] for j in range(self.popNum)]

    def generate_tasks(self):
        def encode(x: float) -> int:
            return round((x - self.lower) / (self.upper - self.lower) * self.bitRange)

        def decode(x: int) -> float:
            return x / self.bitRange * (self.upper - self.lower) + self.lower

        def cross(x: float, y: float) -> float:
            par = encode(x)
            mom = encode(y)
            crossPoint = random.randint(1, self.bitNum - 1)
            move = pow(2, crossPoint) - 1
            par &= move
            move <<= (self.bitNum - crossPoint)
            mom &= move
            return decode(par + mom)

        def mutate(x: float) -> float:
            return decode(encode(x) ^ pow(2, random.randint(0, self.bitNum - 1)))

        # cross
        newPop = [[cross(self.pop[i][k], self.pop[random.randint(0, self.popNum - 1)][k])
                   if random.random() < self.crossFactor else self.pop[i][k]
                   for k in range(self.variableNum)] for i in range(self.popNum)]

        # mutate
        newPop = [[mutate(newPop[i][k])
                   if random.random() < self.mutateFactor else newPop[i][k]
                   for k in range(self.variableNum)]
                  for i in range(self.popNum)]
        return newPop

    def compute_score(self) -> List[float]:
        def general(x):
            return pow(x[0] - 1, 2) + pow(x[1] - 1, 2)

        def rosenbroke(x):
            return float(x[0] - 1) * (x[0] - 1) + 100 * math.pow((x[1] - pow(x[0], 2)), 2)

        def csdn(x):
            return F(x[0], x[1])

        score = []
        for i in self.pop:
            score.append(csdn(i))
        self.fitness = score
        return score

    def handle_rewards(self, score: List[float]) -> None:
        newPop = []

        # tournament algorithm
        index = np.argsort(score)
        for i in range(self.popNum):
            j = 0
            while random.random() > self.tournament and j < self.popNum:
                j += 1
            newPop.append(self.pop[index[j]])

        # # roulette algorithm
        # score = [pow(math.e, i) for i in score]
        # score = [i / sum(score) for i in score]
        # index = np.random.choice(np.arange(len(self.pop)), size=self.popNum, replace=True, p=score)
        # newPop = [self.pop[i] for i in index]

        self.pop = newPop
        return None


a = SearchStrategy()
epochs = 100
for i in range(epochs):
    random.seed(time.time())
    a.pop = a.generate_tasks()
    a.handle_rewards(a.compute_score())
    if i % 10 == 0:
        print(a.pop, a.compute_score(), "\n")
