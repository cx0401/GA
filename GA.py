import math
from typing import List
import random

import numpy


class SearchStrategy(object):
    def __init__(self):
        self.mutateFactor = 0.1
        self.learnRate = 0.9
        self.popNum = 100
        self.variableNum = 2
        self.pop = []
        self.lower = -128
        self.upper = 128
        self.bitNum = 8
        self.select = 0.8
        self.pop = [[random.randint(self.lower, self.upper) for i in range(self.popNum)] for j in range(self.variableNum)]

    def generate_tasks(self):
        # mutate
        for i in range(self.popNum):
            for k in range(self.variableNum):
                M = 0
                m = 1
                for j in range(self.bitNum):
                    if random.random() > self.learnRate:
                        M += m
                    m <<= 1
                self.pop[k].append(self.pop[k][i] ^ M)
        return

    def compute_score(self) -> List[float]:
        score = []
        for i in range(len(self.pop[0])):
            score.append(float(self.pop[0][i] - 1) * (self.pop[0][i] - 1) +
                         100 * math.pow((self.pop[1][i] - pow(self.pop[0][i],2)),2))
        return score

    def handle_rewards(self, score: List[float]) -> None:
        index = numpy.array(score).argsort()[0:self.popNum]
        self.pop = [[e for i, e in enumerate(self.pop[t]) if i in index] for t in range(self.variableNum)]
        return None


a = SearchStrategy()
epochs = 1000
for i in range(epochs):
    a.generate_tasks()
    a.handle_rewards(a.compute_score())
    print(a.pop,a.compute_score(),"\n")
