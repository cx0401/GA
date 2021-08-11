import numpy as np


def compute_score(array):
    score = []
    for i in array:
        score.append(i[0]*i[0] +i[1]*i[1])
    return score

pop = [[1,1],[2,2],[0,0]]
score = compute_score(pop)
index = np.argsort(score)[:1]
pop = [pop[i] for i in index]
print(pop)