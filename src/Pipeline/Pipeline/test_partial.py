from functools import partial

def add_one(n):
    return n + 1

def mutliple_by(n, x):
    return n * x

multiple_by = partial(mutliple_by, x=3)

num = 3
print(add_one(num))
print(multiple_by(num))
