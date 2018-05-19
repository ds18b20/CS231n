import numpy as np
import functools
import time

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsed_time * 1000)))
    return wrapper

@timeit
def mul(a, b):
    row = a.shape[0]
    return np.sum(a.reshape(row, 1, -1) * b, axis=-1)
    
@timeit
def mul_dot(a, b):
    return np.dot(a, b.T)
    
if __name__ == "__main__":
    a = np.random.randn(50, 32*32*3)
    b = np.random.randn(500, 32*32*3)
    
    mul(a, b)
    mul_dot(a, b)
