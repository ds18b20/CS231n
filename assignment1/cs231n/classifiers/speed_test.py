import numpy as np
import functools
import time

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        ret = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsed_time * 1000)))
        return ret
    return wrapper

@timeit
def mul(a, b):
    row = a.shape[0]
    ret = np.sum(a.reshape(row, 1, -1) * b, axis=-1)
    return ret
    
@timeit
def mul_dot(a, b):
    ret = np.dot(a, b.T)
    return ret
    
if __name__ == "__main__":
    a = np.random.randn(50, 32*32*3)
    b = np.random.randn(500, 32*32*3)
    
    print('Return Array[0][0]', mul(a, b)[0][0])
    print('Return Array[0][0]', mul_dot(a, b)[0][0])
