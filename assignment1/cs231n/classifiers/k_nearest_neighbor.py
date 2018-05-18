import numpy as np
from collections import Counter

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        #####################################################################
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        # dists[i][j] = (((X[i]-self.X_train[j])**2).sum())**(0.5)
        # dists[i][j] = np.sqrt(np.sum(np.square(X[i]-self.X_train[j])))
        dists[i][j] = np.sqrt(np.sum((X[i]-self.X_train[j])**2))
        
        '''(x-y)**2 is seperated
        sum_square = np.sum(X[i]**2, axis=-1) + np.sum(self.X_train[j]**2, axis=-1) - 2 * np.sum(X[i] * self.X_train[j], axis=-1)
        dists[i][j] = np.sqrt(sum_square)
        '''
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #######################################################################
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      # dists[i] = np.sqrt(np.sum(np.square(X[i] - self.X_train), axis=1))
      dists[i] = np.sqrt(np.sum((X[i] - self.X_train)**2, axis=1))
      
      '''(x-y)**2 is seperated
      sum_square = np.sum(X[i]**2, axis=-1) + np.sum(self.X_train**2, axis=-1) - 2 * np.sum(X[i] * self.X_train, axis=-1)
      dists[i] = np.sqrt(sum_square)
      '''
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    '''MemoryError!!!
    X = X.reshape(num_test, 1, -1)
    substract = X - self.X_train
    square = substract**2
    su = np.sum(square, axis=-1)
    dists = np.sqrt(su)
    # dists = np.sqrt(np.sum((X.reshape(num_test, 1, -1) - self.X_train)**2, axis=-1))
    '''
    batch_size = 10
    for i in range(int(num_test/batch_size)):
        start = i * batch_size
        end = (i + 1) * batch_size
        dists[start:end] = np.sqrt(np.sum((X[start:end].reshape(batch_size, 1, -1) - self.X_train)**2, axis=-1))
    
    '''(x-y)**2 is seperated
    X = X.reshape(num_test, 1, -1)
    # substract = X - self.X_train  # 500*5000*32*32*3*4(int32)=30,720,000,000
    sum_square = np.sum(X**2, axis=-1) + np.sum(self.X_train**2, axis=-1) - 2 * np.sum(X * self.X_train, axis=-1)

    dists = np.sqrt(sum_square)
    '''
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      mask = np.argsort(dists[i])[:k]  # np.ndarray
      labels = self.y_train[mask]  # np.ndarray
      closest_y = labels.tolist()  # list
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      y_pred[i] = max(set(closest_y), key=closest_y.count)  # np.ndarray
      # y_pred[i] = Counter(closest_y).most_common(1)[0][0]
      
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

if __name__ == '__main__':
    eval = KNearestNeighbor()
    train_x = np.array([[1,1,1,1],[1,1,1,1],[0,0,0,0],[0,0,0,0]])
    train_y = np.array([1,1,0,0])
    
    test_x = np.array([[1,1,1,1],[0,0,1,0]])
    test_y = np.array([1,0])
    
    eval.train(train_x,train_y)
    dists = eval.compute_distances_one_loop(test_x)
    print(dists)
    
    # labels = eval.predict_labels(dists)
    # print(labels)
    
    