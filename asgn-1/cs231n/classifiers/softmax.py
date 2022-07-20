from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def softmax(z):
        exp = np.exp(z)
        return exp / np.sum(exp)

    N = X.shape[0]

    for xi, yi in zip(X, y):
        xi = xi.reshape(-1,1) # (D,1)
        yi # scaler
        
        # forward
        z = W.T @ xi # (C,1)
        z -= np.max(z) # numerical stability
        a = softmax(z) # (C,1)
        l = -np.log(a[yi] + 10**-9).item() # scaler

        loss += l

        dl_dz = a # (C,1)
        dl_dz[yi] -= 1 # dl/dz_k = a_k-1 if k == yi else a_k
        dz_dw = xi # (D,1)
        dl_dw = dz_dw @ dl_dz.T

        dW += dl_dw

    # avg
    loss /= N
    dW /= N

    # reg
    loss += reg * np.sum(W**2)
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]

    def softmax(z):
        exp = np.exp(z) # (N,C)
        sum = np.sum(exp, axis=1).reshape(-1,1) # (N,1)
        return exp / sum

    # forward
    Z = X @ W # (N,C)
    Z -= np.max(Z,axis=1).reshape(-1,1)
    A = softmax(Z) # (N,C)
    class_probs = A[np.arange(N), y] # (500,)
    loss = -np.mean(np.log(class_probs)) + reg*np.sum(W**2)

    # backward
    dl_dZ = A # (N,C)
    dl_dZ[np.arange(N), y] -= 1 # dl/dZ_jk = A_jk-1 if k == yi else a_jk
    dZ_dW = X # (N,D)
    dW = (dZ_dW.T @ dl_dZ)/N + 2*reg*W # (D,C) 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
