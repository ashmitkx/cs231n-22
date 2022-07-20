from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        dS_dW = X[i].reshape(-1,1) # (D,1)
        dL_dS = [0 for _ in range(num_classes)]
        n_above_margin = 0

        for j in range(num_classes):
            if j == y[i]:
                continue
            
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin

                dL_dS[j] = 1. # dL_dS[j] = 1 if margin > 0 else 0, given j != y[i]
                n_above_margin += 1
        
        dL_dS[y[i]] = -n_above_margin # dL_dS[j] = -n_above_margin, given j == y[i]
        dL_dS = np.array(dL_dS).reshape(-1,1) # (C,1)

        dW += dS_dW @ dL_dS.T # (D,C)

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]

    y = y.reshape(-1,1) #(N,1)
    scores = X @ W # (N,C)

    correct_class_idxs = (np.arange(N),y[:,0])
    correct_class_scores = scores[correct_class_idxs].reshape(-1,1) # (N,1)

    margins = scores - correct_class_scores + 1 # (N,C)
    margins[correct_class_idxs] = 0  # make correct score positions 0, so they dont contribute to loss
    margins[margins <= 0] = 0 # apply max(0, margins)

    loss = np.sum(margins) / N
    loss += reg * np.sum(W**2)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dS_dW = X # (N,D)
    
    dL_dS = np.zeros_like(margins) # (N,C)
    dL_dS[margins > 0] = 1 # dL_dS[ij] = 1 if margin > 0 else 0, given j != y[i]
    ns_above_margin = np.sum(margins > 0, axis = 1).reshape(-1,1) # (N,1)
    dL_dS[correct_class_idxs] = -ns_above_margin[:,0] # set gradient of correct score positions

    dW = (dS_dW.T @ dL_dS) / N
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
