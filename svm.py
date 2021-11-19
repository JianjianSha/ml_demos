'''
A SVM demo for 2-dim features

Please refer to https://jianjiansha.github.io/2021/09/22/ml/svm/ for more 
details of this algorithm.
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def train_bin(x, y, lmbda=0.01, T=100000):
    '''
    train a SVM model for binary classification on dataset.
    It need not to assume the dataset is seperable strictly.

    @parameters:
    x: features, (N, D)
    y: targets, (N,). target value set Y must be {0,1} or {1,-1}
    lmbda: lambda, learning rate
    T: epochs of training

    @return:
    w: weights, (D,)
    b: bias, (1,)
    '''
    # save the last k weights
    k = 100
    # y \in {-1, 1}
    if torch.min(y) == 0:
        y = y * 2 -1
    
    # convert to homogeneous problem
    x = torch.hstack((x, torch.ones(x.shape[0], 1)))

    D = x.shape[1]      # Number of features
    N = x.shape[0]
    W = torch.empty(k, D)

    theta = torch.zeros(D)
    I = torch.randint(N, (T,))
    # print(W.shape, x.shape, y.shape)
    for t in range(T):
        idx = t % k
        W[idx,:] = 1/(lmbda * (t+1)) * theta
        if y[I[t]] * torch.dot(W[idx,:], x[I[t],:]) < 1:
            theta += y[I[t]] * x[I[t],:]
    w = W.sum(axis=0, keepdim=True) / k

    # NOTICE: not equal to F.normalize(w, dim=1). The last value of w 
    #   is bias and does not participate in normalization.
    norm = torch.norm(w[0,:-1])
    w /= norm
    
    return w[0, :-1], w[:,-1]



def train(x, y, lr=0.001, epochs=50, B=5000, delta=None, bias=False):
    '''
    Train a multi-class model.
    Let the number of all classes be `C`, then
    target value set must be {0,1,...,C-1}.

    
    @parameters:
    x: (N, D)
    y: (N,), targets
    lr: learning rate
    epochs: Number of epochs
    B: batch
    delta: original loss function, usually using 0-1 loss.
    bias: Whether to use bias or not. If true, then recreate features
        as following:
            x = torch.hstack((x, torch.ones(x.shape[0], 1)))
        But commonly, there is no need to use bias.

    @return:
    A normalized weights with a shape of 
    '''
    # convert to homogeneous problem
    if bias:
        x = torch.hstack((x, torch.ones(x.shape[0], 1)))

    k = 100                     # save the latest k weights
    N, D = x.shape
    classes = torch.unique(y)
    C = len(classes)
    w = torch.zeros((C, D))*2-1     # the latest weight
    W = torch.empty(k, C, D)        # checkpoints
    if delta is None:
        # 0-1 loss, as the default
        delta = lambda c: (torch.arange(C) != c).float()
    for epoch in range(epochs):
        for i in range(B):
            I = torch.randint(N, (B,))
            cx, cy = x[I[i],:], y[I[i]]            # randomly choose a point (x,y)
            trd = torch.dot(w[cy,:], cx)                      # scale
            snd = torch.mm(w, cx.unsqueeze(-1)).squeeze()    # (C,)
            fst = delta(cy)                                          # (C,)
            hat_y = torch.argmax(fst + snd - trd)                   # the output (predicated class for data)
            w[hat_y,:] -= lr * cx
            w[cy,:] += lr * cx
            W[(epoch*B+i)%k,...] = w
        print('epoch:', epoch+1)
    w = W.sum(axis=0) / k                   # (C, D)
    norm = torch.norm(w[:,:D], dim=1).unsqueeze(-1)     # (C,1)
    w /= norm
    # return F.normalize(w, dim=1)
    return w


def predict(x, w):
    '''
    predictor:
        h(x) = argmax_y <w, Phi(x,y)>

    This prediction method does well for binary and multiply
    classification. 
    
    For binary classification, if w has a 
    shape of (D), then convert it to [w;-w] and the shape
    is changed to (2,D). The predicted value hat_y is 
    0 or 1, which can also be interpretated as 1 or -1.

    @parameters:
    x: (N, D), where N represents the number of all datas to
        be predicted, while D is the number of features.
    w: (C, D), where C is the number of all classes.

    @return:
        an index tensor with a shape of (N,), and value of
        each element is in the range of [0, C)
    '''
    # 
    wPhi= torch.mm(x, w.T)  # (N, C)
    return torch.argmax(wPhi, dim=1)   # (N,)
