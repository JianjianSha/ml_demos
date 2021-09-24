'''
A SVM demo for 2-dim features

Please refer to https://jianjiansha.github.io/2021/09/22/ml/svm/ for more 
details of this algorithm.
'''
import numpy as np
import matplotlib.pyplot as plt


def predictor():
    # Randomly create an array of 3 parameters which represents
    # some true predictor line
    w1 = np.random.rand()*0.5+0.2
    w2 = np.sqrt(1-w1**2) if np.random.rand()>0.5 else -np.sqrt(1-w1**2)
    w3 = np.random.uniform(-5, 5)
    w = np.array([w1, w2, w3]).reshape((1, 3))
    return w

    # print(w)
    # print(wt)
    # print(np.sum(w**2))
    # print(np.sum(wt**2))


def predictor_y(w, x):
    '''
    Get y(or x2) values of the predictor according to x(or x1) values
    w1*x+w2*y+b=0
    y=-(b+w1*x)/w2
    '''
    assert x.ndim == 1
    return -(w[0][2]+w[0][0]*x)/w[0][1]


def train_set(w, n=10, y_gamma=1.5):
    '''
    Randoms create a train set, with positive and negative samples
    located at two different side of the predictor line, respectly.

    @parameters:
    y_gamma: This parameter is different with gamma, which represents
                the distance between one sample and the superplane. 
                In fact, gamma = y_gamma * sin(theta), where the theta
                means the angle between y-axis and the superplane.
    '''
    x = np.random.rand(4, n)
    x[:2,:] = x[:2] * 10 - 5
    x[2:,:] = x[2:] * 2 + y_gamma
    
    P = np.vstack((x[0], predictor_y(w, x[0])+x[2], np.ones((1, n))))
    N = np.vstack((x[1], predictor_y(w, x[1])-x[3], np.ones((1, n))))

    if np.prod(w[0][:2]) < 0:
        P, N = N, P

    return P, N


def plot(w, P, N, wo=None):
    # show the plot of:
    # 1. the true predictor
    # 2. positive and negative sample points
    # 3. output predictor after the SVM training (optional)
    wt = np.around(w[0], 3)
    wot = np.around(wo[0], 3)
    true_label = f"true: {wt}"
    out_label = f"out: {wot}"
    x = np.arange(-10, 10, 0.1)
    y = predictor_y(w, x)

    fig, ax = plt.subplots()  # Create a figure and an axes.
    ax.plot(x, y, label=true_label)  # Plot the predictor line
    ax.set_xlabel('x1')  # Add an x-label to the axes.
    ax.set_ylabel('x2')  # Add a y-label to the axes.
    ax.set_title("SVM")  # Add a title to the axes.
    ax.scatter(P[0], P[1], color='r', label='positive')
    ax.scatter(N[0], N[1], color='b', label='negative')
    if wo is not None:
        yo = predictor_y(wo, x)
        ax.plot(x, yo, 'g--', label=out_label)

    ax.legend()  # Add a legend.
    plt.show()


def train(P, N, T=100000):
    '''
    SVM training

    @parameters:
    P: positive samples
    N: negative samples
    T: total iterating number
    '''
    X = np.hstack((P, N))
    Y = np.hstack((np.ones((1, P.shape[1])), -np.ones((1, N.shape[1]))))
    Z = np.vstack((X, Y))
    lamda = 0.01
    theta = np.zeros(3)
    W = np.empty((T, 3))
    for t in range(1, T):
        W[t, :] = 1/(lamda*(t+1)) * theta
        i = np.random.randint(Z.shape[1])
        z = Z[:, i]
        if z[-1]*np.dot(W[t, :], z[:-1]) < 1:
            theta += (z[-1]*z[:-1])
    return W.sum(axis=0, keepdims=True)/T




def main():
    w = predictor()
    P, N = train_set(w, n=50)
    wo = train(P, N)
    plot(w, P, N, wo)


if __name__ == '__main__':
    main()