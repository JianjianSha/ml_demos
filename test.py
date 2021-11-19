from dataset import *
from svm import *


def cylindrical_demo():
    D = np.random.randint(2, 4)
    N = np.random.randint(40, 500)
    C = np.random.randint(2, 6)
    x, y, angles = create_dataset_cylindrical(D, N, C)
    w = train(x, y)
    plot_cylindrical(x, y, angles, w)

def bin_demo():
    D = np.random.randint(1, 4)
    N = np.random.randint(40, 500)
    no = np.random.randint(0, int(np.log(N)))
    plot(*create_bin_dataset(D, N, no=no))

def multi_demo():
    D = np.random.randint(1, 4)
    N = np.random.randint(40, 500)
    C = np.random.randint(2, 6)     # 6 is the upper limit, and can be increased
    plot(*create_dataset_normal(D, N, C))


def main():
    D = np.random.randint(1, 4)
    N = np.random.randint(40, 500)
    C = np.random.randint(2, 6)
    # x,y,w,b = create_multi_dataset_linear(D, N, 2)
    x,y,w,b = create_dataset_normal(D, N, 2)
    pw, pb = train_bin(x, y, T=N * 500)
    plot(x, y, w, b, pw, pb)



if __name__ == '__main__':
    cylindrical_demo()