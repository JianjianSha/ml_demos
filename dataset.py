from svm import train
from matplotlib.collections import PolyCollection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.core.fromnumeric import mean, squeeze
import torch
import torch.nn.functional as F
import itertools
from torch.distributions.multivariate_normal import MultivariateNormal


def create_bin_dataset(D, N, no=0):
    '''
    Create a dataset for binary-classification

    @parameters:
    D: Dimension of features
    N: Dataset size
    no: Number of outliers

    @return: a tuple of (w, x), where
    w: with shape of (D)
    b: bias, (1,)
    x: with shape of (N,D).
    y: target, (N,)
    '''
    assert N > 9 and D > 0 and no < N // 10
    w = torch.rand(1,D) * 0.4 + 0.3
    w = F.normalize(w)  # exp_i / (sum_j exp_j)
    nw = torch.randint(D, (1,))   # number of negative weights
    idx_nw = torch.randint(D, (nw,))
    w[0,idx_nw] *= -1
    b = torch.rand(1)

    x = torch.rand((N, D))
    scale = 5 * int(np.log2(N))
    # shift x(without the last dimension) from [0, 1) to [-scale/2, scale/2)
    x[:,:-1] *= scale 
    x[:,:-1] -= scale / 2
    # for the last dimension of x, we shift it and then see it as the axis distance 
    #   between data point and superplane.
    #   What is the axis distance? Add a auxiliry line through the data point and 
    #   parallel with the axis(here, we use the last axis),  then we get an  
    #   intersection point of the superplane and the auxiliry line, and the 
    #   distance between intersection point and data point is the so called
    #   "axis distance".
    x[:,-1] *= scale / 5
    x[:,-1] += 2.0

    if no > 0:      # set outliers
        idx_no = torch.randint(N, (no,))
        scale_no = torch.rand(no) * 0.1 - 0.5
        x[idx_no,-1] *= scale_no

    # then the last value of x is: add/substract the axis distance to/from the last 
    #   value of intersection
    #`  1. calculate the last value of intersection. according to w * x + b = 0,
    #       (think x as a single data point temporarily)
    #               xi = (-b - w[:-1] * x[:-1]) / w[-1]
    #   2. add/substract axis distance (i.e., x[-1]) to/from xi
    #   do 'add' for positive examples while 'substract' for negative examples
    xi = (-b - torch.mm(x[:,:-1], w[:,:-1].T)) / w[0,-1]   # xi's shape: (N, 1)

    nn = torch.randint(N//4, N*3//4, (1,))        # number of negative examples
    idx_n = torch.randint(N, (nn,))         # indices of negative examples

    # targets
    y = torch.ones(N)
    y[idx_n] = -1

    # do the second step: add/substract axis distance (i.e., x[-1]) to/from xi
    x[idx_n,-1] *= -1
    x[:,-1] += xi[:,-1]

    # pack all data into a tuple
    return x, y, w[0], b


def create_dataset_linear(D, N, C):
    '''
    Create a multi-class dataset, and data points of each class are generated by 
    a linearly distribution.

    We can assume these C classes can be seperated by C-1 superplanes. This is
    reasonable. Suppose there are 4 classes with related 2-dim features, as 
    following:
               ^ y
               |
        c1     |     c2
    -------------------------> x
        c3     |     c4
               |
    then we can map these datas into a space with a higher dimension(e.g. 3-dim),
    and the front view graph may be
                ^ z
                |        c1
    --------------------------
                |        c3  
    --------------------------
                |        c2
    -------------------------> (x-y plane)
                |        c4
    
    @returns:
        returns a 3-tuple (w, b, x), where
        w: with a shape of (D)
        b: with a shape of (C-1), i.e., there are C-1 parallel superplanes
        x: with a shape of (N, D)
        y: target, (N,)
    '''
    assert C > 1, 'Number of classes must be > 1, but got %d' % C
    assert N//C > 5, 'Number of datas for each class should be larger than '\
        "5 on average, but got: (N//C=)%d" % (N//C)
    w = torch.rand(1,D)*0.5+0.2
    w = F.normalize(w)  # exp_i / (sum_j exp_j)     # shape: (1,D)
    nw = torch.randint(D, (1,))   # number of negative weights
    idx_nw = torch.randint(D, (nw,))
    w[0,idx_nw] *= -1

    scale = int(np.log2(N))

    # get the distance along the last dim between each two neighbouring supreplanes
    bs = torch.rand(C-1)*3 + scale * D             # shape: (C-1,)
    # sum cumulatively to the biases of all superplanes
    b = torch.cumsum(bs, dim=0)

    ns = torch.randint(N//C-4, N//C+4, (C-1,))      # shape: (C-1,)
    # do cumsum to get sentinels for C classes, i.e.,
    # 0 ~ ss[0]         is the first class
    # ss[0] ~ ss[1]     is the second class
    # ...
    # ss[C-3] ~ ss[C-2] is the (C-1)-th class
    # ss[C-2] ~         is the C-th class
    ss = torch.cumsum(ns, dim=0)                           # shape: (C-1,)
    
    
    # shift x (except the values of last dim) from [0, 1) to [-scale, scale)
    # we use uniform distribution, so the points (projection on subspace except 
    #   the last dim) arange linearly,
    #   but if we use gaussian distribution instead, then points (projection on
    #   subspace except the last dim) arange in  the form of ellipse.
    x = torch.randn((N, D)) * 2 * scale - scale   # shape: (N, D-1)
    

    # re-calculate the values of last dim
    scale_d = (bs[1] if C > 2 else bs[0]) / torch.abs(w[0, -1])

    # for the last dim, use randn or rand, both work well.
    # lx = torch.randn(N) * scale_d * (0.05 if C > 2 else 0.1)
    lx = torch.rand(N) * D

    scale_d = torch.rand(1)*0.4+0.3 * scale_d
    y = []
    sign = torch.sign(w[0, -1])
    for c in range(C):
        if c != C-1:
            y.append(torch.ones(ns[c])*c)
        else:
            y.append(torch.ones(N-ss[-1])*c)
        if c == 0:
            # the last axis coordinates of intersections
            xi = (-b[c] - torch.mm(x[:ss[0],:-1], w[:,:-1].T).squeeze()) / w[0,-1]
            d = lx[:ss[c]] + scale_d
            x[:ss[0],-1] = xi + sign * d
        else:
            end_idx = ss[c] if c != C-1 else N
            xi = (-b[c-1] - torch.mm(x[ss[c-1]:end_idx,:-1], w[:,:-1].T).squeeze()) / w[0,-1]
            d = lx[ss[c-1]:end_idx] + scale_d
            x[ss[c-1]:end_idx,-1] = xi - sign * d
            scale_d = torch.abs(bs[c]/w[0,-1]) - scale_d if c!= C-1 else 0
        
    y = torch.hstack(y)
    return x, y, w[0], b

            
def create_dataset_normal(D, N, C):
    '''
    Create a multi-class dataset, and data points of each class are generated by 
    a normal distribution.

    Use C-1 parallel superplanes to seperate the dataset containing C classes.
    Different with `create_dataset_linear` which generate data with a linear 
    arrangement, this method generate data by sampling from a normal distribution.

    Centers of clusters of all classes are located at a straight line which is 
    perpendicular with these parallel superplanes.
    '''
    w = torch.rand(1,D)*0.5+0.2
    w = F.normalize(w)  # exp_i / (sum_j exp_j)     # shape: (1,D)
    nw = torch.randint(D, (1,))   # number of negative weights
    idx_nw = torch.randint(D, (nw,))
    w[0,idx_nw] *= -1

    bs = (torch.rand(C-1) * 5 + 20) * 1
    b  = torch.cumsum(bs, dim=0)


    #======================== method 1. ================================
    #
    # xc1 = torch.randint(5, 10, (C, D-1)).float()    # (C, D-1)
    # mask = torch.rand(C, D-1) > 0.5
    # xc1[mask] *= -1
    # xc1 = torch.cumsum(xc1, dim=0)            # (C, D-1)

    # xi = torch.mm(xc1, w[:,:-1].T).squeeze()          # (C,)
    # dxi = torch.rand(C) * 5 + 5
    # xi[:-1] = (-b - xi[:-1] + dxi[:-1]) / w[0, -1]
    # xi[-1] = (-b[-1] - xi[-1] - dxi[-1]) / w[0, -1]
    
    # xc = torch.empty(C, D)
    # xc[:,:-1] = xc1
    # xc[:,-1]  = xi
    # # return xc, torch.arange(C), w[0], b     # plot centers
    # # calculate the true distance between center and superplanes
    # # xc: (C, D);  w: (1, D)
    # distances = torch.mm(xc, w.T).squeeze()             # (C,)
    # distances[:-1] = torch.abs(distances[:-1] + b)
    # distances[-1]  = torch.abs(distances[-1] + b[-1])

    # radius = (torch.rand(C)*0.2+0.4) * distances              # (C,)
    #===================================================================

    #================================ method 2. ==================================
    # Another better implementation, compared with method 1.
    # features:
    #   1. Two clusters located at two sides of one superplane, margins of these
    #       two cluseters are equal, but radius can be nonequal.
    #   2. Centers of all clusters are located at the normal line of these
    #       parallel superplanes. In fact, there is no need to askew these 
    #       centers; otherwise, superplanes cannot be parallel and the dataset
    #       may not be seperable linearly.
    xc = torch.randint(5, 10, (C, D)).float()       # first center coordinates
    dx = torch.min(bs) * 0.4
    radius = torch.empty(C)
    radius[0] = dx * (torch.rand(1) * 0.2+0.3)

    xc[0,-1] = (-b[0] - torch.dot(xc[0,:-1], w[0,:-1]).squeeze() + dx) / w[0,-1]
    margin = dx - radius[0]

    for c in range(1, C-1):
        radius[c] = ((bs[c] - margin) / 2) * (torch.rand(1) * 0.1+0.3)
        xc[c] = xc[c-1] - w * (dx + margin + radius[c])
        dx = bs[c] - (margin + radius[c])
        margin = dx - radius[c]
    
    radius[-1] = torch.max(bs) * (torch.rand(1)*0.3+0.4)
    xc[-1] = xc[-2] - w * (dx + margin + radius[-1])
    # return xc, torch.arange(C), w[0], b     # plot centers
    #=============================================================================

    

    # initialize numbers of all classes
    ns = torch.randint(N//C-4, N//C+4, (C,))      # shape: (C,)
    ns[-1] = N - torch.sum(ns[:-1])
    ss = torch.cumsum(ns, dim=0)                           # shape: (C,)

    x = torch.empty(N, D)
    y = torch.ones(N)
    for c in range(C):
        start_idx = ss[c-1] if c > 0 else 0
        mn = MultivariateNormal(xc[c], torch.eye(D) * radius[c])
        # because ticks of all axises are nonequal and always differ much, 
        # the points sampled seem not like a sphere, but in fact, they
        # form a sphere.
        samples =  mn.sample(torch.Size([int(ns[c])]))
        x[start_idx:ss[c],:] = samples
        y[start_idx:ss[c]] *= c
    
    return x, y, w[0], b
    

def create_dataset_cylindrical(D, N, C):
    assert D > 1, 'Number of features must be larger than 1, but got: %d' % D


    mean_theta = 2 * torch.pi / C
    delta_theta = mean_theta * 0.15                 # max delta of angle, when disturbed
    delta_thetas = torch.rand(C) * delta_theta
    delta_thetas[torch.rand(C) > 0.5] *= -1
    thetas = torch.arange(C+1) * mean_theta         # (C+1)
    thetas[:-1] += delta_thetas
    thetas[thetas < 0] += torch.pi * 2
    # thetas are the marginal angle of class ranges
    # 1-st class: [thetas[0], thetas[1])
    # 2-nd class: [thetas[1], thetas[2])
    # ...
    # C-th class: [theta[C-1], thetas[C])
    thetas[-1] = thetas[0]

    mn = MultivariateNormal(torch.zeros(D), torch.eye(D))
    x = mn.sample(torch.Size([N]))
    y = torch.zeros(N, dtype=torch.int16)
    norm = torch.norm(x[:,:2], dim=1)

    ss = torch.asin(x[:,1]/norm)        # shape: (N,), range: [-90, 90]
    cs = torch.acos(x[:,0]/norm)        # shape: (N,), range: [0, 180]
    angles = torch.empty(N)
    fst_idx = (ss >= 0) & (cs <= torch.pi / 2)
    snd_idx = (ss >= 0) & (cs > torch.pi / 2)
    trd_idx = (ss < 0) & (cs > torch.pi / 2)
    fth_idx = (ss < 0) & (cs <= torch.pi / 2)
    angles[fst_idx] = ss[fst_idx]
    angles[snd_idx] = cs[snd_idx]
    angles[trd_idx] = torch.pi - ss[trd_idx]
    angles[fth_idx] = torch.pi * 2 + ss[fth_idx]


    for c in range(C):
        if thetas[c] < thetas[c+1]:
            indices = (thetas[c] <= angles) & (angles < thetas[c+1])
        else:
            fst = (thetas[c] <= angles) & (angles < torch.pi * 2)
            snd = (0 <= angles) & (angles < thetas[c+1])
            indices = fst | snd
        y[indices] = c
    return x, y, thetas[:-1]



def plot_cylindrical(x, y, angles, pw=None):
    '''
    Plot the dataset for classification, especially for multi classification.
    Each class is corresponding to a cylinder perpenticular with x-y plane.

    @parameters:
    x: (N, D)
    y: (N,), targets
    angles: (C,), angles of projection of cylinders onto x-y planes
        1. [angles[0], angles[1])  -> class 1
        2. [angles[1], angles[2])  -> class 2
        3. ...
        4. [angles[C-2], angles[C-1]) -> class C-1
        5. [angles[C-1], angles[0])   -> class C
        All angles are in range of [0, pi * 2), and in increased order except
        the first angle `angles[0]`, because `angles[0]` may be less or larger 
        then `angles[1]`, and if the latter is the fact, we then check if
        some angle in [angles[0], pi*2) \cup [0, angles[1]), and if so, the 
        data point related to this angle belongs to class 1.
    pw: (C, D+1), predicator weights. the last column of weights is bias.
        Because the dataset shown here is arranged around the origin point `O`,
        so the bias should be zero in idea case, so we show weights except the
        bias with arrows plotted on the graph, and print the bias in title line.
    '''
    classes = torch.unique(y)
    C = len(classes)          # Number of classes
    N = x.shape[0]
    D = x.shape[1]
    group_masks = [y == c for c in classes]
    subplot_kw = {'projection': '3d'} if D == 3 else None
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if x.shape[1] == 3:
        ax.set_zlabel('z')
    

    num_x = 59             # number points used to plot margin planes
    x_min, x_max = torch.min(x, dim=0)[0], torch.max(x, dim=0)[0]       # (D,), (D,)
    lx = torch.linspace(x_min[0], 0, num_x)
    lx = torch.hstack((lx, torch.linspace(0, x_max[0], num_x+1)))#.repeat((C,1))
    if D == 3:
        lz = torch.linspace(x_min[-1], x_max[-1], 100)
        lx, lz = torch.meshgrid(lx, lz)
        # print(lx.shape, lz.shape)
        lx = lx.repeat((C, 1, 1))
        lz = lz.repeat((C, 1, 1))
        # print(lx.shape, lz.shape)
        ly = torch.tan(angles).repeat((1, 1, 1)).permute(2, 0, 1)*lx
    else:
        lx = lx.repeat((C, 1))
        ly = torch.tan(angles).unsqueeze(-1) * lx
    lines = []
    for a in range(C):
        angle = angles[a]
        if angle < torch.pi / 2 or angle >= torch.pi * 3 / 2:
            st_idx = num_x
            ed_idx = num_x * 2 + 1
        else:
            st_idx = 0
            ed_idx = num_x + 1
        if D == 3:
            cur_lx = lx[a,st_idx:ed_idx,:]
            cur_ly = ly[a,st_idx:ed_idx,:]
            cur_lz = lz[a,st_idx:ed_idx,:]
            mask_y = (cur_ly >= x_min[1]) & (cur_ly <= x_max[1])
            cur_lx = cur_lx[mask_y].reshape(-1, lx.shape[-1])
            cur_ly = cur_ly[mask_y].reshape(-1, lx.shape[-1])
            cur_lz = cur_lz[mask_y].reshape(-1, lx.shape[-1])
            lines.append([cur_lx, cur_ly, cur_lz])
        else:
            cur_lx = lx[a,st_idx:ed_idx]
            cur_ly = ly[a,st_idx:ed_idx]
            mask_y = (cur_ly >= x_min[1]) & (cur_ly <= x_max[1])
            cur_lx = cur_lx[mask_y]
            cur_ly = cur_ly[mask_y]
            lines.append([cur_lx, cur_ly])


    if D == 3:
        for i in range(C):
            s = ax.plot_surface(lines[i][0].numpy(), lines[i][1].numpy(), lines[i][2].numpy(), 
                                alpha=0.5, label='angle: {:.2f}'.format(angles[i] * 180 / torch.pi))
            s._facecolors2d = s._facecolor3d
            s._edgecolors2d = s._edgecolor3d
    else:
        args = itertools.chain(*lines)
        ax.plot(*args)

    title = f'D: {D}, N: {N}, C: {C}'
    

    if pw is not None:
        if pw.shape[1] == D + 1:
            rw = np.around(pw.numpy(), 2)
            title += f', bias: {rw[:,-1]}'
        w = pw * D * 1.5
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        for i in range(C):
            if D == 3:
                ax.quiver3D(0, 0, 0, w[i,0], w[i,1], w[i,2], label=f'{i}', linestyle='--', color=colors[i])
            else:
                # ax.quiver(0, 0, w[i,0], w[i,1], scale=0.5)
                ax.arrow(0,0,w[i,0],w[i,1], head_width=0.1, label=f'{i}', color=colors[i])

    ax.set_title(title)

    for mask, c in zip(group_masks, classes):
        group_x = x[mask,:]
        if group_x.shape[1] == 1:
            ax.scatter(group_x[:,0], torch.zeros(group_x.shape[0]), label='%d' % int(c))
        elif group_x.shape[1] == 2:
            ax.scatter(group_x[:,0], group_x[:,1], label='%d' % int(c))
        else:   # shape[1] must be 3, otherwise dataset cannot be plotted
            ax.scatter(group_x[:,0], group_x[:,1], group_x[:,2], label='%d' % int(c))
    ax.legend(bbox_to_anchor=(0.9, 0.8), loc=6)
    plt.show()


def plot(x, y, tw=None, tb=None, pw=None, pb=None):
    '''
    plot dataset(for binary classification task) and the superplane (if given)
    Number of features must be in range [1, 3]
    Let `C` be the number of classes.

    @parameters:
    x: (N, D)
    y: (N,), target
    tw: (D), true weights
    tb: (C-1,), true bias
    pw: (D), predictor weights
    pb: (C-1), predictor bias
    '''
    def plot_superplane(w, b, predictor=False):
        if w is not None:
            if x.shape[1] == 1:
                # 10 is a arbitrary number, which means we use 10 points to 
                # plot a superplane
                fmt = '--' if predictor else '-'
                xw = torch.ones((C-1, 10)) * (-b.unsqueeze(-1)/w[0])     # (C-1, 10)
                yw = torch.linspace(-1, 1, 10).expand(C-1, 10)
                args = itertools.chain(*([xw[i,:], yw[i,:], fmt] for i in range(C-1)))
                ax.plot(*args)
            elif x.shape[1] == 2:
                fmt = '--' if predictor else '-'
                mi, ma = torch.min(x[:,0]), torch.max(x[:,0])
                xw = torch.linspace(mi, ma, N).expand(C-1, N)     # (C-1, N)
                yw = (-b.expand(N, C-1).T - xw * w[0]) / w[1]   # xi's shape: (N, 1)
                args = itertools.chain(*([xw[i,:], yw[i,:], fmt] for i in range(C-1)))
                ax.plot(*args)
            else:
                # get the margin of superplane (which is the part to be shown)
                mi, ma = torch.amin(x[:,:2], dim=0), torch.amax(x[:,:2], dim=0)
                x1 = torch.linspace(mi[0], ma[0], N)
                x2 = torch.linspace(mi[1], ma[1], N)
                x1, x2 = torch.meshgrid(x1, x2)
                x1 = x1.expand(C-1, N, N)
                x2 = x2.expand(C-1, N, N)
                x3 = (-b.expand(N, N, C-1).permute(2, 0, 1)- x1*w[0]-x2*w[1]) / w[2]
                lp = 'pb' if predictor else 'tb'
                cmap = cm.coolwarm if predictor else cm.rainbow
                for i in range(C-1):
                    s = ax.plot_surface(x1[i,:].numpy(), x2[i,:].numpy(), x3[i,:].numpy(), alpha=0.5, label='{}: {:.2f}'.format(lp, b[i]))
                    s._facecolors2d = s._facecolor3d
                    s._edgecolors2d = s._edgecolor3d
    assert 1 <= x.shape[1] <= 3, "Data can only be plotted when its number of "\
        "features is in range [1, 3], but got %d" % x.shape[1]

    classes = torch.unique(y)
    group_masks = [y == c for c in classes]
    # groups  = [x[x[:,-1]==c,:] for c in classes]
    
    subplot_kw = {'projection': '3d'} if x.shape[1] == 3 else None
    fig, ax = plt.subplots(subplot_kw=subplot_kw)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if x.shape[1] == 3:
        ax.set_zlabel('z')
    C = len(classes)          # Number of classes
    N = x.shape[0]
    D = x.shape[1]
    plot_superplane(tw, tb)
    plot_superplane(pw, pb, predictor=True)

    title = f'D: {D}, N: {N}, C: {C}'
    if tw is not None:
        rw = np.around(tw.numpy(), 2)        # rounded weights
        title += f', tw: {rw}'
    if pw is not None:
        rw = np.around(pw.numpy(), 2)
        title += f', pw: {rw}'
    ax.set_title(title)


    for mask, c in zip(group_masks, classes):
        group_x = x[mask,:]
        if group_x.shape[1] == 1:
            ax.scatter(group_x[:,0], torch.zeros(group_x.shape[0]), label='%d' % int(c))
        elif group_x.shape[1] == 2:
            ax.scatter(group_x[:,0], group_x[:,1], label='%d' % int(c))
        else:   # shape[1] must be 3
            ax.scatter(group_x[:,0], group_x[:,1], group_x[:,2], label='%d' % int(c))
    ax.legend()
    plt.show()




if __name__ == '__main__':
    pass