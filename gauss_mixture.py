import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import os
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

def visualize_3d_gmm(points, w, mu, stdev, export=True):
    '''
    plots points and their corresponding gmm model in 3D
    Input:
        points: N X 3, sampled points
        w: n_gaussians, gmm weights
        mu: 3 X n_gaussians, gmm means
        stdev: 3 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    '''

    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))
    # Visualize data
    fig = plt.figure(figsize=(8, 8))
    axes = fig.add_subplot(111, projection='3d')
    n = 10
    axes.set_xlim(np.array([-1, 1])*n)
    axes.set_ylim(np.array([-1, 1])*n)
    axes.set_zlim(np.array([-1, 1])*n)
    plt.set_cmap('Set1')
    colors = cmx.Set1(np.linspace(0, 1, n_gaussians))
    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        axes.scatter(points[idx, 0], points[idx, 1], points[idx, 2], alpha=0.3, c=colors[i])
        plot_sphere(w=w[i], c=mu[:, i], r=stdev[:, i], ax=axes)

    plt.title('3D GMM')
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.view_init(35.246, 45)
    if export:
        if not os.path.exists('images/'): os.mkdir('images/')
        plt.savefig('images/3D_GMM_demonstration.png', dpi=100, format='png')
    plt.show()


def plot_sphere(w=0, c=[0,0,0], r=[1, 1, 1], subdev=10, ax=None, sigma_multiplier=3):
    '''
        plot a sphere surface
        Input:
            c: 3 elements list, sphere center
            r: 3 element list, sphere original scale in each axis ( allowing to draw elipsoids)
            subdiv: scalar, number of subdivisions (subdivision^2 points sampled on the surface)
            ax: optional pyplot axis object to plot the sphere in.
            sigma_multiplier: sphere additional scale (choosing an std value when plotting gaussians)
        Output:
            ax: pyplot axis object
    '''

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:complex(0,subdev), 0.0:2.0 * pi:complex(0,subdev)]
    x = sigma_multiplier*r[0] * sin(phi) * cos(theta) + c[0]
    y = sigma_multiplier*r[1] * sin(phi) * sin(theta) + c[1]
    z = sigma_multiplier*r[2] * cos(phi) + c[2]
    cmap = cmx.ScalarMappable()
    cmap.set_cmap('jet')
    c = cmap.to_rgba(w)

    ax.plot_surface(x, y, z, color=c, alpha=0.2, linewidth=1)

    return ax

def visualize_2D_gmm(points, w, mu, stdev, export=True):
    '''
    plots points and their corresponding gmm model in 2D
    Input:
        points: N X 2, sampled points
        w: n_gaussians, gmm weights
        mu: 2 X n_gaussians, gmm means
        stdev: 2 X n_gaussians, gmm standard deviation (assuming diagonal covariance matrix)
    Output:
        None
    '''
    n_gaussians = mu.shape[1]
    N = int(np.round(points.shape[0] / n_gaussians))
    # Visualize data
    fig = plt.figure(figsize=(8, 8))
    axes = plt.gca()
    axes.set_xlim([-1, 1]*40)
    axes.set_ylim([-1, 1]*40)
    plt.set_cmap('Set1')
    colors = cmx.Set1(np.linspace(0, 1, n_gaussians))
    for i in range(n_gaussians):
        idx = range(i * N, (i + 1) * N)
        plt.scatter(points[idx, 0], points[idx, 1], alpha=0.3, c=colors[i])
        for j in range(8):
            axes.add_patch(
                patches.Ellipse(mu[:, i], width=(j+1) * stdev[0, i], height=(j+1) *  stdev[1, i], fill=False, color=[0.0, 0.0, 1.0, 1.0/(0.5*j+1)]))
        plt.title('GMM')
    plt.xlabel('X')
    plt.ylabel('Y')

    if export:
        if not os.path.exists('images/'): os.mkdir('images/')
        plt.savefig('images/2D_GMM_demonstration.png', dpi=100, format='png')

    plt.show()

def efit(x, y, z):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    D = [
        x * x + y * y - 2 * z * z, x * x + z * z - 2 * y * y, 2 * x * y,
        2 * x * z, 2 * y * z, 2 * x, 2 * y, 2 * z, 1 + 0 * x
    ]
    D = np.array(D)

    d2 = x * x + y * y + z * z
    d2 = d2.reshape((d2.shape[0], 1))
    Q = np.dot(D, D.T)
    b = np.dot(D, d2)
    u = np.linalg.solve(Q, b)

    v = np.zeros((u.shape[0] + 1, u.shape[1]))
    v[0] = u[0] + u[1] - 1
    v[1] = u[0] - 2 * u[1] - 1
    v[2] = u[1] - 2 * u[0] - 1
    v[3:10] = u[2:9]

    A = np.array([
        v[0], v[3], v[4], v[6], v[3], v[1], v[5], v[7], v[4], v[5], v[2], v[8],
        v[6], v[7], v[8], v[9]
    ]).reshape((4, 4))

    center = np.linalg.solve(-A[:3, :3], v[6:9])
    T = np.eye(4)
    T[3, :3] = center.T
    center = center.reshape((3, ))
    R = T.dot(A).dot(T.conj().T)
    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])

    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    sgns = np.sign(evals)
    radii = np.sqrt(sgns / evals)

    d = np.array([x - center[0], y - center[1], z - center[2]])
    d = np.dot(d.T, evecs)
    d = np.array([d[:, 0] / radii[0], d[:, 1] / radii[1],
                  d[:, 2] / radii[2]]).T
    chi2 = np.sum(
        np.abs(1 - np.sum(d**2 * np.tile(sgns, (d.shape[0], 1)), axis=1)))

    return center, radii, evecs, v, chi2



bckg_pts_all = np.load("/home/cstar/workspace/grid-data/bckg_pts_all.npy")[::1000]
frg_pts_all = np.load("/home/cstar/workspace/grid-data/frg_pts_all.npy")
print("bckg_pts shape: ", bckg_pts_all.shape, bckg_pts_all.dtype)

bckg_pts_all = bckg_pts_all[np.abs(bckg_pts_all[:,0]) < 10]
bckg_pts_all = bckg_pts_all[np.abs(bckg_pts_all[:,1]) < 50]
bckg_pts_all = bckg_pts_all[np.abs(bckg_pts_all[:,2]) < 50]

histogram, bin_edges = np.histogram(bckg_pts_all[:,2], bins=512, range=(-100, 100))

plt.figure()
plt.title("Image Difference Histogram, #1")
plt.xlabel("Intensity")
plt.ylabel("Count")
plt.ylim([0, 100])  # <- named arguments do not work here
plt.plot(bin_edges[0:-1], histogram)  # <- or here
plt.show()
# create the histogram, plot #2

x_mean, x_std = np.mean(bckg_pts_all[:, 0]), np.std(bckg_pts_all[:, 0])
y_mean, y_std = np.mean(bckg_pts_all[:, 1]), np.std(bckg_pts_all[:, 1])
z_mean, z_std = np.mean(bckg_pts_all[:, 2]), np.std(bckg_pts_all[:, 2])

# bckg_pts_all[:, 0] = (bckg_pts_all[:, 0] - x_mean) / x_std
# bckg_pts_all[:, 1] = (bckg_pts_all[:, 1] - y_mean) / y_std
# bckg_pts_all[:, 2] = (bckg_pts_all[:, 2] - z_mean) / z_std

print(x_mean, x_std)
print(y_mean, y_std)
print(z_mean, z_std)
## Generate synthetic data
N, D = 1000, 3  # number of points and dimenstinality
#
# if D == 2:
#     #set gaussian ceters and covariances in 2D
#     means = np.array([[0.5, 0.0],
#                       [0, 0],
#                       [-0.5, -0.5],
#                       [-0.8, 0.3]])
#     covs = np.array([np.diag([0.01, 0.01]),
#                      np.diag([0.025, 0.01]),
#                      np.diag([0.01, 0.025]),
#                      np.diag([0.01, 0.01])])
# elif D == 3:
#     # set gaussian ceters and covariances in 3D
#     means = np.array([[0.5, 0.0, 0.0],
#                       [0.0, 0.0, 0.0],
#                       [-0.5, -0.5, -0.5],
#                       [-0.8, 0.3, 0.4]])
#     covs = np.array([np.diag([0.01, 0.01, 0.03]),
#                      np.diag([0.08, 0.01, 0.01]),
#                      np.diag([0.01, 0.05, 0.01]),
#                      np.diag([0.03, 0.07, 0.01])])
#
n_gaussians = 1  # means.shape[0]
#
# points = []
# for i in range(len(means)):
#     x = np.random.multivariate_normal(means[i], covs[i], N)
#     points.append(x)
# points = np.concatenate(points)
points = bckg_pts_all.copy()
print(points.shape)

#fit the gaussian model
gmm = BayesianGaussianMixture(n_components=n_gaussians, covariance_type='diag', weight_concentration_prior=1,
                              weight_concentration_prior_type='dirichlet_process')  # 'diag'
gmm.fit(points)

print(gmm.covariances_)
print(gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T)
#visualize
if D == 2:
    visualize_2D_gmm(points, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T)
elif D == 3:
    visualize_3d_gmm(points, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T)


