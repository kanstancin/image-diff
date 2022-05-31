import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import os
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


def visualize_3d_gmm(points, points2, w, mu, stdev, export=True):
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
    n = 40
    axes.set_xlim(np.array([-1, 1])*n)
    axes.set_ylim(np.array([-1, 1])*n)
    axes.set_zlim(np.array([-1, 1])*n)
    plt.set_cmap('Set1')
    colors = cmx.Set1([0, 1])
    for i in range(1):
        idx = range(i * N, (i + 1) * N)
        axes.scatter(points[idx, 0], points[idx, 1], points[idx, 2], alpha=0.3, c=colors[1])
        plot_sphere(w=w[i], c=mu[:, i], r=stdev[:, i], ax=axes)

    N = int(np.round(points2.shape[0] / n_gaussians))
    for i in range(1):
        idx = range(i * N, (i + 1) * N)
        axes.scatter(points2[idx, 0], points2[idx, 1], points2[idx, 2], alpha=0.3, c=colors[0])

    plt.title('3D GMM')
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.view_init(35.246, 45)
    if export:
        if not os.path.exists('images/'): os.mkdir('images/')
        plt.savefig('images/3D_GMM_demonstration.png', dpi=100, format='png')
    plt.show()


def plot_sphere(w=0, c=[0,0,0], r=[1, 1, 1], subdev=10, ax=None, sigma_multiplier=1):
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


def remove_noise_gauss(pts, std_iters=2, std_range=2, axis=0):
    for i in range(std_iters):
        mean, std = np.mean(pts[:, axis]), np.std(pts[:, axis])
        print("thresh: ", std * std_range)
        pts = pts[np.abs((pts[:, axis] - mean)) < std_range * std]
    return pts


from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors
def dst_classification(bckg_pts_all):
    # dist = DistanceMetric.get_metric('euclidean')
    # kde = KernelDensity(kernel='gaussian', bandwidth=5).fit(bckg_pts_all)
    # kde = kde.score_samples(bckg_pts_all)
    neigh = NearestNeighbors(n_neighbors=5)
    nei = neigh.fit(bckg_pts_all)
    kde = neigh.kneighbors(bckg_pts_all)
    print(kde)
    means = np.amax(kde[0], axis=1)
    print("std", np.std(means-np.mean(means)))
    histogram, bin_edges = np.histogram(means)#
    plt.figure()
    plt.title("density hist")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    # plt.ylim([0, 100])  # <- named arguments do not work here
    plt.plot(bin_edges[0:-1], histogram)  # <- or here
    plt.show()
    return means


def remove_noise_dens(pts, thresh):
    density = dst_classification(pts)
    return pts[density < thresh], pts[density >= thresh]
    # cl_dens = neigh_classifier(density, im_diff, thresh=4500)


def get_elps_pts(pts, c, r):
    arr = ((pts[:, 0] - c[0])**2 / (r[0]**2) + (pts[:, 1] - c[1])**2 / (r[1]**2) + (pts[:, 2] - c[2])**2 / (r[2]**2)) \
          < 1
    return pts[arr]


def get_ellipse(bckg_pts_all, frg_pts_all):
    bckg_pts_all_prev = bckg_pts_all.copy()
    # bckg_pts_all = remove_noise_gauss(bckg_pts_all, std_iters=1, std_range=11, axis=0)  # 11
    # bckg_pts_all = remove_noise_gauss(bckg_pts_all, std_iters=1, std_range=11, axis=1)  # 11
    # bckg_pts_all = remove_noise_gauss(bckg_pts_all, std_iters=1, std_range=25, axis=2)  # 25
    bckg_pts_all, filtered_pts = remove_noise_dens(bckg_pts_all, thresh=2.5)

    histogram, bin_edges = np.histogram(bckg_pts_all[:, 0], bins=512, range=(-100, 100))
    plt.figure()
    plt.title("Image Difference Histogram, #1")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.ylim([0, 100])  # <- named arguments do not work here
    plt.plot(bin_edges[0:-1], histogram)  # <- or here
    plt.show()


    # create the histogram, plot #2

    # x_mean, x_std = np.mean(bckg_pts_all[:, 0]), np.std(bckg_pts_all[:, 0])
    # y_mean, y_std = np.mean(bckg_pts_all[:, 1]), np.std(bckg_pts_all[:, 1])
    # z_mean, z_std = np.mean(bckg_pts_all[:, 2]), np.std(bckg_pts_all[:, 2])

    # bckg_pts_all[:, 0] = (bckg_pts_all[:, 0] - x_mean) / x_std
    # bckg_pts_all[:, 1] = (bckg_pts_all[:, 1] - y_mean) / y_std
    # bckg_pts_all[:, 2] = (bckg_pts_all[:, 2] - z_mean) / z_std

    n_gaussians = 1  # means.shape[0]
    #
    points = bckg_pts_all.copy()

    # fit the gaussian model
    gmm = BayesianGaussianMixture(n_components=n_gaussians, covariance_type='diag', weight_concentration_prior=1,
                                  weight_concentration_prior_type='dirichlet_process')  # 'diag'
    gmm.fit(points)

    c = gmm.means_.reshape(3)
    r = np.sqrt(gmm.covariances_).reshape(3) * 12
    elp_pts = get_elps_pts(bckg_pts_all_prev, c, r)

    print(f"final ellipse: \t{c}\n\t{r}")
    print(
        f"number of points reduced by {(len(bckg_pts_all_prev) - len(bckg_pts_all)) / len(bckg_pts_all_prev) * 100 :.2f}%")

    frg_pts_elp = get_elps_pts(frg_pts_all, c, r)
    print(f"{(len(frg_pts_elp)) / len(frg_pts_all) * 100 :.2f}% of frg points are in ellipse")
    print(f"{(len(elp_pts)) / len(bckg_pts_all_prev) * 100 :.2f}% of bkg points are in ellipse")
    # visualize
    visualize_3d_gmm(bckg_pts_all_prev, frg_pts_all, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T * 12)
    visualize_3d_gmm(filtered_pts, bckg_pts_all, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T * 12)

    return c, r

def get_ellipse_debug(bckg_pts_all, frg_pts_all, c, r):
    bckg_pts_all_prev = bckg_pts_all.copy()

    points = bckg_pts_all.copy()
    elp_pts = get_elps_pts(points, c, r)


    frg_pts_elp = get_elps_pts(frg_pts_all, c, r)
    print(f"{(len(frg_pts_elp)) / len(frg_pts_all) * 100 :.2f}% of frg points are in ellipse")
    print(f"{(len(elp_pts)) / len(points) * 100 :.2f}% of bkg points are in ellipse")
    # visualize
    visualize_3d_gmm(bckg_pts_all_prev, frg_pts_all, [1], c.reshape(-1, 3).T, r.reshape(-1, 3).T)

    return c, r


def visualize_pts(pts1, pts2, c, r):
    visualize_3d_gmm(pts1, pts2, [1], c.reshape(-1, 3).T, r.reshape(-1, 3).T)


from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
def one_class_svm(bckg_pts_all, frg_pts_all):
    # clf = IsolationForest(random_state=0, contamination=0.01,n_estimators=300).fit(bckg_pts_all)  # fast, works well
    clf = LocalOutlierFactor(n_neighbors=3, algorithm="brute", metric='euclidean', novelty=True, contamination="auto").fit(bckg_pts_all)
    cls = clf.predict(bckg_pts_all)
    return cls




def neigh_classifier(dens, im_pts, thresh):
    orig_shape = im_pts.shape[:2]
    im_pts = im_pts.reshape(-1, 3)
    neigh = NearestNeighbors(n_neighbors=5).fit(im_pts)
    kde = neigh.kneighbors(im_pts)
    dens = np.mean(kde[1], axis=1)
    dens_im = dens.reshape(orig_shape)
    dens_im[dens_im <= thresh] = 0
    dens_im[dens_im > thresh] = 255
    dens_im = dens_im.astype(np.uint8)
    return dens_im



# arr_name = f"G10-Z10-D500-0"
# bckg_pts_all = np.load(f"/home/cstar/workspace/grid-data/diff-data-arr/bckg_pts_dataset-{arr_name}.npy")[::50]
# frg_pts_all = np.load(f"/home/cstar/workspace/grid-data/diff-data-arr/frg_pts_dataset-{arr_name}.npy")[::10]
# # filter zeros
# frg_pts_all = frg_pts_all[np.any(frg_pts_all, axis=1)]
#
# print("bckg_pts shape: ", bckg_pts_all.shape)
# print("frg_pts shape: ", frg_pts_all.shape)
#
# # bckg_pts_all = bckg_pts_all[np.abs(bckg_pts_all[:,0]) < 10]
# # bckg_pts_all = bckg_pts_all[np.abs(bckg_pts_all[:,1]) < 50]
# # bckg_pts_all = bckg_pts_all[np.abs(bckg_pts_all[:,2]) < 50]
#
# histogram, bin_edges = np.histogram(bckg_pts_all[:, 0], bins=512, range=(-100, 100))
# plt.figure()
# plt.title("Image Difference Histogram, #1")
# plt.xlabel("Intensity")
# plt.ylabel("Count")
# plt.ylim([0, 100])  # <- named arguments do not work here
# plt.plot(bin_edges[0:-1], histogram)  # <- or here
# plt.show()
#
# c, r = get_ellipse(bckg_pts_all, frg_pts_all)
