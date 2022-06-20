import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os
from scipy import stats
from scipy.io import savemat
import gauss_mixture

def normalize_img(img):
    assert (len(img.shape) == 2)
    img = img - np.min(img)
    img = ((img / np.max(img)) * 255).astype(np.uint8)
    return img


def normalize_img2(inp):
    assert (len(inp.shape) == 2)
    print(np.min(inp), np.max(inp))
    mean, _ = stats.mode(inp, axis=None)
    print(mean)
    inp = inp - mean + 256 / 2
    inp[inp < 0] = 0
    inp[inp > 255] = 255
    inp = inp.astype(np.uint8)
    return inp


def normalize_img3(img):
    assert len(img.shape) == 2, "only single-ch imgs are supported"
    img = np.abs(img)
    img = img.astype(np.uint8)
    # img = ((img / np.max(img)) * 255).astype(np.uint8)
    return img


def getAOIMask(img_shape, poly_pts):
    assert len(poly_pts.shape) == 3, "poly shape should be 3"
    mask = np.zeros((img_shape[0:2]), dtype=np.uint8)
    mask2 = np.zeros((img_shape[0] + 2, img_shape[1] + 2), dtype=np.uint8)
    cv.polylines(mask, [poly_pts], True, 255)
    fill_pt = (int(img_shape[1]/2), int(img_shape[0]/2))
    _, mask, _, _ = cv.floodFill(mask, mask2, fill_pt, 255)
    mask = cv.bitwise_not(mask)
    mask[mask == 1] = 255
    print(mask.dtype)
    return mask


def prepropImg(img, kernel=(15, 15)):
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img = cv.GaussianBlur(img, kernel, cv.BORDER_DEFAULT).astype(np.int16)
    return img


def getImDiff(im1, im2, maskAOI=None, method="saturation"):
    im1_hsv = prepropImg(im1, kernel=(3, 3))
    im2_hsv = prepropImg(im2, kernel=(3, 3))

    # to process
    im_diff = im1_hsv - im2_hsv  # s_diff.copy()  # s_diff best
    # apply ROI mask
    im_diff[maskAOI == 255] = [0, 0, 0]

    return im_diff


def is_inp(img_name):
    return img_name.endswith(("jpg", "jpeg", "png"))


def get_img_paths(im_in_dir, verbose=True):
    all_inps = os.listdir(im_in_dir)
    all_inps = [name for name in all_inps if is_inp(name)]
    if verbose:
        print('*'*33)
        print("Found the following {} images:".format(len(all_inps)))
        for im_path in all_inps:
            print(im_path)
        print('*'*33)
    return all_inps


def classify_circle(img, rad=10, cx=0,  cy=0, cz=0):
    img = img.astype(np.float16)
    ch1 = img[:, :, 0] - cx
    ch1 = ch1 ** 2
    ch2 = img[:, :, 1] - cy
    ch2 = ch2 ** 2
    ch3 = img[:, :, 1] - cz
    ch3 = ch3 ** 2
    val_ch = ch1 + ch2 + ch3
    res = np.zeros(img.shape[:2]).astype(np.uint8)
    res[val_ch > rad**2] = 255
    return res


def classify_ellipse(img, c, r):
    img = img.astype(np.float16)
    ch1 = ((img[:, :, 0] - c[0]) ** 2) / r[0] ** 2
    ch2 = ((img[:, :, 1] - c[1]) ** 2) / r[1] ** 2
    ch3 = ((img[:, :, 2] - c[2]) ** 2) / r[2] ** 2
    val_ch = ch1 + ch2 + ch3
    res = np.zeros(img.shape[:2]).astype(np.uint8)
    res[val_ch > 1] = 255
    return res

def get_dst_elps(im_diff, c, r, std_div=12, stds_num=2):
    orig_shape = im_diff.shape[:2]
    pts = im_diff.reshape(-1, 3).astype(np.float32)
    pts_outside = ((pts[:, 0] - c[0])**2 / (r[0]**2) + (pts[:, 1] - c[1])**2 / (r[1]**2) +
                         (pts[:, 2] - c[2])**2 / (r[2]**2)) > 1
    pts_inside = np.logical_not(pts_outside)
    pts_std = pts.copy()
    means = np.mean(pts, axis=0)
    stds = np.std(pts, axis=0)
    means = c
    stds = r/std_div
    pts_std[:, 0] = (pts[:, 0] - means[0]) / stds[0]
    pts_std[:, 1] = (pts[:, 1] - means[1]) / stds[1]
    pts_std[:, 2] = (pts[:, 2] - means[2]) / stds[2]
    dsts = np.sqrt(pts_std[:, 0]**2 + pts_std[:, 1]**2 + pts_std[:, 2]**2) - 1
    dsts_img = dsts.reshape(orig_shape)

    dsts_inside = dsts[pts_inside]
    mean_dst_inside = np.mean(dsts_inside)
    std_dst_inside = np.std(dsts_inside)
    dst_lower_thresh = mean_dst_inside + stds_num * std_dst_inside
    dsts_img[dsts_img < dst_lower_thresh] = 0
    dsts_img[dsts_img >= dst_lower_thresh] = 255
    return dsts_img

def smart_dilate(im_diff, cl_res, c, r, std_div=12, stds_num=2):
    kernel = np.ones((25, 25), np.uint8)
    cl_res = cv.dilate(cl_res, kernel, iterations=1)
    kernel = np.ones((15, 15), np.uint8)
    cl_res = cv.erode(cl_res, kernel, iterations=1)
    # get distances to ellipsoid
    im_dsts = get_dst_elps(im_diff, c, r, std_div=std_div, stds_num=stds_num)
    dil_res = np.logical_and(cl_res, im_dsts).astype(np.uint8) * 255
    # debug
    # plt.title("cl_res")
    # plt.imshow(cl_res, cmap="gray")
    # plt.show()
    # plt.title("dst elps thresholded")
    # plt.imshow(im_dsts, cmap="gray", vmin=np.amin(im_dsts), vmax=np.amax(im_dsts))
    # plt.show()
    # plt.title("dil_res final")
    # plt.imshow(dil_res, cmap="gray", vmin=np.amin(dil_res), vmax=np.amax(dil_res))
    # plt.show()
    return dil_res


def avg_img_from_path(inp_img_nospag_paths):
    img_noshadows_avg = np.zeros((720, 1280, 3)).astype(np.uint64)
    for i1 in range(len(inp_img_nospag_paths)):
        im1_path = os.path.join(inp_path_nospag, inp_img_nospag_paths[i1])
        im1 = cv.imread(im1_path, 1)
        img_noshadows_avg += im1
        print(i1)
    img_noshadows_avg = img_noshadows_avg / len(inp_img_nospag_paths)
    img_noshadows_avg = img_noshadows_avg.astype(np.uint8)
    cv.imwrite('/home/cstar/workspace/grid-data/img_avg.png', img_noshadows_avg)
    return img_noshadows_avg


def draw_rect(im, cl_res):
    ROI_number = 0
    cnts = cv.findContours(cl_res, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    found_cntr = False
    for c in cnts:
        if len(c) < 100:
            continue
        print(len(c))
        found_cntr = True
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return im, found_cntr


def plt_show_img(im, title="", cmap=None, to_rgb=False):
    if (len(im.shape) > 2) and to_rgb:
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    plt.title(title)
    plt.imshow(im, cmap=cmap)
    plt.show()


def avg_imgs(arr):
    img_avg = np.zeros((720, 1280, 3)).astype(np.uint64)
    for i1 in range(arr.shape[0]):
        im1 = arr[i1]
        img_avg += im1
    img_avg = img_avg / arr.shape[0]
    img_avg = img_avg.astype(np.uint8)
    return img_avg


# buffer with two sets of imgs on input, each has 10 imgs
data_dir = 'dataset-G10-Z50-D500-1'
data_spag_path = f'/home/cstar/workspace/grid-data/preproc_data/{data_dir}/dataset-im-diff-spag-avg-3/'

inp_path_spag = os.path.join(data_spag_path, 'images/train/')
inp_img_spag_paths = get_img_paths(inp_path_spag)
inp_path_mask = os.path.join(data_spag_path, 'masks/train/')

inp_path_nospag = f'/home/cstar/workspace/grid-data/preproc_data/{data_dir}/no-shadow/'  # should be not avg-3
inp_img_nospag_paths = get_img_paths(inp_path_nospag)
display = False
bckg_pts_all = np.empty((0, 3))
frg_pts_all = np.empty((0, 3))

res_imgs = np.empty((0, 720, 1280, 3)).astype(np.uint8)
img_noshadows_avg = avg_img_from_path(inp_img_nospag_paths)

# create buffer for test
buffer = np.zeros((2, 10, 720, 1280, 3)).astype(np.uint8)
for i in range(10):
    im1_path = os.path.join(inp_path_nospag, inp_img_nospag_paths[i])
    im1 = cv.imread(im1_path, 1)
    buffer[0, i] = im1.copy()

for i in range(10):
    im1_path = os.path.join(inp_path_nospag, inp_img_nospag_paths[i+10])
    im1 = cv.imread(im1_path, 1)
    buffer[1, i] = im1.copy()

# takes buffer of size 2 x 10 on input
def get_det_res(buffer, show=False):
    # avg of prev img
    im_avg1 = avg_imgs(buffer[0, :5])
    im_avg2 = avg_imgs(buffer[0, 5:])

    # create mask
    # create ROI mask
    pts_Z10 = np.array([[1093, 647], [1096, 177], [523, 90], [0, 230], [0, 240]])
    pts_Z30 = np.array([[0, 347], [569, 128], [1080, 220], [1008, 719], [683, 719], [0, 356]])
    pts_Z50 = np.array([[0, 347], [569, 128], [1080, 220], [1008, 719], [553, 719], [0, 356]])
    pts_Z70 = np.array([[0, 443], [298, 720], [1000, 720], [1085, 205], [574, 102]])
    pts = pts_Z50
    pts = pts.reshape((-1, 1, 2))
    maskAOI = getAOIMask(img_shape=im_avg1.shape, poly_pts=pts)
    im_diff = getImDiff(im_avg1, im_avg2, maskAOI=maskAOI, method="saturation")
    print('here1', im_diff.reshape(-1, 3).shape)
    # find ellipse
    c, r = gauss_mixture.get_ellipse_no_frg(im_diff.reshape(-1, 3)[::50].astype(np.int16), show=False)
    print("Found ellipse: ", c, r)
    print('here1')
    # classify and get result
    im_avg1 = avg_imgs(buffer[0, :5])
    im_avg2 = avg_imgs(buffer[1, :5])
    im_diff = getImDiff(im_avg1, im_avg2, maskAOI=maskAOI, method="saturation")

    # cl_res = classify_circle(im_diff[:, :, :], rad=10, cx=0,  cy=0, cz=0)
    cl_res = classify_ellipse(im_diff[:, :, :], c, r)
    cl_res_dil = smart_dilate(im_diff, cl_res, c, r, std_div=9, stds_num=2)

    # gauss_mixture.visualize_pts(im_diff.reshape(-1, 3)[::50], im_diff.reshape(-1, 3)[::50], c, r)

    im2, found_cntr = draw_rect(im_avg2, cl_res_dil)
    im2 = cv.polylines(im2, [pts], True, (255, 0, 0))

    if show:
        plt_show_img(im2, title=f"detection result, im2", cmap=None, to_rgb=True)

    return im2, found_cntr

im_det, found_cntr = get_det_res(buffer, show=True)
