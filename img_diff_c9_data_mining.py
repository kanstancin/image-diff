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
    assert len(pts.shape) == 3, "poly shape should be 3"
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
    for c in cnts:
        if len(c) < 100:
            continue
        print(len(c))
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return im


def plt_show_img(im, title="", cmap=None, to_rgb=False):
    if (len(im.shape) > 2) and to_rgb:
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    plt.title(title)
    plt.imshow(im, cmap=cmap)
    plt.show()


arr_name = f"G10-Z50-D500-1"
bckg_pts_all = np.load(f"/home/cstar/workspace/grid-data/diff-data-arr/bckg_pts_dataset-{arr_name}.npy")[::25]
frg_pts_all = np.load(f"/home/cstar/workspace/grid-data/diff-data-arr/frg_pts_dataset-{arr_name}.npy")[::5]
# filter zeros
frg_pts_all = frg_pts_all[np.any(frg_pts_all, axis=1)]

print(f"len bckg {len(bckg_pts_all)} \nlen frg: {len(frg_pts_all)}")
c, r = gauss_mixture.get_ellipse(bckg_pts_all, frg_pts_all)
print(c,r,np.std(bckg_pts_all, axis=0))
# cls_res_svm = gauss_mixture.one_class_svm(bckg_pts_all, frg_pts_all)
# cl_1 = bckg_pts_all[cls_res_svm==-1]
# cl_2 = bckg_pts_all[cls_res_svm==1]
# print("cl1:", len(cl_1), len(cl_2))
# gauss_mixture.visualize_3d_gmm(cl_2, cl_1, [1], c.reshape(-1,3).T, r.reshape(-1,3).T)

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

i1 = 0
for i2 in range(len(inp_img_spag_paths)):
    # if i2 != 6: continue
    # im1_path = os.path.join(inp_path_nospag, inp_img_nospag_paths[i1])
    im2_path = os.path.join(inp_path_spag, inp_img_spag_paths[i2])
    # if inp_img_nospag_paths[i1] == inp_img_spag_paths[i2]:
    #     print("skipped")
    #     continue

    mask_img_name = f"mask{inp_img_spag_paths[i2]}"
    im2_path_mask = os.path.join(inp_path_mask, mask_img_name)
    print(f"input imgs: \n\t {inp_img_nospag_paths[i1]} \n\t {inp_img_spag_paths[i2]}")

    # im1 = cv.imread(im1_path, 1)
    im1 = img_noshadows_avg.copy()
    im2 = cv.imread(im2_path, 1)  # img_X43.50_Y345.00_Z50.00.jpg #img_X177.50_Y144.00_Z50.00
    im2_mask = cv.imread(im2_path_mask, 1)[:, :, 0]

    # create ROI mask
    pts_Z10 = np.array([[1093, 647], [1096, 177], [523, 90], [0, 230], [0, 240]])
    pts_Z30 = np.array([[0, 347], [569, 128], [1080, 220], [1008, 719], [683, 719], [0, 356]])
    pts_Z50 = np.array([[0, 347], [569, 128], [1080, 220], [1008, 719], [553, 719], [0, 356]])
    pts_Z70 = np.array([[0, 443], [298, 720], [1000, 720], [1085, 205], [574, 102]])
    pts = pts_Z50
    pts = pts.reshape((-1, 1, 2))
    maskAOI = getAOIMask(img_shape=im1.shape, poly_pts=pts)
    im_diff = getImDiff(im1, im2, maskAOI=maskAOI, method="saturation")

    # merge masks
    bckg1 = maskAOI == 255
    bckg2 = im2_mask == 255
    bckg_res = bckg1 + bckg2
    bckg_res = bckg_res.astype(np.uint8)

    # cl_res = classify_ellipse(im_diff[:, :, :], c, r)

    # bckg_pts = im_diff[(bckg_res != 1) * (cl_res==255)][::10]
    bckg_pts = im_diff[(bckg_res != 1)][::10]
    frg_pts = im_diff[im2_mask == 255]  # cut
    bckg_pts_all = np.append(bckg_pts_all, bckg_pts, axis=0)
    frg_pts_all = np.append(frg_pts_all, frg_pts, axis=0)

    # cl_res = classify_circle(im_diff[:, :, :], rad=10, cx=0,  cy=0, cz=0)
    cl_res = classify_ellipse(im_diff[:, :, :], c, r)
    cl_res_dil = smart_dilate(im_diff, cl_res, c, r, std_div=9, stds_num=2)

    # plt.show()
    # if i2 == 19:
    # debug
    # gauss_mixture.visualize_pts(bckg_pts, frg_pts, c, r)

    #
    # histogram, bin_edges = np.histogram(density)#
    # plt.figure()
    # plt.title("density hist")
    # plt.xlabel("Intensity")
    # plt.ylabel("Count")
    # # plt.ylim([0, 100])  # <- named arguments do not work here
    # plt.plot(bin_edges[0:-1], histogram)  # <- or here
    # plt.show()

    im2 = draw_rect(im2, cl_res_dil)
    im2 = cv.polylines(im2, [pts], True, (255, 0, 0))

    im_dsts = get_dst_elps(im_diff, c, r)

    res_imgs = np.append(res_imgs, cv.cvtColor(im2, cv.COLOR_BGR2RGB).reshape(1, 720, 1280, 3), axis=0)
    if display:
        plt_show_img(cl_res, title="classification mask", cmap="gray", to_rgb=False)
        plt_show_img(cl_res_dil, title="classification mask _dil", cmap="gray", to_rgb=False)
        plt_show_img(im2, title=f"detection result, im2\n {i2}", cmap=None, to_rgb=True)
        fig, ax = plt.subplots(2, 2)
        im_diff[cl_res != 255] = 255
        ax[0, 0].imshow(im_diff, vmin=-50, vmax=50)
        ax[0, 1].imshow(im_diff[:, :, 0], cmap='gray')
        ax[1, 0].imshow(im_diff[:, :, 1], cmap='gray')
        ax[1, 1].imshow(im_diff[:, :, 2], cmap='gray')
        plt.show()



# bckg_pts_all_dict = {"data": bckg_pts_all, "label": "experiment"}
# savemat(f"/home/cstar/workspace/grid-data/diff-data-arr/bckg_pts_{data_dir}.mat", bckg_pts_all_dict)
# np.save(f"/home/cstar/workspace/grid-data/diff-data-arr/bckg_pts_{data_dir}.npy", bckg_pts_all)
#
# frg_pts_all_dict = {"data": frg_pts_all, "label": "experiment"}
# savemat(f"/home/cstar/workspace/grid-data/diff-data-arr/frg_pts_{data_dir}.mat", frg_pts_all_dict)
# np.save(f"/home/cstar/workspace/grid-data/diff-data-arr/frg_pts_{data_dir}.npy", frg_pts_all)

# plots ######################################################################
# print(res_imgs.shape)
# fig, ax = plt.subplots(3, 7)
# for i in range(3):
#     for j in range(7):
#         ax[i, j].imshow(res_imgs[i * j + j])
# plt.show()

res_imgs = res_imgs[:]
from mpl_toolkits.axes_grid1 import ImageGrid

for _ in range(1):
    row = 7
    col = 4
    images     = []

    fig = plt.figure(figsize=(row*15, col*15))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(col, row),  # creates 2x2 grid of axes
                     axes_pad=0.05,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, res_imgs):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig('det_res_grid.png')
    plt.show()
