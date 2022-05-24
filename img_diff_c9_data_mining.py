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
    cnts = cv.findContours(cl_res, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
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


arr_name = f"G10-Z50-D500-0"
bckg_pts_all = np.load(f"/home/cstar/workspace/grid-data/diff-data-arr/bckg_pts_dataset-{arr_name}.npy")[::50]
frg_pts_all = np.load(f"/home/cstar/workspace/grid-data/diff-data-arr/frg_pts_dataset-{arr_name}.npy")[::10]
# filter zeros
frg_pts_all = frg_pts_all[np.any(frg_pts_all, axis=1)]

print(f"len bckg {len(bckg_pts_all)} \nlen frg: {len(frg_pts_all)}")
c, r = gauss_mixture.get_ellipse(bckg_pts_all, frg_pts_all)

data_dir = 'dataset-G10-Z50-D500-0'
data_spag_path = f'/home/cstar/workspace/grid-data/preproc_data/{data_dir}/dataset-im-diff-spag-avg-3/'

inp_path_spag = os.path.join(data_spag_path, 'images/train/')
inp_img_spag_paths = get_img_paths(inp_path_spag)
inp_path_mask = os.path.join(data_spag_path, 'masks/train/')

inp_path_nospag = f'/home/cstar/workspace/grid-data/preproc_data/{data_dir}/no-shadow/'  # should be not avg-3
inp_img_nospag_paths = get_img_paths(inp_path_nospag)
display = True
bckg_pts_all = np.empty((0, 3))
frg_pts_all = np.empty((0, 3))

res_imgs = np.empty((0, 720, 1280, 3)).astype(np.uint8)
img_noshadows_avg = avg_img_from_path(inp_img_nospag_paths)

i1 = 0
for i2 in range(len(inp_img_spag_paths)):
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
    pts = pts_Z10
    pts = pts.reshape((-1, 1, 2))
    maskAOI = getAOIMask(img_shape=im1.shape, poly_pts=pts)

    im_diff = getImDiff(im1, im2, maskAOI=maskAOI, method="saturation")

    # merge masks
    bckg1 = maskAOI == 255
    bckg2 = im2_mask == 255
    bckg_res = bckg1 + bckg2
    bckg_res = bckg_res.astype(np.uint8)

    bckg_pts = im_diff[bckg_res != 1][::10]
    frg_pts = im_diff[im2_mask == 255]  # cut
    bckg_pts_all = np.append(bckg_pts_all, bckg_pts, axis=0)
    frg_pts_all = np.append(frg_pts_all, frg_pts, axis=0)

    # cl_res = classify_circle(im_diff[:, :, :], rad=10, cx=0,  cy=0, cz=0)
    cl_res = classify_ellipse(im_diff[:, :, :], c, r)

    im2 = draw_rect(im2, cl_res)
    im2 = cv.polylines(im2, [pts], True, (255,0,0))

    res_imgs = np.append(res_imgs, cv.cvtColor(im2, cv.COLOR_BGR2RGB).reshape(1, 720, 1280, 3), axis=0)
    if display:
        plt_show_img(cl_res, title="classification mask", cmap="gray", to_rgb=False)
        plt_show_img(im2, title=f"detection result, im2\n {i2}", cmap=None, to_rgb=True)



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

res_imgs = res_imgs[:-1]
from mpl_toolkits.axes_grid1 import ImageGrid

for _ in range(1):
    row = 7
    col = 3
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
