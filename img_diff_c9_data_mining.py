import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os
import glob
from scipy import stats


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


def getImDiffMask(im1, im2, maskAOI=None, method="saturation"):
    im1_hsv = prepropImg(im1, kernel=(3, 3))
    im2_hsv = prepropImg(im2, kernel=(3, 3))

    h_diff = im2_hsv[:, :, 0] - im1_hsv[:, :, 0]
    h_diff = normalize_img3(h_diff)

    s_diff = im2_hsv[:, :, 1] - im1_hsv[:, :, 1]
    s_diff = normalize_img3(s_diff)

    v_diff = im2_hsv[:, :, 2] - im1_hsv[:, :, 2]
    v_diff = normalize_img3(v_diff)

    # to process
    mask_diff = im1_hsv - im2_hsv  # s_diff.copy()  # s_diff best
    print("base", mask_diff.dtype)
    vmax = 100
    # apply ROI mask
    mask_diff[maskAOI == 255] = [0, 0, 0]

    im = mask_diff[:, :, 1]
    im = np.abs(im)
    # fig, ax = plt.subplots(2, 2)
    # fig.suptitle('Im2, HSV diff plots', fontsize=16)
    # ax[0, 0].imshow(cv.cvtColor(im2, cv.COLOR_BGR2RGB))
    # ax[0, 1].imshow(np.abs(mask_diff[:,:,0]), cmap='gray', vmin=0, vmax=30)
    # ax[1, 0].imshow(np.abs(mask_diff[:,:,1]), cmap='gray', vmin=0, vmax=30)
    # ax[1, 1].imshow(np.abs(mask_diff[:,:,2]), cmap='gray', vmin=0, vmax=30)
    # plt.show()

    # print("here", np.amin(im), np.mean(im))
    # plt.title("here")
    # plt.imshow(im.astype(np.uint8), cmap='gray')
    # plt.show()

    # create the histogram, plot #1
    # histogram, bin_edges = np.histogram(base_ch, bins=256, range=(0, 255))


    # threshold diff
    # threshold = 10
    # mask_diff = np.zeros(base_ch.shape)
    # mask_diff[base_ch > threshold] = 255
    # print(mask_diff.shape)
    # mask_diff = mask_diff.astype("uint8")
    # kernel = np.ones((5, 5), np.uint8)
    # # mask_diff = cv.erode(mask_diff, kernel, iterations=1)

    return mask_diff


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


inp_path_spag = '/home/cstar/workspace/grid-data/dataset-im-diff-no-shadows-spag-Z30-avg-3/images/train/'
inp_img_spag_paths = get_img_paths(inp_path_spag)
inp_path_mask = '/home/cstar/workspace/grid-data/dataset-im-diff-no-shadows-spag-Z30-avg-3/masks/train/'
inp_path_nospag = '/home/cstar/workspace/grid-data/dataset-im-diff-no-shadows-no-spag-Z30-avg-3/'
inp_img_nospag_paths = get_img_paths(inp_path_nospag)
display = False
bckg_pts_all = np.empty((0, 3))
frg_pts_all = np.empty((0, 3))

img_noshadows_avg = np.zeros((720, 1280, 3)).astype(np.uint64)
for i1 in range(len(inp_img_nospag_paths)):
    im1_path = os.path.join(inp_path_nospag, inp_img_nospag_paths[i1])
    im1 = cv.imread(im1_path, 1)
    img_noshadows_avg += im1
    print(i1)

img_noshadows_avg = img_noshadows_avg / len(inp_img_nospag_paths)
img_noshadows_avg = img_noshadows_avg.astype(np.uint8)
cv.imwrite('/home/cstar/workspace/grid-data/img_avg.png', img_noshadows_avg)
# i1, i2 = (0, 20)
i1 = 0
for i2 in range(len(inp_img_spag_paths)):
    im1_path = os.path.join(inp_path_nospag, inp_img_nospag_paths[i1])
    im2_path = os.path.join(inp_path_spag, inp_img_spag_paths[i2])
    if inp_img_nospag_paths[i1] == inp_img_spag_paths[i2]:
        print("skipped")
        continue

    mask_img_name = f"mask{inp_img_spag_paths[i2]}"
    im2_path_mask = os.path.join(inp_path_mask, mask_img_name)
    print(f"input imgs: \n\t {inp_img_nospag_paths[i1]} \n\t {inp_img_spag_paths[i2]}")

    im1 = cv.imread(im1_path, 1)
    im1 = img_noshadows_avg.copy()
    im2 = cv.imread(im2_path, 1)  # img_X43.50_Y345.00_Z50.00.jpg #img_X177.50_Y144.00_Z50.00
    im2_mask = cv.imread(im2_path_mask, 1)[:, :, 0]

    # TODO: func img, img, maskAOI -> mask of diff [x]

    # create ROI mask
    pts = np.array([[0, 347], [504, 166], [1075, 261], [1008, 719], [683, 719], [0, 356]])
    pts = pts.reshape((-1, 1, 2))
    maskAOI = getAOIMask(img_shape=im1.shape, poly_pts=pts)

    mask_diff = getImDiffMask(im1, im2, maskAOI=maskAOI, method="saturation")


    # diff_points_arr = mask_diff[mask_diff != [0,0,0]]
    # print(diff_points_arr.shape)
    bckg1 = maskAOI == 255
    bckg2 = im2_mask == 255
    bckg_res = bckg1 + bckg2
    bckg_res = bckg_res.astype(np.uint8)
    # plt.imshow(bckg_res, cmap='gray')
    # plt.show()
    bckg_pts = mask_diff[bckg_res != 1][::10]
    frg_pts = mask_diff[im2_mask == 255]  # cut
    bckg_pts_all = np.append(bckg_pts_all, bckg_pts, axis=0)
    frg_pts_all = np.append(frg_pts_all, frg_pts, axis=0)

    cl_res = classify_circle(mask_diff[:, :, :], rad=10, cx=0,  cy=0, cz=0)

    ROI_number = 0
    cnts = cv.findContours(cl_res, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        if len(c) < 100:
            continue
        print(len(c))
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(im2, (x, y), (x + w, y + h), (0, 0, 255), 2)

    plt.title("classification")
    plt.imshow(cl_res, cmap='gray')
    plt.show()
    plt.title("input im2 with detections")
    plt.imshow(cv.cvtColor(im2, cv.COLOR_BGR2RGB))
    plt.show()


# np.save("/home/cstar/workspace/grid-data/bckg_pts_all.npy", bckg_pts_all)
# np.save("/home/cstar/workspace/grid-data/frg_pts_all.npy", frg_pts_all)


from scipy.io import savemat

bckg_pts_all_dict = {"data": bckg_pts_all, "label": "experiment"}
savemat("/home/cstar/workspace/grid-data/bckg_pts_all.mat", bckg_pts_all_dict)

frg_pts_all_dict = {"data": frg_pts_all, "label": "experiment"}
savemat("/home/cstar/workspace/grid-data/frg_pts_all.mat", frg_pts_all_dict)

# t_arr = t_arr*255
# t_arr = t_arr.astype("uint8")
# plt.imshow(a_res, cmap='gray')
# plt.show()