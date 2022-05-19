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
    img = cv.GaussianBlur(img, kernel, cv.BORDER_DEFAULT).astype(np.int32)
    return img


def getImDiffMask(im1, im2, maskAOI=None, method="saturation"):
    im1_hsv = prepropImg(im1, kernel=(7, 7))
    im2_hsv = prepropImg(im2, kernel=(7, 7))

    h_diff = im2_hsv[:, :, 0] - im1_hsv[:, :, 0]
    h_diff = normalize_img3(h_diff)

    s_diff = im2_hsv[:, :, 1] - im1_hsv[:, :, 1]
    s_diff = normalize_img3(s_diff)

    v_diff = im2_hsv[:, :, 2] - im1_hsv[:, :, 2]
    v_diff = normalize_img3(v_diff)

    # to process
    base_ch = s_diff.copy()  # s_diff best

    vmax = 100
    if display:
        plt.title("base_ch before maskAOI")
        plt.imshow(base_ch, cmap='gray', vmin=0, vmax=vmax)
        plt.show()
    # apply ROI mask
    base_ch[maskAOI == 255] = 0
    if display:
        plt.title("base_ch after maskAOI")
        plt.imshow(base_ch, cmap='gray', vmin=0, vmax=vmax)
        plt.show()

        # plot 2 selected images
        # plot hsv
        fig, ax = plt.subplots(1, 2)
        fig.suptitle('Selected images: im1, im2', fontsize=16)
        ax[0].imshow(cv.cvtColor(im1, cv.COLOR_BGR2RGB))
        ax[1].imshow(cv.cvtColor(im2, cv.COLOR_BGR2RGB))
        plt.show()

        # plot hsv
        fig, ax = plt.subplots(2, 2)
        fig.suptitle('Im2, HSV plots', fontsize=16)
        ax[0, 0].imshow(cv.cvtColor(im2, cv.COLOR_BGR2RGB))
        ax[0, 1].imshow(im2_hsv[:, :, 0], cmap='gray')
        ax[1, 0].imshow(im2_hsv[:, :, 1], cmap='gray')
        ax[1, 1].imshow(im2_hsv[:, :, 2], cmap='gray')
        plt.show()

        # plot hsv diff
        fig, ax = plt.subplots(2, 2)
        fig.suptitle('Im2, HSV diff plots', fontsize=16)
        ax[0, 0].imshow(cv.cvtColor(im2, cv.COLOR_BGR2RGB))
        ax[0, 1].imshow(h_diff, cmap='gray')
        ax[1, 0].imshow(s_diff, cmap='gray')
        ax[1, 1].imshow(v_diff, cmap='gray')
        plt.show()

    # create the histogram, plot #1
    histogram, bin_edges = np.histogram(base_ch, bins=256, range=(0, 255))
    if display:
        plt.figure()
        plt.title("Image Difference Histogram, #1")
        plt.xlabel("Intensity")
        plt.ylabel("Count")
        plt.ylim([0, 5000])  # <- named arguments do not work here
        plt.plot(bin_edges[0:-1], histogram)  # <- or here
        plt.show()
        # create the histogram, plot #2
        plt.ylim([0, 5000])
        plt.hist(base_ch.reshape(-1), bins=256, range=(0, 255))
        plt.title("hist #2")
        plt.show()

    # threshold diff
    threshold = 10
    mask_diff = np.zeros(base_ch.shape)
    mask_diff[base_ch > threshold] = 255
    print(mask_diff.shape)
    mask_diff = mask_diff.astype("uint8")
    kernel = np.ones((5, 5), np.uint8)
    # mask_diff = cv.erode(mask_diff, kernel, iterations=1)
    if display:
        plt.title("mask value after threshold")
        plt.imshow(mask_diff, cmap='gray')
        plt.show()

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


inp_path_spag = '/home/cstar/workspace/grid-data/dataset-im-diff-no-shadows-Z30/images/train/'
inp_img_spag_paths = get_img_paths(inp_path_spag)
inp_path_nospag = '/home/cstar/workspace/grid-data/im-test-no-shadow/'
inp_img_nospag_paths = get_img_paths(inp_path_nospag)
display = True
i1, i2 = (0, 8)
im1_path = os.path.join(inp_path_nospag, inp_img_nospag_paths[i1])
im2_path = os.path.join(inp_path_spag, inp_img_spag_paths[i2])
print(f"input imgs: \n\t {inp_img_nospag_paths[i1]} \n\t {inp_img_spag_paths[i2]}")


im1 = cv.imread(os.path.join(inp_path_nospag, im1_path), 1)
im2 = cv.imread(os.path.join(inp_path_spag, im2_path),
                  1)  # img_X43.50_Y345.00_Z50.00.jpg #img_X177.50_Y144.00_Z50.00

# TODO: func img, img, maskAOI -> mask of diff

# create ROI mask
pts = np.array([[0, 347], [504, 166], [1075, 261], [1008, 719], [683, 719], [0, 356]])
pts = pts.reshape((-1, 1, 2))
maskAOI = getAOIMask(img_shape=im1.shape, poly_pts=pts)
if display:
    plt.imshow(maskAOI)
    plt.title("maskAOI")
    plt.show()

# resize
# width = 1000
# height = int(width / v_diff.shape[1] * v_diff.shape[0])
# print(np.amin(v_diff), np.amax(v_diff))
# print(v_diff.shape)
# shape = (width, height)
# v_diff = cv.resize(v_diff, shape, interpolation=cv.INTER_AREA)
# img = cv.resize(img_2, shape, interpolation=cv.INTER_AREA)
# mask = cv.resize(mask, shape, interpolation=cv.INTER_NEAREST)

# plot difference
# fig, ax = plt.subplots(2, 2)
# ax[0, 0].imshow(img_2)
# ax[0, 1].imshow(h_diff, cmap='gray')
# ax[1, 0].imshow(s_diff, cmap='gray')
# ax[1, 1].imshow(v_diff, cmap='gray')
# plt.show()

mask_diff = getImDiffMask(im1, im2, maskAOI=maskAOI, method="saturation")

ROI_number = 0
cnts = cv.findContours(mask_diff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    if len(c) < 100:
        continue
    print(len(c))
    x, y, w, h = cv.boundingRect(c)
    cv.rectangle(im2, (x, y), (x + w, y + h), (0, 0, 255), 2)

if display:
    plt.title("input im2 with detections")
    plt.imshow(cv.cvtColor(im2, cv.COLOR_BGR2RGB))
    plt.show()
