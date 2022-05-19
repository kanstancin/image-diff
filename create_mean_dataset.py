import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os
import glob
from scipy import stats


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

DISPLAY = False
inp_path_nospag = '/home/cstar/workspace/grid-data/im-test-no-shadow/'
inp_img_nospag_paths = get_img_paths(inp_path_nospag)

num_to_average = 3
img_noshadows_avg = np.zeros((720, 1280, 3)).astype(np.uint64)
for i in range(len(inp_img_nospag_paths)):
    for j in range(num_to_average):
        im_i = (i - (num_to_average // 2) + j) % len(inp_img_nospag_paths)
        im_path = os.path.join(inp_path_nospag, inp_img_nospag_paths[im_i])
        im = cv.imread(im_path, 1)
        img_noshadows_avg += im
        print("read", im_i)
    img_noshadows_avg = img_noshadows_avg / num_to_average
    img_noshadows_avg = img_noshadows_avg.astype(np.uint8)
    print(i)
    cv.imwrite(f'/home/cstar/workspace/grid-data/dataset-im-diff-no-shadows-Z30-avg-3/{inp_img_nospag_paths[i]}',
               img_noshadows_avg)

    im_orig_path = os.path.join(inp_path_nospag, inp_img_nospag_paths[i])
    if DISPLAY:
        im_orig = cv.imread(im_orig_path, 1)
        fig, ax = plt.subplots(1, 2)
        fig.suptitle('Im blur_noblur', fontsize=16)
        ax[0].imshow(cv.cvtColor(im_orig, cv.COLOR_BGR2RGB))
        ax[1].imshow(cv.cvtColor(img_noshadows_avg, cv.COLOR_BGR2RGB))
        plt.show()
    img_noshadows_avg = np.zeros((720, 1280, 3)).astype(np.uint64)