import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os
import glob
from scipy import stats


def normalize_img(img):
    assert(len(img.shape) == 2)
    img = img - np.min(img)
    img = ((img / np.max(img)) * 255).astype(np.uint8)
    return img


def normalize_img2(inp):
    assert(len(inp.shape) == 2)
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

inp_path = 'data/'

img = cv.imread(os.path.join(inp_path, 'IMG_2485.jpg'), 1)
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
kernel = (15, 15)
img_hsv = cv.GaussianBlur(img_hsv, kernel, cv.BORDER_DEFAULT).astype(np.int32)

# fig, ax = plt.subplots(2, 2)
# ax[0, 0].imshow(img)
# ax[0, 1].imshow(img_hsv[:, :, 0], cmap='gray')
# ax[1, 0].imshow(img_hsv[:, :, 1])
# ax[1, 1].imshow(img_hsv[:, :, 2])
# plt.show()

img_2 = cv.imread(os.path.join(inp_path, 'IMG_2489.jpg'), 1)
img_hsv_2 = cv.cvtColor(img_2, cv.COLOR_BGR2HSV)
img_hsv_2 = cv.GaussianBlur(img_hsv_2, kernel, cv.BORDER_DEFAULT).astype(np.int32)

h_diff = img_hsv_2[:, :, 0] - img_hsv[:, :, 0]
h_diff = normalize_img3(h_diff)

s_diff = img_hsv_2[:, :, 1] - img_hsv[:, :, 1]
s_diff = normalize_img3(s_diff)

v_diff = img_hsv_2[:, :, 2] - img_hsv[:, :, 2]
# histogram, bin_edges = np.histogram(v_diff, bins=100)
# plt.plot(bin_edges[0:-1], histogram)  # <- or here
# plt.show()


v_diff = normalize_img3(v_diff)
width = 1000
height = int(width / v_diff.shape[1] * v_diff.shape[0])
print(np.amin(v_diff), np.amax(v_diff))
print(v_diff.shape)
shape = (width, height)
v_diff = cv.resize(v_diff, shape, interpolation=cv.INTER_AREA)
img = cv.resize(img_2, shape, interpolation=cv.INTER_AREA)
# mask = cv.resize(mask, shape, interpolation=cv.INTER_NEAREST)

# create the histogram
histogram, bin_edges = np.histogram(v_diff, bins=256, range=(0, 255))

plt.figure()
plt.title("Image Difference Histogram")
plt.xlabel("Intensity")
plt.ylabel("Count")
# plt.xlim([0.0, 1.0])  # <- named arguments do not work here
plt.plot(bin_edges[0:-1], histogram)  # <- or here
plt.show()

plt.hist(v_diff.reshape(-1), bins=256)
plt.show()

threshold = 60
mask = np.zeros(v_diff.shape)
mask[v_diff > threshold] = 255
print(mask.shape)
mask = mask.astype("uint8")
kernel = np.ones((5, 5), np.uint8)
# mask = cv.erode(mask, kernel, iterations=1)
plt.imshow(mask, cmap='gray')
plt.show()

# fig, ax = plt.subplots(2, 2)
# ax[0, 0].imshow(img_2)
# ax[0, 1].imshow(h_diff, cmap='gray')
# ax[1, 0].imshow(s_diff, cmap='gray')
# ax[1, 1].imshow(v_diff, cmap='gray')
# plt.show()

plt.imshow(v_diff, cmap='gray')
plt.show()
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img, cmap='gray')
plt.show()

ROI_number = 0
cnts = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    if len(c) < 100:
        continue
    print(len(c))
    x,y,w,h = cv.boundingRect(c)
    cv.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)

plt.imshow(img)
plt.show()



