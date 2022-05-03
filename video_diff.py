import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


def preprocess(img, gauss_kernel=(30, 30), des_width=1000, cut_off_line=266):
    dim = (des_width, int(des_width * img.shape[0] / img.shape[1]))
    img = cv.resize(img, dim, interpolation=cv.INTER_AREA)
    img = cv.GaussianBlur(img, gauss_kernel, cv.BORDER_DEFAULT)
    img = img[cut_off_line:, :]
    return img


def to_cspace(img):
    img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    return img


def apply_kmeans(img, k=5):
    orig_shape = img.shape
    img = img[:, :, 1].reshape((-1, 1))
    img = np.float32(img)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 0.5)
    attempts = 10
    _, label, center = cv.kmeans(img, k, None, criteria, attempts, cv.KMEANS_PP_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((list(orig_shape[:2]) + [1]))

    return result_image


def add_upper_img(img, upper_size):
    img_upper = (np.ones((upper_size, img.shape[1], 1)) * np.median(img)).astype(np.uint8)
    img = np.concatenate((img_upper, img), axis=0)
    return img


def add_to_buff(buff, img, buff_len=100):
    if (len(buff.shape) == 4) and (buff.shape[0] == buff_len):
        buff[:buff_len-1] = buff[1:buff_len]
        buff[buff_len-1] = img
    elif len(buff.shape) < 3:
        buff = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    else:
        buff = np.concatenate((buff, img.reshape(1, img.shape[0], img.shape[1], img.shape[2])), axis=0)
    return buff


def get_similar_img_buff(img_req, buff, thresh=0):
    img_req = np.repeat(img_req, buff.shape[0]).reshape(buff.shape[0], img_req.shape[0], img_req.shape[1], img_req.shape[2])
    diff = img_req - buff
    mean = np.mean(diff, axis=0)
    return np.argmin(mean)

buff = np.array([])
vid_path = "/home/cstar/Downloads/beryllium-4-Spiral-Gear-V2_20220314162608.mp4"
cap = cv.VideoCapture(vid_path)
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # Display the resulting frame
        cv.imshow('Frame', frame)

        buff = add_to_buff(buff, frame)
        print(buff.shape)

        # Press Q on keyboard to  exit
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()


inp_path = 'data/'
img_1 = cv.imread(os.path.join(inp_path, 'Screenshot from beryllium-4-Spiral-Gear-V2_20220314162608.mp4 - 2.png'), 1)
img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)
img_2 = cv.imread(os.path.join(inp_path, 'Screenshot from beryllium-4-Spiral-Gear-V2_20220314162608.mp4 - 3.png'), 1)
img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2RGB)

img_1_prep = preprocess(img_1, gauss_kernel=(15, 15), des_width=1000, cut_off_line=266)
img_1_hsv = to_cspace(img_1_prep)

img_2_prep = preprocess(img_2, gauss_kernel=(15, 15), des_width=1000, cut_off_line=266)
img_2_hsv = to_cspace(img_2_prep)

fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(img_1)
ax[0, 1].imshow(img_1_hsv[:, :, 0], cmap='gray')
ax[1, 0].imshow(img_1_hsv[:, :, 1], cmap='gray')
ax[1, 1].imshow(img_1_hsv[:, :, 2], cmap='gray')
plt.show()

k = 3
result_image_1 = apply_kmeans(img_1_hsv, k=k)

img_hsv_concat = np.concatenate((img_1_hsv, img_2_hsv), axis=1)
print(img_hsv_concat.shape)
result_image_con = apply_kmeans(img_hsv_concat, k=k)

# _, thresh = cv.threshold(result_image_con[:,:,0], 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
# _, result_image_con, _, _ = cv.floodFill(result_image_con, None, (1360, 605), 255)
# _, result_image_con, _, _ = cv.floodFill(result_image_con, None, (1335, 371), 255)
# print(thresh.shape)

result_image_con = add_upper_img(result_image_con, upper_size=266)




[img1_res, img2_res] = [result_image_con[:, :img_1_hsv.shape[1], :], result_image_con[:, img_1_hsv.shape[1]:, :]]


img_concat = np.concatenate((img_1, img_2), axis=1)

# figure_size = 15
# plt.figure(figsize=(figure_size, figure_size))
# plt.subplot(1, 2, 1), plt.imshow(img_1)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(1, 2, 2), plt.imshow(result_image_1[:, :, 0])
# plt.title('Segmented Image when K = %i' % k), plt.xticks([]), plt.yticks([])
# plt.show()

fig, ax = plt.subplots(2, 1)
ax[0].imshow(img_concat)
# ax[0, 1].imshow(result_image_1[:, :, 0])
ax[1].imshow(result_image_con[:, :, 0])
# ax[1, 1].imshow(thresh)
plt.show()