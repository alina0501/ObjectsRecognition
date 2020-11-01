import cv2 as cv
import pandas as pd
import numpy as np
import os
import time


def train_des(img_path, outp_path):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    sift = cv.xfeatures2d.SIFT_create()
    keypoints_train, des_train = sift.detectAndCompute(img, None)

    img_train = cv.drawKeypoints(img, keypoints_train, img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite(outp_path, img_train)

    return keypoints_train, des_train


sift = cv.xfeatures2d.SIFT_create()

keypoints_train, des_train = {}, {}
keypoints_train['alina'], des_train['alina'] = train_des('src/object.jpg', 'src/outp/keypts_object.jpg')
keypoints_train['seva'], des_train['seva'] = train_des('src/origin.jpg', 'src/outp/keypts_object_seva.jpg')
keypoints_train['olya'], des_train['olya'] = train_des('src/goose_cup_train/train_goose2.jpg',
                                                       'src/outp/keypts_object_olya.jpg')


collection = {}
os.getcwd()
collection['alina'] = f'{os.getcwd()}/src/alina'
collection['seva'] = f'{os.getcwd()}/src/photo'
collection['olya'] = f'{os.getcwd()}/src/goose_cup'

for key in collection.keys():

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    good = dict()
    MIN_MATCH_COUNT = 10
    metrics = {'1': [], '2': [], '3': []}

    for i, filename in enumerate(os.listdir(collection[key])):
        img_test = cv.imread(f"{collection[key]}/{filename}", cv.IMREAD_GRAYSCALE)

        start_time = time.time()

        keypoints_test, des_test = sift.detectAndCompute(img_test, None)

        metrics['3'].append((time.time() - start_time, img_test.shape[0] + img_test.shape[1]))

        matches = flann.knnMatch(des_test, des_train[key], k=2)

        good[str(i)] = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good[f'{i}'].append(m)

        metrics['1'].append(len(good[str(i)]) / len(keypoints_test))
        metrics['2'].append(np.mean([m.distance for m, n in matches]))

        # if len(good[str(i)]) >= MIN_MATCH_COUNT:
        #
        #     src_pts = np.float32([keypoints_test[m.queryIdx].pt for m in good[str(i)]]).reshape(-1, 1, 2)
        #     dst_pts = np.float32([keypoints_train[m.trainIdx].pt for m in good[str(i)]]).reshape(-1, 1, 2)
        #     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        #     matchesMask = mask.ravel().tolist()
        #     metric1[str(i)] = sum(matchesMask) / len(keypoints_test)
        #
        # else:
        #     print(f"{i}.jpg Not enough matches are found - {len(good[str(i)])}/{MIN_MATCH_COUNT}")
        #     matchesMask = None
        #     metric1[str(i)] = len(good[str(i)])/len(keypoints_test)

    metrics = pd.DataFrame(data=metrics, index=os.listdir(collection[key]))

    metrics.to_csv(f'outp/{key}.csv')


# print(good)
