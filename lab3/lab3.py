import time
import cv2
from skvideo import io, measure, motion, utils
import os
import numpy as np
from motion_algos import blockMotion, blockComp
import skvideo.datasets
import csv
import matplotlib.pyplot as plt

VID_NAME = 'video.mp4'
ENCODED_NAME = f'encoded_{VID_NAME[:-4]}.npz'
METHODS = ['4SS', '3SS', 'DS']

metrics = {'4SS': {}, '3SS': {}, 'DS': {}}


def encoder(videodata, method='DS', mbSize=8, p=2, compute_motion_if_exists=False):
    encoded_name = method + '_' + ENCODED_NAME

    if not (compute_motion_if_exists) and os.path.isfile(encoded_name):
        data_calculated = np.load(encoded_name)
        return (data_calculated['video'], data_calculated['motion'])

    motion = blockMotion(videodata, method=method, mbSize=mbSize, p=p)
    cut_vid = np.delete(videodata, list(range(1, videodata.shape[0], 2)), axis=0)

    np.savez_compressed(encoded_name, video=cut_vid, motion=motion)

    return (cut_vid, motion)


def player(videodata, video, motionData, method):
    T, M, N, C = videodata.shape
    T_orig, M_orig, N_orig, C_orig = video.shape
    # motionData = blockMotion(videodata)
    start_time = time.time()
    writer = io.FFmpegWriter(f"{method}_motion.mp4")

    for i in range(T - (T_orig % 2)):
        outputframe = videodata[i]
        writer.writeFrame(outputframe)

        outputframe = blockComp(videodata[i], motionData[i], mbSize=8)
        writer.writeFrame(outputframe)

        outputframe = utils.rgb2gray(utils.vshape(outputframe))
        origframe = utils.rgb2gray(utils.vshape(video[i * 2 + 1, :, :, :]))
        mae = measure.mae(origframe, outputframe)
        metrics[method]['visual_distance'].append(float(mae))
    if T_orig % 2:
        writer.writeFrame(videodata[T - 1])
    writer.close()

    play_time = time.time() - start_time
    # video = cv2.VideoCapture(f'{method}_motion.mp4')
    # while not video.isOpened():
    #     if video.isOpened():
    #         break
    # fps = video.get(cv2.CAP_PROP_FPS)
    #
    # while True:
    #
    #     ret, frame = video.read()
    #     if not ret:
    #         break
    #
    #     cv2.imshow(f'{method}_player', frame)
    #     if cv2.waitKey(int(1000. / float(fps))) & 0xFF == ord('q'):
    #         break
    #
    # video.release()
    #
    # cv2.destroyAllWindows()
    return play_time


def metrics_calc(method, video):
    start_time = time.time()
    encoder(video, method=method, mbSize=8)
    metrics[method]['time_enc'] = time.time() - start_time

    encoded_file = np.load(f"{method}_encoded_video.npz")
    videodata = encoded_file['video']
    motionData = encoded_file['motion']
    print(motionData.shape)
    metrics[method]['time_play'] = player(videodata, video, motionData, method)

    plt.plot(metrics[method]['visual_distance'], label=f'{method} visual distance')
    plt.show()


if __name__ == '__main__':
    videodata = skvideo.io.vread(VID_NAME)
    print(videodata.shape)
    for method in METHODS:
        metrics[method]['visual_distance'] = []
        metrics_calc(method, videodata)

        w = csv.writer(open(f"{method}_output.csv", "w+"))
        for key in metrics[method].keys():
            w.writerow([key, metrics[method][key]])
