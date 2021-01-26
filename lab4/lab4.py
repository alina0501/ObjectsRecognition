from skvideo import io, motion
from motion_algos import blockComp
import skvideo.datasets


VID_NAME = 'video.mp4'
METHODS = ['3SS', '4SS', 'DS']


def encoder(videodata, method):

    motionData = motion.blockMotion(videodata, mbSize=32, method=method, p=2)

    return (videodata, motionData)


def player(videodata, motionData, method):
    T, M, N, C = videodata.shape
    writer = io.FFmpegWriter(f"{method}_mosh.mp4")
    outputframe = videodata[0]
    writer.writeFrame(outputframe)

    for i in range(T-1):
        outputframe = blockComp(outputframe,  motionData[i], mbSize=32)

        writer.writeFrame(outputframe)

    writer.close()

    return 0


if __name__ == '__main__':
    videodata = skvideo.io.vread(VID_NAME)
    #videodata = videodata[:60, :, :, :]
    for method in METHODS:
        video, motionData = encoder(videodata, method)
        player(video, motionData, method)

