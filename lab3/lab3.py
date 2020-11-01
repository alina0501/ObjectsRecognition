import numpy as np
from motion_est import blockMotion, motion_comp

from skvideo import io,  datasets


filename = datasets.bigbuckbunny()

videodata = io.vread(filename)

videometadata = io.ffprobe(filename)
frame_rate = videometadata['video']['@avg_frame_rate']

T, M, N, C = videodata.shape

motionData = blockMotion(videodata)

writer = io.FFmpegWriter("motion.mp4", inputdict={
    "-r": frame_rate
})

j = 0

for i in range(0, T - 1, 2):

    outputframe = videodata[i]
    writer.writeFrame(outputframe)
    outputframe = motion_comp(videodata[i], motionData[i//2], mbSize=8)
    j += 1
    writer.writeFrame(outputframe)


writer.close()