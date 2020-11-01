import numpy as np
from skvideo.utils import vshape, rgb2gray


def _costMAD(block1, block2):
    block1 = block1.astype(np.float32)
    block2 = block2.astype(np.float32)
    return np.mean(np.abs(block1 - block2))


def _minCost(costs):
    h, w = costs.shape
    mi = costs[np.int((h - 1) / 2), np.int((w - 1) / 2)]
    dy = np.int((h - 1) / 2)
    dx = np.int((w - 1) / 2)
    # mi = 65535
    # dy = 0
    # dx = 0

    for i in range(h):
        for j in range(w):
            if costs[i, j] < mi:
                mi = costs[i, j]
                dy = i
                dx = j

    return dx, dy, mi


def _checkBounded(xval, yval, w, h, mbSize):
    if ((yval < 0) or
            (yval + mbSize >= h) or
            (xval < 0) or
            (xval + mbSize >= w)):
        return False
    else:
        return True


def arps(imgP, imgI, mbSize, p):
    # Computes motion vectors using Adaptive Rood Pattern Search method
    #
    # Input
    #   imgP : The image for which we want to find motion vectors
    #   imgI : The reference image
    #   mbSize : Size of the macroblock
    #   p : Search parameter  (read literature to find what this means)
    #
    # Ouput
    #   motionVect : the motion vectors for each integral macroblock in imgP
    #   ARPScomputations: The average number of points searched for a macroblock

    h, w = imgP.shape

    vectors = np.zeros((np.int(h / mbSize), np.int(w / mbSize), 2))
    costs = np.ones((6)) * 65537

    SDSP = []
    SDSP.append([0, -1])
    SDSP.append([-1, 0])
    SDSP.append([0, 0])
    SDSP.append([1, 0])
    SDSP.append([0, 1])

    LDSP = {}

    checkMatrix = np.zeros((2 * p + 1, 2 * p + 1))

    computations = 0

    for i in range(0, h - mbSize + 1, mbSize):
        for j in range(0, w - mbSize + 1, mbSize):
            x = j
            y = i

            costs[2] = _costMAD(imgP[i:i + mbSize, j:j + mbSize], imgI[i:i + mbSize, j:j + mbSize])

            checkMatrix[p, p] = 1
            computations += 1

            if (j == 0):
                stepSize = 2
                maxIndex = 5
            else:
                u = vectors[np.int(i / mbSize), np.int(j / mbSize) - 1, 0]
                v = vectors[np.int(i / mbSize), np.int(j / mbSize) - 1, 1]
                stepSize = np.int(np.max((np.abs(u), np.abs(v))))

                if (((np.abs(u) == stepSize) and (np.abs(v) == 0)) or
                        ((np.abs(v) == stepSize) and (np.abs(u) == 0))):
                    maxIndex = 5
                else:
                    maxIndex = 6
                    LDSP[5] = [np.int(v), np.int(u)]

            # large diamond search
            LDSP[0] = [0, -stepSize]
            LDSP[1] = [-stepSize, 0]
            LDSP[2] = [0, 0]
            LDSP[3] = [stepSize, 0]
            LDSP[4] = [0, stepSize]

            for k in range(maxIndex):
                refBlkVer = y + LDSP[k][1]
                refBlkHor = x + LDSP[k][0]

                if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                    continue

                if ((k == 2) or (stepSize == 0)):
                    continue

                costs[k] = _costMAD(imgP[i:i + mbSize, j:j + mbSize],
                                    imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                computations += 1
                checkMatrix[LDSP[k][1] + p, LDSP[k][0] + p] = 1

            if costs[2] != 0:
                point = np.argmin(costs)
                cost = costs[point]
            else:
                point = 2
                cost = costs[point]

            x += LDSP[point][0]
            y += LDSP[point][1]
            costs[:] = 65537
            costs[2] = cost

            doneFlag = 0

            while not doneFlag:
                for k in range(5):
                    refBlkVer = y + SDSP[k][1]
                    refBlkHor = x + SDSP[k][0]

                    if not _checkBounded(refBlkHor, refBlkVer, w, h, mbSize):
                        continue

                    if k == 2:
                        continue
                    elif ((refBlkHor < j - p) or
                          (refBlkHor > j + p) or
                          (refBlkVer < i - p) or
                          (refBlkVer > i + p)):
                        continue
                    elif checkMatrix[y - i + SDSP[k][1] + p, x - j + SDSP[k][0] + p]:
                        continue

                    costs[k] = _costMAD(imgP[i:i + mbSize, j:j + mbSize],
                                        imgI[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize])
                    checkMatrix[y - i + SDSP[k][1] + p, x - j + SDSP[k][0] + p] = 1
                    computations += 1

                if costs[2] != 0:
                    point = np.argmin(costs)
                    cost = costs[point]
                else:
                    point = 2
                    cost = costs[point]

                doneFlag = 1
                if point != 2:
                    doneFlag = 0
                    y += SDSP[point][1]
                    x += SDSP[point][0]
                    costs[:] = 65537
                    costs[2] = cost

            vectors[np.int(i / mbSize), np.int(j / mbSize), :] = [x - j, y - i]

            costs[:] = 65537

            checkMatrix[:, :] = 0

    return vectors, computations / ((h * w) / mbSize ** 2)


def blockMotion(videodata, mbSize=8, p=2, **plugin_args):
    """Block-based motion estimation

    Given a sequence of frames, this function
    returns motion vectors between frames.
    Parameters
    ----------
    videodata : ndarray, shape (numFrames, height, width, channel)
        A sequence of frames
    mbSize : int
        Macroblock size
    p : int
        Algorithm search distance parameter
    Returns
    ----------
    motionData : ndarray, shape (numFrames - 1, height/mbSize, width/mbSize, 2)
        The motion vectors computed from videodata. The first element of the last axis contains the y motion component, and second element contains the x motion component.

    .. Yao Nie and Kai-Kuang Ma, "Adaptive rood pattern search for fast block-matching motion estimation." IEEE Transactions on Image Processing, 11 (12) 1442-1448, Dec 2002
    """
    videodata = vshape(videodata)

    # grayscale
    luminancedata = rgb2gray(videodata)

    numFrames, height, width, channels = luminancedata.shape
    assert numFrames > 1, "Must have more than 1 frame for motion estimation!"

    # luminance is 1 channel, so flatten for computation
    luminancedata = luminancedata.reshape((numFrames, height, width))

    motionData = np.zeros((numFrames - 1, np.int(height / mbSize), np.int(width / mbSize), 2), np.int8)

    # BROKEN, check this
    for i in range(0, numFrames - 1, 2):
        motion, comps = arps(luminancedata[i + 1, :, :], luminancedata[i, :, :], mbSize, p)
        motionData[i, :, :, :] = motion

    return motionData


def motion_comp(framedata, motionVect, mbSize):
    M, N, C = framedata.shape

    compImg = np.zeros((M, N, C))

    for i in range(0, M - mbSize + 1, mbSize):
        for j in range(0, N - mbSize + 1, mbSize):
            dy = motionVect[np.int(i / mbSize), np.int(j / mbSize), 0]
            dx = motionVect[np.int(i / mbSize), np.int(j / mbSize), 1]

            refBlkVer = i + dy
            refBlkHor = j + dx

            # check bounds
            if not _checkBounded(refBlkHor, refBlkVer, N, M, mbSize):
                continue

            compImg[i:i + mbSize, j:j + mbSize, :] = framedata[refBlkVer:refBlkVer + mbSize, refBlkHor:refBlkHor + mbSize, :]
    return compImg
