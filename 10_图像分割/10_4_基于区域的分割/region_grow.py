import numpy as np


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


def getGrayDiff(input_image, currentPoint, tmpPoint):
    return abs(255 - int(input_image[tmpPoint.x, tmpPoint.y]))

    # return abs(
    #     int(input_image[currentPoint.x, currentPoint.y]) -
    #     int(input_image[tmpPoint.x, tmpPoint.y]))


def regionGrow(input_image, seeds, thresh):
    input_image_cp = np.copy(input_image)
    w, h = input_image.shape
    seedMark = np.zeros(input_image.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    connects = [
        Point(-1, -1),
        Point(0, -1),
        Point(1, -1),
        Point(1, 0),
        Point(1, 1),
        Point(0, 1),
        Point(-1, 1),
        Point(-1, 0)
    ]
    while (len(seedList) > 0):
        currentPoint = seedList.pop(0)
        seedMark[currentPoint.x, currentPoint.y] = 1
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= w or tmpY >= h:
                continue
            if seedMark[tmpX, tmpY] == 1:
                continue
            grayDiff = getGrayDiff(input_image_cp, currentPoint,
                                   Point(tmpX, tmpY))
            if grayDiff < thresh:
                seedMark[tmpX, tmpY] = 1
                seedList.append(Point(tmpX, tmpY))
    return seedMark
