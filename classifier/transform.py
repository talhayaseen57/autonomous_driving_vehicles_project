# import the necessary packages
import numpy as np
import cv2


def order_points(pts):
    """takes an array of points coordinates,
    and returns a sorted array: [tl, tr, br, bl]"""

    rect = pts[np.lexsort((pts[:, 0], pts[:, 1]))].astype("float32")
    return rect


def four_point_transform(image, pts):
    """takes image and an array of points and an optional shape,
    and returns the new transformed prespective (birds eye view)"""

    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, bl, br) = rect

    dst = np.array([
        [370, 0],
        [630, 0],
        [630, 800],
        [370, 800]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (1200, 800), cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    warped = cv2.resize(warped, None, fx= 0.3, fy= 0.3, interpolation= cv2.INTER_NEAREST)
    # return the warped image
    return warped
