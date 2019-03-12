import cv2

def warp_image(img, pts_src, pts_dst, grayscale=False):
    """
    Dewarp image

    :param img:
    :param pts_src:
    :param pts_dst:
    :return:
    """
    if grayscale:
        height, width = img.shape
    else:
        height, width, _ = img.shape

    #  Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    #  Warp source image to destination based on homography; format (4,2)
    return cv2.warpPerspective(src=img, M=h, dsize=(width, height))


def visualize_xy(X, Y):
    """
    Draw detected corner points on original image

    :param X:
    :param Y:
    :return:
    """
    # unpack corner points
    x1, y1, x2, y2, x3, y3, x4, y4 = Y
    img = X.copy()

    # draw circles
    img = cv2.circle(img, (int(x1), int(y1)), 5, (0,0,255))
    img = cv2.circle(img, (int(x2), int(y2)), 5, (0,0,255))
    img = cv2.circle(img, (int(x3), int(y3)), 5, (0,0,255))
    img = cv2.circle(img, (int(x4), int(y4)), 5, (0,0,255))
    return img