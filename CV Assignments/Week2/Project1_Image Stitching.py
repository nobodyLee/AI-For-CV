"""
Image Stitching

"""

import numpy as np
import cv2 as cv


def show(images):
    """show multiple images."""
    for i, image in enumerate(images):
        cv.namedWindow(str(i), cv.WINDOW_NORMAL)
        cv.imshow(str(i), image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def keypoints_detect(img):
    """Use SIFT to detect keypoints."""
    sift = cv.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def match(descripter1, descripter2):
    """Match keypoints through descripter."""
    bf = cv.BFMatcher()
    matches = bf.knnMatch(descripter1, descripter2, k=2)
    real_matches = [[i] for i, j in matches if i.distance < 0.75 * j.distance]
    return real_matches


def homography(keypoints1, keypoints2, matches):
    """Calculate homography"""
    pts1 = np.float32([keypoints1[i[0].queryIdx].pt for i in matches])
    pts2 = np.float32([keypoints2[i[0].trainIdx].pt for i in matches])
    matrix, mask = cv.findHomography(pts2, pts1, cv.RANSAC, 5.0)
    return matrix


def image_stitch(image1, image2, homography):
    image = cv.warpPerspective(image2, homography, (image2.shape[1] * 2, image2.shape[0]))
    image[0:image1.shape[0], 0:image1.shape[1]] = image1
    return image


if __name__ == '__main__':
    # Load two images
    img1 = cv.imread('data/qh1.jpg')
    img2 = cv.imread('data/qh2.jpg')

    # Find keypoints
    keypoints1, descripters1 = keypoints_detect(img1)
    keypoints2, descripters2 = keypoints_detect(img2)

    # Calculate homography
    matches = match(descripters1, descripters2)
    M = homography(keypoints1, keypoints2, matches)

    # Stitch images
    img3 = image_stitch(img1, img2, M)

    # Show Matches
    img_match = cv.drawMatchesKnn(img1, keypoints1, img2, keypoints2,
                                  matches[:50], outImg=np.array([]))
    show([img_match, img3])
