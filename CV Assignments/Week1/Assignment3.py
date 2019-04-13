"""
Assignment 3:
Combine image crop, color shift, rotation and perspective transform together to
complete a data augmentation script
"""
import os
import random
import numpy as np
import cv2 as cv


class ImagAug:

    def __init__(self):
        self.img = None
        self.name = ''

    def crop(self, img):
        h, w, _ = img.shape
        rand = random.random() / 4
        up = int(h * rand)
        bottom = int(h * (1 - rand))
        left = int(w * rand)
        right = int(w * (1 - rand))
        return img[up:bottom, left:right]

    def color_shift(self, img):
        B, G, R = cv.split(img)
        for i in [B, G, R]:
            rand = random.randint(-50, 50)
            # print(rand)
            if rand == 0:
                pass
            elif rand > 0:
                lim = 255 - rand
                i[i > lim] = 255
                i[i <= lim] = (rand + i[i <= lim]).astype(img.dtype)
            else:
                lim = 0 - rand
                i[i < lim] = 0
                i[i >= lim] = (rand + i[i >= lim]).astype(img.dtype)
        return cv.merge((B, G, R))

    def rotate(self, img):
        angle = random.randrange(10, 350, 10)
        M = cv.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
        img_rotate = cv.warpAffine(img, M, (img.shape[1], img.shape[0]))
        return img_rotate

    def perspective_transform(self, img):
        height, width, channels = img.shape
        random_margin = 60
        param = dict()
        for i in ['x1', 'x2', 'y1', 'y2', 'x4', 'dx1', 'dy1', 'dy2', 'dx4']:
            param[i] = random.randint(-random_margin, random_margin)
        for i in ['x2', 'x3', 'dx2', 'dx3']:
            param[i] = random.randint(width - random_margin - 1, width - 1)
        for i in ['y3', 'y4', 'dy3', 'dy4']:
            param[i] = random.randint(height - random_margin - 1, height - 1)

        pts1 = np.float32([[param['x1'], param['y1']], [param['x2'], param['y2']],
                           [param['x3'], param['y3']], [param['x4'], param['y4']]])
        pts2 = np.float32([[param['dx1'], param['dy1']], [param['dx2'], param['dy2']],
                           [param['dx3'], param['dy3']], [param['dx4'], param['dy4']]])
        M_warp = cv.getPerspectiveTransform(pts1, pts2)
        img_warp = cv.warpPerspective(img, M_warp, (width, height))
        return img_warp

    def generate(self, img):
        gene_funcs = random.choices([self.crop, self.color_shift, self.rotate, self.perspective_transform],
                                    k=random.randint(1, 4))
        for func in gene_funcs:
            self.img = func(img)
            self.name = self.name + ' ' + func.__name__
        return self.img

    def save(self, name):
        img_name = self.name + ' ' + name
        # Check if img_name exists, if True, plus 1
        img_num = 1
        while os.path.exists('data/{}'.format(img_name)):
            img_name = str(img_num) + self.name + ' ' + name
            img_num += 1
        cv.imwrite('data/{}'.format(img_name), self.img)


if __name__ == '__main__':
    image = cv.imread('data/cat.jpg')
    for _ in range(50):
        new_img = ImagAug()
        new_img.generate(image)
        new_img.save('cat.jpg')


