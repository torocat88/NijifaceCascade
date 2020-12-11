import math
import os

import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('nijiface_cascade.xml')
factor = 1.2

os.makedirs('./faces/', exist_ok=True)     # directory for face images
os.makedirs('./detected/', exist_ok=True)  # directory for source images with detected-face frame

for img_i in os.listdir('./source/'):
    if os.path.splitext(img_i)[-1] == '.jpg':  # skip processing when the extension is not 'jpg'.
        facenum = 0

        src = cv2.imread('./source/' + img_i)

        srcwidth, srcheight = src.shape[:2]

        imwidth  = int(math.hypot(srcwidth, srcheight)) + 2
        imheight = imwidth

        # rotate image by 5 degrees and detect
        for angle_i in range(-20, 25, 5):
            tM = np.float32([[1, 0, (imheight - srcheight) / 2], [0, 1, (imwidth - srcwidth) / 2]])
            img_moved = cv2.warpAffine(src, tM, (imwidth, imheight))  # move image to center

            tM = cv2.getRotationMatrix2D((imwidth * 0.5, imheight * 0.5), angle_i, 1.0)
            img_rotated = cv2.warpAffine(img_moved, tM, (imwidth, imheight))  # rotate image

            src_gray = cv2.cvtColor(img_rotated, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(src_gray, scaleFactor=factor)

            if len(faces) >= 1:
                for x, y, w, h in faces:
                    face = img_rotated[y: y + h, x: x + w]
                    cv2.imwrite('./faces/' + os.path.splitext(os.path.basename(img_i))[0] + '_' + str(angle_i).zfill(3) + '_' + str(facenum) + '.jpg', face)  # save face image
                    cv2.rectangle(img_rotated, (x, y), (x + w, y + h), (255, 0, 0), 2)  # draw frame at detected area of source image
                    facenum += 1

                tM = cv2.getRotationMatrix2D((imheight * 0.5, imwidth * 0.5), -angle_i, 1.0)
                img_crotated = cv2.warpAffine(img_rotated, tM, (imheight, imwidth))

                tM = np.float32([[1, 0, -(imheight - srcheight) / 2], [0, 1, -(imwidth - srcwidth) / 2]])
                img_cmoved = cv2.warpAffine(img_crotated, tM, (srcheight, srcwidth))

                cv2.imwrite('./detected/' + os.path.splitext(os.path.basename(img_i))[0] + '_' + str(angle_i).zfill(3) + '.jpg', img_cmoved)  # save source image with frame
