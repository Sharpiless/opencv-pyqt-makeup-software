import dlib
import cv2
import numpy as np
import math


def SharpenImage(src):
    blur_img = cv2.GaussianBlur(src, (0, 0), 5)
    usm = cv2.addWeighted(src, 1.5, blur_img, -0.5, 0)
    return usm


def BilinearInsert(src, ux, uy):
    # 双线性插值法
    w, h, c = src.shape
    if c == 3:
        x1 = int(ux)
        x2 = x1+1
        y1 = int(uy)
        y2 = y1+1

        part1 = src[y1, x1].astype(np.float)*(float(x2)-ux)*(float(y2)-uy)
        part2 = src[y1, x2].astype(np.float)*(ux-float(x1))*(float(y2)-uy)
        part3 = src[y2, x1].astype(np.float) * (float(x2) - ux)*(uy-float(y1))
        part4 = src[y2, x2].astype(np.float) * \
            (ux-float(x1)) * (uy - float(y1))

        insertValue = part1+part2+part3+part4

        return insertValue.astype(np.int8)


def localTranslationWarp(srcImg, startX, startY, endX, endY, radius):

    ddradius = float(radius * radius)
    copyImg = np.zeros(srcImg.shape, np.uint8)
    copyImg = srcImg.copy()

    # 计算公式中的|m-c|^2
    ddmc = (endX - startX) * (endX - startX) + \
        (endY - startY) * (endY - startY)
    H, W, C = srcImg.shape
    for i in range(W):
        for j in range(H):
            # 计算该点是否在形变圆的范围之内
            # 优化，第一步，直接判断是会在（startX,startY)的矩阵框中
            if math.fabs(i-startX) > radius and math.fabs(j-startY) > radius:
                continue

            distance = (i - startX) * (i - startX) + \
                (j - startY) * (j - startY)

            if(distance < ddradius):
                # 计算出（i,j）坐标的原坐标
                # 计算公式中右边平方号里的部分
                ratio = (ddradius-distance) / (ddradius - distance + ddmc)
                ratio = ratio * ratio

                # 映射原位置
                UX = i - ratio * (endX - startX)
                UY = j - ratio * (endY - startY)

                # 根据双线性插值法得到UX，UY的值
                value = BilinearInsert(srcImg, UX, UY)
                # 改变当前 i ，j的值
                copyImg[j, i] = value

    return copyImg


def landmark_dec_dlib_fun(img_src, detector, predictor):
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    land_marks = []

    rects = detector(img_gray, 0)

    for i in range(len(rects)):
        land_marks_node = np.matrix(
            [[p.x, p.y] for p in predictor(img_gray, rects[i]).parts()])
        land_marks.append(land_marks_node)

    return land_marks


def face_thin_auto(src, detector, predictor):

    landmarks = landmark_dec_dlib_fun(src, detector, predictor)

    # 如果未检测到人脸关键点，就不进行瘦脸
    if len(landmarks) == 0:
        return src

    for landmarks_node in landmarks:
        left_landmark = landmarks_node[3]
        left_landmark_down = landmarks_node[5]

        right_landmark = landmarks_node[13]
        right_landmark_down = landmarks_node[15]

        endPt = landmarks_node[30]

        # 计算第4个点到第6个点的距离作为瘦脸距离
        r_left = math.sqrt((left_landmark[0, 0]-left_landmark_down[0, 0])*(left_landmark[0, 0]-left_landmark_down[0, 0]) +
                           (left_landmark[0, 1] - left_landmark_down[0, 1]) * (left_landmark[0, 1] - left_landmark_down[0, 1]))

        # 计算第14个点到第16个点的距离作为瘦脸距离
        r_right = math.sqrt((right_landmark[0, 0]-right_landmark_down[0, 0])*(right_landmark[0, 0]-right_landmark_down[0, 0]) +
                            (right_landmark[0, 1] - right_landmark_down[0, 1]) * (right_landmark[0, 1] - right_landmark_down[0, 1]))

        # 瘦左边脸
        thin_image = localTranslationWarp(
            src, left_landmark[0, 0], left_landmark[0, 1], endPt[0, 0], endPt[0, 1], r_left)
        # 瘦右边脸
        thin_image = localTranslationWarp(
            thin_image, right_landmark[0, 0], right_landmark[0, 1], endPt[0, 0], endPt[0, 1], r_right)

        return thin_image
