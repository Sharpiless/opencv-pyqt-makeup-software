import cv2
import dlib
import numpy as np
import imutils

predictor_path = "./data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


class NoFace(Exception):
    '''
    没脸
    '''
    pass


class Organ():
    def __init__(self, im_bgr, im_hsv, temp_bgr, temp_hsv, landmark, name, ksize=None):
        '''
        五官部位类
        '''
        self.im_bgr, self.im_hsv, self.landmark, self.name = im_bgr, im_hsv, landmark, name
        self.get_rect()
        self.shape = (int(self.bottom-self.top), int(self.right-self.left))
        self.size = self.shape[0]*self.shape[1]*3
        self.move = int(np.sqrt(self.size/3)/20)
        self.ksize = self.get_ksize()
        self.patch_bgr, self.patch_hsv = self.get_patch(
            self.im_bgr), self.get_patch(self.im_hsv)
        self.set_temp(temp_bgr, temp_hsv)
        self.patch_mask = self.get_mask_re()
        pass

    def set_temp(self, temp_bgr, temp_hsv):
        self.im_bgr_temp, self.im_hsv_temp = temp_bgr, temp_hsv
        self.patch_bgr_temp, self.patch_hsv_temp = self.get_patch(
            self.im_bgr_temp), self.get_patch(self.im_hsv_temp)

    def confirm(self):
        '''
        确认操作
        '''
        self.im_bgr[:], self.im_hsv[:] = self.im_bgr_temp[:], self.im_hsv_temp[:]

    def update_temp(self):
        '''
        更新临时图片
        '''
        self.im_bgr_temp[:], self.im_hsv_temp[:] = self.im_bgr[:], self.im_hsv[:]

    def get_ksize(self, rate=15):
        size = max([int(np.sqrt(self.size/3)/rate), 1])
        size = (size if size % 2 == 1 else size+1)
        return (size, size)

    def get_rect(self):
        '''
        获得定位方框
        '''
        ys, xs = self.landmark[:, 1], self.landmark[:, 0]
        self.top, self.bottom, self.left, self.right = np.min(
            ys), np.max(ys), np.min(xs), np.max(xs)

    def get_patch(self, im):
        '''
        截取局部切片
        '''
        shape = im.shape
        x = im[np.max([self.top-self.move, 0]):np.min([self.bottom+self.move, shape[0]]),
               np.max([self.left-self.move, 0]):np.min([self.right+self.move, shape[1]])]
        return x

    def _draw_convex_hull(self, im, points, color):
        '''
        勾画多凸边形
        '''
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)

    def get_mask_re(self, ksize=None):
        '''
        获得局部相对坐标遮罩
        '''
        if ksize == None:
            ksize = self.ksize
        landmark_re = self.landmark.copy()
        landmark_re[:, 1] -= np.max([self.top-self.move, 0])
        landmark_re[:, 0] -= np.max([self.left-self.move, 0])
        mask = np.zeros(self.patch_bgr.shape[:2], dtype=np.float64)

        self._draw_convex_hull(mask,
                               landmark_re,
                               color=1)

        mask = np.array([mask, mask, mask]).transpose((1, 2, 0))
        mask = (cv2.GaussianBlur(mask, ksize, 0) > 0) * 1.0
        return cv2.GaussianBlur(mask, ksize, 0)[:]

    def get_mask_abs(self, ksize=None):
        '''
        获得全局绝对坐标遮罩
        '''
        if ksize == None:
            ksize = self.ksize
        mask = np.zeros(self.im_bgr.shape, dtype=np.float64)
        patch = self.get_patch(mask)
        patch[:] = self.patch_mask[:]
        return mask

    def whitening(self, rate=0.15, confirm=True):
        '''
        提亮美白
        '''
        if confirm:
            self.confirm()
            self.patch_hsv[:, :, -1] = np.minimum(self.patch_hsv[:, :, -1]+self.patch_hsv[:, :, -1]
                                                  * self.patch_mask[:, :, -1]*rate, 255).astype('uint8')
            self.im_bgr[:] = cv2.cvtColor(self.im_hsv, cv2.COLOR_HSV2BGR)[:]
            self.update_temp()
        else:
            self.patch_hsv_temp[:] = cv2.cvtColor(
                self.patch_bgr_temp, cv2.COLOR_BGR2HSV)[:]
            self.patch_hsv_temp[:, :, -1] = np.minimum(self.patch_hsv_temp[:, :, -1] +
                                                       self.patch_hsv_temp[:, :, -1]*self.patch_mask[:, :, -1]*rate, 255).astype('uint8')
            self.patch_bgr_temp[:] = cv2.cvtColor(
                self.patch_hsv_temp, cv2.COLOR_HSV2BGR)[:]

    def brightening(self, rate=0.3, confirm=True):
        '''
        提升鲜艳度
        '''
        patch_mask = self.get_mask_re((1, 1))
        if confirm:
            self.confirm()
            patch_new = self.patch_hsv[:, :, 1]*patch_mask[:, :, 1]*rate
            patch_new = cv2.GaussianBlur(patch_new, (3, 3), 0)
            self.patch_hsv[:, :, 1] = np.minimum(
                self.patch_hsv[:, :, 1]+patch_new, 255).astype('uint8')
            self.im_bgr[:] = cv2.cvtColor(self.im_hsv, cv2.COLOR_HSV2BGR)[:]
            self.update_temp()
        else:
            self.patch_hsv_temp[:] = cv2.cvtColor(
                self.patch_bgr_temp, cv2.COLOR_BGR2HSV)[:]
            patch_new = self.patch_hsv_temp[:, :, 1]*patch_mask[:, :, 1]*rate
            patch_new = cv2.GaussianBlur(patch_new, (3, 3), 0)
            self.patch_hsv_temp[:, :, 1] = np.minimum(
                self.patch_hsv[:, :, 1]+patch_new, 255).astype('uint8')
            self.patch_bgr_temp[:] = cv2.cvtColor(
                self.patch_hsv_temp, cv2.COLOR_HSV2BGR)[:]

    def smooth(self, rate=0.6, ksize=(7, 7), confirm=True):
        '''
        磨皮
        '''
        if ksize == None:
            ksize = self.get_ksize(80)
        index = self.patch_mask > 0
        if confirm:
            self.confirm()
            patch_new = cv2.GaussianBlur(cv2.bilateralFilter(
                self.patch_bgr, 3, *ksize), ksize, 0)
            self.patch_bgr[index] = np.minimum(
                rate*patch_new[index]+(1-rate)*self.patch_bgr[index], 255).astype('uint8')
            self.im_hsv[:] = cv2.cvtColor(self.im_bgr, cv2.COLOR_BGR2HSV)[:]
            self.update_temp()
        else:
            patch_new = cv2.GaussianBlur(cv2.bilateralFilter(
                self.patch_bgr_temp, 3, *ksize), ksize, 0)
            self.patch_bgr_temp[index] = np.minimum(
                rate*patch_new[index]+(1-rate)*self.patch_bgr_temp[index], 255).astype('uint8')
            self.patch_hsv_temp[:] = cv2.cvtColor(
                self.patch_bgr_temp, cv2.COLOR_BGR2HSV)[:]

    def sharpen(self, rate=0.3, confirm=True):
        '''
        锐化
        '''
        patch_mask = self.get_mask_re((3, 3))
        kernel = np.zeros((9, 9), np.float32)
        kernel[4, 4] = 2.0  # Identity, times two!
        # Create a box filter:
        boxFilter = np.ones((9, 9), np.float32) / 81.0

        # Subtract the two:
        kernel = kernel - boxFilter
        index = patch_mask > 0
        if confirm:
            self.confirm()
            sharp = cv2.filter2D(self.patch_bgr, -1, kernel)
            self.patch_bgr[index] = np.minimum(
                ((1-rate)*self.patch_bgr)[index]+sharp[index]*rate, 255).astype('uint8')
            self.update_temp()
        else:
            sharp = cv2.filter2D(self.patch_bgr_temp, -1, kernel)
            self.patch_bgr_temp[:] = np.minimum(
                self.patch_bgr_temp+self.patch_mask*sharp*rate, 255).astype('uint8')
            self.patch_hsv_temp[:] = cv2.cvtColor(
                self.patch_bgr_temp, cv2.COLOR_BGR2HSV)[:]


class Forehead(Organ):
    def __init__(self, im_bgr, im_hsv, temp_bgr, temp_hsv, landmark, mask_organs, name, ksize=None):
        self.mask_organs = mask_organs
        super(Forehead, self).__init__(im_bgr, im_hsv,
                                       temp_bgr, temp_hsv, landmark, name, ksize)

    def get_mask_re(self, ksize=None):
        '''
        获得局部相对坐标遮罩
        '''
        if ksize == None:
            ksize = self.ksize
        landmark_re = self.landmark.copy()
        landmark_re[:, 1] -= np.max([self.top-self.move, 0])
        landmark_re[:, 0] -= np.max([self.left-self.move, 0])
        mask = np.zeros(self.patch_bgr.shape[:2], dtype=np.float64)

        self._draw_convex_hull(mask,
                               landmark_re,
                               color=1)

        mask = np.array([mask, mask, mask]).transpose((1, 2, 0))

        mask = (cv2.GaussianBlur(mask, ksize, 0) > 0) * 1.0
        patch_organs = self.get_patch(self.mask_organs)
        mask = cv2.GaussianBlur(mask, ksize, 0)[:]
        mask[patch_organs > 0] = (1-patch_organs[patch_organs > 0])
        return mask


class Face(Organ):
    '''
    脸类
    '''

    def __init__(self, im_bgr, img_hsv, temp_bgr, temp_hsv, landmarks, index):
        self.index = index
        # 五官名称
        self.organs_name = ['jaw', 'mouth', 'nose',
                            'left eye', 'right eye', 'left brow', 'right brow']

        # 五官等标记点
        self.organs_points = [list(range(0, 17)), list(range(48, 61)), list(range(27, 35)), list(
            range(42, 48)), list(range(36, 42)), list(range(22, 27)), list(range(17, 22))]

        # 实例化脸对象和五官对象
        self.organs = {name: Organ(im_bgr, img_hsv, temp_bgr, temp_hsv, landmarks[points], name) for name, points in zip(
            self.organs_name, self.organs_points)}

        # 获得额头坐标，实例化额头
        mask_nose = self.organs['nose'].get_mask_abs()
        mask_organs = (self.organs['mouth'].get_mask_abs()+mask_nose+self.organs['left eye'].get_mask_abs(
        )+self.organs['right eye'].get_mask_abs()+self.organs['left brow'].get_mask_abs()+self.organs['right brow'].get_mask_abs())
        forehead_landmark = self.get_forehead_landmark(
            im_bgr, landmarks, mask_organs, mask_nose)
        self.organs['forehead'] = Forehead(
            im_bgr, img_hsv, temp_bgr, temp_hsv, forehead_landmark, mask_organs, 'forehead')
        mask_organs += self.organs['forehead'].get_mask_abs()

        # 人脸的完整标记点
        self.FACE_POINTS = np.concatenate([landmarks, forehead_landmark])
        super(Face, self).__init__(im_bgr, img_hsv,
                                   temp_bgr, temp_hsv, self.FACE_POINTS, 'face')

        mask_face = self.get_mask_abs()-mask_organs
        self.patch_mask = self.get_patch(mask_face)
        pass

    def get_forehead_landmark(self, im_bgr, face_landmark, mask_organs, mask_nose):
        '''
        计算额头坐标
        '''
        # 画椭圆
        radius = (np.linalg.norm(
            face_landmark[0]-face_landmark[16])/2).astype('int32')
        center_abs = tuple(
            ((face_landmark[0]+face_landmark[16])/2).astype('int32'))

        angle = np.degrees(np.arctan(
            (lambda l: l[1]/l[0])(face_landmark[16]-face_landmark[0]))).astype('int32')
        mask = np.zeros(mask_organs.shape[:2], dtype=np.float64)
        cv2.ellipse(mask, center_abs, (radius, radius), angle, 180, 360, 1, -1)
        # 剔除与五官重合部分
        mask[mask_organs[:, :, 0] > 0] = 0
        # 根据鼻子的肤色判断真正的额头面积
        index_bool = []
        for ch in range(3):
            mean, std = np.mean(im_bgr[:, :, ch][mask_nose[:, :, ch] > 0]), np.std(
                im_bgr[:, :, ch][mask_nose[:, :, ch] > 0])
            up, down = mean+0.5*std, mean-0.5*std
            index_bool.append((im_bgr[:, :, ch] < down)
                              | (im_bgr[:, :, ch] > up))
        index_zero = (
            (mask > 0) & index_bool[0] & index_bool[1] & index_bool[2])
        mask[index_zero] = 0
        index_abs = np.array(np.where(mask > 0)[::-1]).transpose()
        landmark = cv2.convexHull(index_abs).squeeze()
        return landmark


class Makeup():
    '''
    化妆器
    '''

    def __init__(self, predictor_path="./data/shape_predictor_68_face_landmarks.dat"):
        self.photo_path = []
        self.PREDICTOR_PATH = predictor_path
        self.faces = {}

        # 人脸定位、特征提取器，来自dlib
        self.detector = detector
        self.predictor = predictor

    def get_faces(self, im_bgr, im_hsv, temp_bgr, temp_hsv, name, n=1):
        '''
        人脸定位和特征提取，定位到两张及以上脸或者没有人脸将抛出异常
        im:
            照片的numpy数组
        fname:
            照片名字的字符串
        返回值:
            人脸特征(x,y)坐标的矩阵
        '''
        rects = self.detector(im_bgr, 1)

        if len(rects) < 1:
            raise NoFace('Too many faces in '+name)
        return {name: [Face(im_bgr, im_hsv, temp_bgr, temp_hsv, np.array([[p.x, p.y] for p in self.predictor(im_bgr, rect).parts()]), i) for i, rect in enumerate(rects)]}

    def read_im(self, fname, scale=1):
        '''
        读取图片
        '''
        im = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), -1)
        im = imutils.resize(im, width=600)
        if type(im) == type(None):
            raise ValueError(
                'Opencv error reading image "{}" , got None'.format(fname))
        return im

    def read_and_mark(self, fname):
        im_bgr = self.read_im(fname)
        im_hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
        temp_bgr, temp_hsv = im_bgr.copy(), im_hsv.copy()
        return im_bgr, temp_bgr, self.get_faces(im_bgr, im_hsv, temp_bgr, temp_hsv, fname)
