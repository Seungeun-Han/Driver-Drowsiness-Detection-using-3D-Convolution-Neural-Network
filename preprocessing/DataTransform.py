import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class DataTransform():
    def __init__(self, path):
        '''
        :param path: ./Datasets/Detected_Training_Dataset/
        '''
        self.path = path
    def Encode_input(self, resize, file_number, class1_number, class2_number, class3_number):
        '''
        :param file_number: 1 ~ 18
        :param class1_number: glasses, night no glasses, night glasses, no glasses, sun glasses 1 ~ 5
        :param class2_number: no sleepy, sleepy, slow blink with nodding, yawning 1 ~ 4
        :param class3_number: left eye, mouth, right eye 1 ~ 3
        :return: numpy array (N, 64, 64, 3)
        '''
        fileN = ['001/', '002/', '005/', '006/', '008/', '009/', '012/', '013/', '015/', '020/', '023/', '024/', '031/', '032/', '033/', '034/', '035/', '036/']
        c1N = ['glasses/', 'night_noglasses/', 'nightglasses/', 'noglasses/', 'sunglasses/']
        c2N = ['nonsleepyCombination/', 'sleepyCombination/', 'slowBlinkWithNodding/', 'yawning/']
        c3N = ['left_eye/', 'mouth/', 'right_eye/', 'face/']

        path = self.path
        total_path = os.path.join(path, fileN[file_number-1], c1N[class1_number-1], c2N[class2_number-1], c3N[class3_number-1])

        imgs = os.listdir(total_path)
        size = len(imgs)
        figs = []
        for i in range(size):
            dir4 = os.path.join(total_path, imgs[i])
            m = cv2.imread(dir4, cv2.IMREAD_COLOR)
            if resize != 0: m = cv2.resize(m, dsize=(resize, resize), interpolation=cv2.INTER_AREA)
            figs.append(np.array(m))
        result = np.array(figs)
        return result

    def Encode_target(self, file_number, class1_number, class2_number):
        '''
        :param file_number: 1 ~ 18
        :param class1_number: glasses, night no glasses, night glasses, no glasses, sun glasses 1 ~ 5
        :param class2_number: no sleepy, sleepy, slow blink with nodding, yawning 1 ~ 4
        :return: numpy array (N, 2)
        '''
        fileN = ['001', '002', '005', '006', '008', '009', '012', '013', '015', '020', '023', '024', '031','032', '033', '034', '035', '036']
        c1N = ['glasses/', 'night_noglasses/', 'nightglasses/', 'noglasses/', 'sunglasses/']
        c2N = ['_nonsleepyCombination_drowsiness', '_sleepyCombination_drowsiness', '_slowBlinkWithNodding_drowsiness', '_yawning_drowsiness']
        path = self.path
        file = fileN[file_number - 1] +'/'
        dir = fileN[file_number - 1] + c2N[class2_number - 1] + '.txt'
        total_path = os.path.join(path, file, c1N[class1_number - 1], dir)

        ret = []
        target = open(total_path, 'r')
        text = target.read()
        for x in range(len(text)):
            t = int(text[x])
            code = []
            if t == 0:
                code.append(1)
                code.append(0)
            if t == 1:
                code.append(0)
                code.append(1)
            ret.append(code)
        return np.array(ret)






