from DataTransform import *
from C3D_Preprocess import *
import os
import numpy as np
import time
from PIL import Image

'''필요 시 수정  ========================================================================================='''
upper_dir = './Data/C3D_6_9_f16_ovp0'  # to_npy를 통해 생성된 폴더
saveInput_dir = './Data/Concat_Output/'
saveLabel_dir = './Data/Concat_Output/'
frame_overlap = 'f16p0'
suffix = 'right_eye.npy' # left_eye.npy , mouth.npy, right_eye.npy
'''필요 시 수정   ========================================================================================'''

def Collect_all_npy(u_dir):
    inputs_list = []
    labels_list = []
    for dir in os.listdir(upper_dir):
        if dir.startswith('c3d_input') and dir.endswith(suffix):
            d = os.path.join(upper_dir, dir)
            s = np.load(d)
            inputs_list.append(s)

    for dir in os.listdir(upper_dir):
        if dir.startswith('c3d_label') and dir.endswith(suffix):
            d = os.path.join(upper_dir, dir)
            s = np.load(d)
            labels_list.append(s)

    return inputs_list, labels_list

def Concat_all_npy(i_dir, t_dir, input_save_dir, label_save_dir):
    in_len = len(i_dir)
    la_len = len(t_dir)

    print('\n')
    inputs = i_dir[0]
    labels = t_dir[0]
    for i in range(in_len-1):
        p = np.concatenate((inputs, i_dir[i + 1]))
        inputs = p
        print(i+1, 'input concat')
    print('\n')
    for i in range(la_len-1):
        t = np.concatenate((labels, t_dir[i + 1]))
        labels = t
        print(i+1, 'label concat')

    print('\n')
    print(inputs.shape)
    print(labels.shape)

    ## 파일 명 고치기
    input_name = 'Inputs_%s_%08d_%s'%(frame_overlap, inputs.shape[0], suffix)
    label_name = 'Labels_%s_%08d_%s'%(frame_overlap, labels.shape[0], suffix)

    input_save_dir += input_name
    label_save_dir += label_name
    np.save(input_save_dir, inputs)
    np.save(label_save_dir, labels)
    print('Loading again ...\n')



if __name__ == '__main__':
    isConcat = False

    nx = 0
    ny = 0
    for dir in os.listdir(upper_dir):
        if dir.startswith('c3d_input') and dir.endswith(suffix):
            d = os.path.join(upper_dir, dir)
            nx +=1
            print(dir)
    print('total input npy:', nx, '\n')
    for dir in os.listdir(upper_dir):
        if dir.startswith('c3d_label') and dir.endswith(suffix):
            d = os.path.join(upper_dir, dir)
            ny +=1
            print(dir)
    print('total label npy:', ny, '\n')

    xd, yd = Collect_all_npy(upper_dir)
    while True:
        print('check the directories...')
        check = input('proceed [y], stop [n]\n')
        if check == 'y':
            print('proceed')
            isConcat = True
            break
        elif check == 'n':
            print('stop')
            isConcat = False
            break
        else:
            print('Try again')
            continue
    print('\n')
    if isConcat == True:
        Concat_all_npy(xd, yd, saveInput_dir, saveLabel_dir)

        # CHECK
        x = np.load(saveInput_dir)
        y = np.load(saveLabel_dir)

        s3 = x[0, 0]
        img5 = Image.fromarray(s3)

        img5.show()