from DataTransform import *
from C3D_Preprocess import *
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

'''필요 시 수정 ========================================================================================'''
# eyemouth: em = left (1), mouth (2), right (3), face (4)
em = 2
d_dir = './NTHU-DDD-gc/Training_Evaluation_Dataset/Training_Dataset/' # 데이터 디렉터리
t_dir = './NTHU-DDD-gc/Training_Evaluation_Dataset/Training_Dataset/' # target 데이터 디렉터리 (레이블 디렉터리)
u_dir = './NTHU-DDD-npy/'# 저장할 디렉터리
scale = 112
'''필요 시 수정 ======================================================================================'''

people = ['001', '002', '005', '006', '008', '009', '012', '013', '015', '020', '023', '024', '031', '032', '033',
          '034', '035', '036']

today = 'C3D_' + str(time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday)


def prompt_computing(person, glass, sleepy, eyemouth, f, ovp, data_dir, label_dir, upper_dir, save_dir, down_size):
    d = DataTransform(data_dir)
    t = DataTransform(label_dir)

    fileN = ['001', '002', '005', '006', '008', '009', '012', '013', '015', '020', '023', '024', '031', '032', '033',
             '034', '035', '036']
    c1N = ['gls', 'nxgls', 'ngls', 'xgls',
           'sgls']  # ['glasses', 'night_noglasses', 'nightglasses', 'noglasses', 'sunglasses']
    c2N = ['xsleepy', 'sleepy', 'slowBlink',
           'yawn']  # ['nonsleepyCombination', 'sleepyCombination', 'slowBlinkWithNodding', 'yawning']
    c3N = ['left_eye', 'mouth', 'right_eye', 'face']

    inputName = 'c3d_input' + '_' + str(fileN[person - 1]) + '_' + str(c1N[glass - 1]) + '_' + str(
        c2N[sleepy - 1]) + '_' + str(c3N[eyemouth - 1])
    labelName = 'c3d_label' + '_' + str(fileN[person - 1]) + '_' + str(c1N[glass - 1]) + '_' + str(
        c2N[sleepy - 1]) + '_' + str(c3N[eyemouth - 1])
    c3d = C3D_Preprocessing(upper_dir)
    filePath = save_dir
    inputPath = filePath + inputName
    labelPath = filePath + labelName

    print(f'저장 경로 = {filePath}')
    print(f'{inputName}, {labelName} 계산 시작 >>>>>', end=' ')
    rawinput = d.Encode_input(down_size, person, glass, sleepy, eyemouth)
    rawlabel = t.Encode_target(person, glass, sleepy)
    c3d_input = c3d.Preprocess_OnlyInput(rawinput, f, ovp)
    c3d_label = c3d.Preprocess_DemocraticLabel(rawlabel, f, ovp)
    print('계산완료.')
    print('[dimension]')
    print(f'{inputName}= {c3d_input.shape}')
    print(f'{labelName}= {c3d_label.shape}')

    print('저장 시작 >>>>>', end=' ')
    np.save(inputPath, c3d_input)
    np.save(labelPath, c3d_label)
    print(f'{filePath}에 {inputName}, {labelName} 저장 완료.\n')


def main():
    # glass = 1, 2, 3, 4, 5 = 'glasses', 'night_noglasses', 'nightglasses', 'noglasses', 'sunglasses'

    # select = 첫 피험자 번호, size = 마지막 피험자까지 길이 (0, 1, 2 ~)
    select = 0
    size = 0

    ## frame = 16, overlap = 8로 할경우 (True), 다른걸로 지정할 경우 (False)
    isPrompt = True

    frame = 0
    overlap = 0
    step = 0

    print('계산할 깊이 (Frame) 차원을 먼저 정합니다.\n'
          '(N x Width x Height X Channel) ==> (N2 x Frame x Width x Height x Channel)\n')


    while True:
        key = input('frame = 16, overlap = 8, step = 30로 빠르게 진행[y] 혹은 다르게 지정[n]? \n')
        if key == 'y':
            isPrompt = True
            frame = 16
            overlap = 8
            step = 30
            break
        elif key == 'n':
            isPrompt = False
            break
        else:
            print('다시 입력하세요\n')
            continue

    if isPrompt == False:
        while True:
            f = int(input('>>>>> 입력할 frame. 작으면 작을 수록 더 순간적인 행동을 탐지합니다. \n '
                          'Normally even number and frame <=120: \n'))
            print(f'{f} 선택함\n')
            frame = f
            if frame > 0 and frame <= 120:
                break
            else:
                print('적절하지 않는 값 입력. 다시 입력.')
                continue

        while True:
            ovp = int(input('>>>>> 입력할 overlap. 크면 클수록 한 영상에서 많은 데이터를 가져올 수 있습니다. \n '
                            f'Normally even number and overlap < frame (={frame}): \n'))
            print(f'{ovp} 선택함\n')
            overlap = ovp
            if overlap >= 0 and overlap < frame:
                break
            else:
                print('적절하지 않는 값 입력. 다시 입력.')
                continue

    ## 저장 경로 생성 및 지정
    upDir = C3D_Preprocessing(u_dir)
    filePath = upDir.MakeDir_C3D(today, frame, overlap)
    print(f'frame = {frame}, overlap = {overlap}')
    print(f'저장 경로 {filePath}\n')

    print('피험자들은 무조건 순차적으로 계산됩니다.\n')
    while True:
        for i in range(len(people)):
            if (i + 1) % 6 > 0:
                print(f'[{i + 1}]: {people[i]}', end=', ')
            else:
                print(f'[{i + 1}]: {people[i]}')
        print('\n')
        x = int(input('>>>>> 입력할 처음 피험자 번호 (1~18) '))
        y = int(input('>>>>> 입력할 마지막 피험자 번호 (처음 피험자~18)'))
        print(f'{people[x - 1]} 선택함\n')
        if x >= 1 and y >= x and y <= 18:
            select = x
            size = y - select
            print(people[x - 1], '피험자 부터 ', people[size + select - 1], ' 피험자까지')
            break
        else:
            print('해당되는 피험자 없음. 다시 숫자 입력.')
            continue

    for p in range(size + 1):
        n = select + p

        if n - 1 != 2:
            print(f'================={people[n - 1]} 저장 시작=================\n')
            for gls in range(5):
                for slp in range(4):
                    prompt_computing(person=n, glass=(gls + 1), sleepy=(slp + 1), eyemouth=em, f=frame, ovp=overlap,
                                     data_dir=d_dir, label_dir=t_dir, upper_dir=u_dir, save_dir=filePath,
                                     down_size=scale)
            print(f'================={people[n - 1]} 저장 완료=================\n')
        else:
            print(f'================={people[n - 1]} 저장 불가=================\n')


if __name__ == "__main__":
    main()
    print('모든 것을 종료 합니다.')
    # while True:
    #     main()
    #     repeat = input('<<<<다시 수행 하겠습니까[y]? 종료시 아무거나 누르세요.>>>>:\n')
    #
    #     if repeat == 'y':
    #         continue
    #     else:
    #         print('모든 것을 종료 합니다.')
    #         break