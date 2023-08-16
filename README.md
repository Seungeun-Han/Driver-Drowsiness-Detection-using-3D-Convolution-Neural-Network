# Driver Drowsiness Detection using 3D Convolution Neural Network

이 프로젝트는 운전자 졸음 탐지를 위하여 인간로봇상호작용연구실에서 개발한 딥러닝 SW 입니다.
주요 기능 및 특징은 다음과 같습니다.

- 기능 : 운전자 졸음 탐지
- 특징 : 운전자의 현재 상태 뿐만 아니라 연속적인 행동 흐름도 파악하여 더 빠르고 정확한 졸음 탐지 가능

<br>
## paper
[Driver Drowsiness Detection based on 3D Convolution Neural Network with Optimized Window Size](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9952988)
N. Kang et al., "Driver Drowsiness Detection based on 3D Convolution Neural Network with Optimized Window Size," 2022 13th International Conference on Information and Communication Technology Convergence (ICTC), Jeju Island, Korea, Republic of, 2022, pp. 425-428, doi: 10.1109/ICTC55196.2022.9952988.

<br>

## Table of Contents
- [Prerequisites](#prerequisites)
- [Data](#data)
- [Training](#training)
- [Inference](#inference)
- [Results](#results)
- [Authors](#authors)

<br>

## Prerequisites

- ML 프레임워크 : Tensorflow 2.6.0, Scikit-Learn
- 학습 데이터셋 : NTHU-DDD
- 컴퓨팅 환경 : NVIDIA RTX A6000 48GB

<br>

## Data

**데이터셋 내용 및 구조**

운전자 졸음 탐지를 위한 데이터셋은 NTHU-DDD를 사용합니다.

위 데이터셋은 라이센스 동의서를 해당 기관에 보내야 얻을 수 있습니다.  데이터 구조(메타데이터)는 다음과 같습니다.

<br>

**데이터셋 소스 정보**

데이터셋 다운로드 url :  http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/

데이터셋 보유 부서 및 연락처 : <인간로봇상호작용연구실> <hse@etri.re.kr>

<br>

#### 데이터 처리 방법

**데이터 준비, 전처리 절차**

NTHU-DDD 데이터셋의 준비, 전처리 절차는 다음과 같습니다 

1. 데이터셋 소스 정보를 참조하여 데이터셋 라이센스 동의서를 작성해 해당 기관 이메일로 보낸 후, 다운로드 서버를 취득한다.(약 2주 소요)
2. 데이터셋은 학습, 검증, 테스트 데이터셋으로 분류되어 있다. 학습 데이터셋은 18명의 피실험자 360개 동영상, 검증 데이터셋은 4명의 피실험자 16개 동영상, 테스트 데이터셋은 14명의 피실험자 60개의 동영상으로 구성된다.
3. 동영상을 프레임 별로 읽고 Gamma Correction, Contrast Normalization 을 수행한다.
4. 눈, 입을 64x64로 검출하여 이미지로 저장한다.
5. 이미지를 설정한 temporal-depth 만큼씩 가져와 (W x H x T) 형태의 데이터셋으로 재구성한다.
6. 재구성된 데이터셋을 하나의 .npy 파일로 만들어 학습 데이터를 생성한다.

<br>

**코드 실행 방법**

데이터 처리를 위한 Python 코드의 파일명은 preprocessing/make_images.py, preprocessing/to_npy.py 그리고 preprocessing/concat.py 입니다.
코드 실행 절차는 다음과 같습니다.

1. IDE(PyCharm)에서 make_images.py 실행하여 이미지 생성
2. IDE(PyCharm)에서 to_npy.py 실행하여 (W x H x T) 형태의 데이터셋으로 재구성
3. IDE(PyCharm)에서 concat.py 실행하여 concatenate

* 현재 소스코드에는 데이터셋 경로를 절대 경로로 작성하였습니다. 절대 경로 부분 표시를 위해 "필요시 수정" 등의 문구로 표기를 해놨습니다.
예를 들어, preprocessing/to_npy.py 의 9~16번째 라인을 본인에 맞는 데이터셋 경로로 수정해야 합니다.

<br>

## Training

#### 학습 및 최적화 방법(필수)

**학습 방법**

운전자 졸음 탐지를 위한 학습은 기존 3DCNN을 활용하여 수행하였습니다. 학습 방법 및 모델 구조는 다음과 같습니다.

![image](https://user-images.githubusercontent.com/101082685/177717220-0e771f94-a0b8-4e48-99e1-30df587864d2.png)

<br>

**코드 실행 방법**

모델 학습을 위한 Python 코드의 파일명은 train.py 입니다. 코드 실행 절차는 다음과 같습니다.

- IDE(PyCharm)에서 train.py 실행

<br>

**하이퍼파라미터**

(예) 학습 결과로 얻은 최적화된 하이퍼파라미터를 설명합니다.

* temporal-depth: 10 frame
* overlap: 5 frame
* Optimizer: Adam
* Mini batch size: 32 or 64
* Number of epochs: 100

<br>

**학습모델**

학습된 모델 파일을 설명합니다. 

<br>

## Inference

운전자 졸음 탐지를 위한 학습 모델을 평가하기 위한 방법은 다음과 같습니다.

<br>

**코드 실행 방법**

새로운 데이터로 모델 테스트(추론)를 위한 Python 코드의 파일명은 inference.py 입니다. 코드 실행 절차는 다음과 같습니다.

* IDE(PyCharm)에서 해당 데이터셋애 대한 경로 설정 후 inference.py 실행

<br>

## Results
- Accuracy

![0526_ubuntu_model_accuracy](https://user-images.githubusercontent.com/101082685/221487863-f6e02227-29d2-4149-b4f3-a0419aa2f8cd.png)


<br>

## Authors

한승은 &nbsp;&nbsp;&nbsp;  hse@etri.re.kr

<br>
<br>
<br>

