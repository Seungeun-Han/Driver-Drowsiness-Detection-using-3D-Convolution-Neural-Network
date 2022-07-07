# Driver Drowsiness Detection using 3D Convolution Neural Network

이 프로젝트는 운전자 졸음 탐지를 위하여 인간로봇상호작용연구실에서 개발한 딥러닝 SW 입니다.
주요 기능 및 특징은 다음과 같습니다.

- 기능 : 운전자 졸음 탐지
- 특징 : 운전자의 현재 상태 뿐만 아니라 연속적인 행동 흐름도 파악하여 더 빠르고 정확한 졸음 탐지 가능

<br>

## Table of Contents
- [Prerequisites](#prerequisites)
- [Data](#data)
- [Training](#training)
- [Inference](#inference)
- [Authors](#authors)

<br>

## Prerequisites

```
* 소스코드를 실행하기 위해 필요한 라이브러리/프레임워크, 데이터셋, 컴퓨팅 환경을 명시
```

- ML 프레임워크 : Tensorflow 2.6.0, Scikit-Learn
- 학습 데이터셋 : NTHU-DDD
- 컴퓨팅 환경 : NVIDIA RTX A6000

<br>

## Data

#### 데이터셋 설명(필수)

**데이터셋 내용 및 구조**

(예) 이미지 인식을 위한 데이터셋은 오픈 데이터셋인 Cifar10을 사용합니다.

(예) 이미지 인식을 위한 데이터셋은 연구사업에서 자체적으로 구축한 데이터셋입니다. 데이터 구조(메타데이터)는 다음과 같습니다.

<br>

**데이터셋 소스 정보**

(예) 데이터셋 다운로드 url :  https://www.cs.toronto.edu/~kriz/cifar.html 

(예) 데이터셋 보유 부서 및 연락처 : <부서명> <email 주소> <전화번호> 

<br>

#### 데이터 처리 방법(옵션)

**데이터 준비, 전처리 절차**

(예) 이미지 데이터셋의 준비, 전처리 절차는 다음과 같습니다 

1. 데이터셋 소스 정보를 참조하여 데이터셋을 다운로드 받는다
2. 데이터와 레이블은 별개의 Hive 테이블로 분류하여 저장한다
3. JSON 구조의 메타정보로 이미지 연관 정보를 저장한다. json 파일명은 metadata.json
4. 데이터셋은 학습, 검증, 테스트 데이터셋으로 분류되어 있다. 학습 데이터셋은 50,000개, 테스트 데이터셋은 10,000개 데이터셋으로 구성된다

<br>

**코드 실행 방법**

(예) 데이터 처리를 위한 Python 코드의 파일명은 dataproc.py 입니다.  코드 실행 절차는 다음과 같습니다.
* 주피터 노트북에서 실행 방법 또는
* Python 스크립트로 실행 방법 또는
* IDE(PyCharm)에서 실행 방법 또는
* Docker 이미지의 실행 방법

<br>

## Training

#### 학습 및 최적화 방법(필수)

**학습 방법**

(예) 이미지 인식을 위한 학습은 기존 CNN 모델인 ResNet, DenseNet을 활용하여 수행하였습니다. 학습 방법 및 모델 구조는 다음과 같습니다.

<br>

**코드 실행 방법**

(예) 모델 학습을 위한 Python 코드의 파일명은 train.py 입니다.  코드 실행 절차는 다음과 같습니다.

- 주피터 노트북에서 실행 방법 또는
- Python 스크립트로 실행 방법 또는
- IDE(PyCharm)에서 실행 방법 또는
- Docker 이미지의 실행 방법

<br>

**하이퍼파라미터**

(예) 학습 결과로 얻은 최적화된 하이퍼파라미터를 설명합니다.

* Number of hidden units
* Optimizer information(e.g. stochastic gradient descent, momentum, Adam, RMSProp, etc...)
* Learning rate, Mini batch size, Number of epochs

<br>

**학습모델**

학습된 모델 파일을 설명합니다. 

<br>

#### 성능 지표(옵션)

**목표 성능**
예) Precision, Recall, Accuracy, AUC, MAE, MSE, RMSE 등

<br>

**코드 설명**

성능을 평가하기 위한 코드 및 절차를 설명합니다.

<br>

## Inference

####  추론 방법(필수)

(예) 이미지 인식을 위한 학습 모델을 평가하기 위한 방법 및 추론 구조는 다음과 같습니다.

<br>

**코드 실행 방법**

(예) 새로운 데이터로 모델 테스트(추론)를 위한 Python 코드의 파일명은 inference.py 입니다.  코드 실행 절차는 다음과 같습니다.
* 주피터 노트북에서 실행 방법 또는
* Python 스크립트로 실행 방법 또는
* IDE(PyCharm)에서 실행 방법 또는
* Docker 이미지의 실행 방법

<br>

#### 직렬화(옵션)

학습모델을 직렬화(serialization)하는 방법을 설명합니다. 학습모델의 포맷과 배포하는 모델의 포맷이 다를 경우에만 명시합니다(예. TFLite를 활용한 모델 경량화, ONNX 표준 포맷으로 변환)

<br>

## Authors
```
* 개발에 참여한 개발자 정보 작성 
* 개발자 정보 표기방법:  <성명>  <이메일 주소>   
```
* (예) 	홍길동 &nbsp;&nbsp;&nbsp;  aaa@etri.re.kr   

<br>
<br>
<br>


## Version (optional)
```
* 현재 버전 정보 입력 (예) major.minor.patch (1.0.0) / release date (2021.10.30) 
* 이전 버전 정보 및 관련 url 정보 
```
<br>
<br>


## Thanks (optional)
```
* 프로젝트 개발에 도움을 준 사람 또는 타 프로젝트 정보 입력  
```

<br>
