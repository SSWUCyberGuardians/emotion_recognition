# opencv_emotion_recognition

## CNN 활용 얼굴 감정 인식
### 1. Haar Cascade 기반 얼굴 감지
 감정 인식이 이뤄지려면 먼저 주어진 자료에서 얼굴을 인식한 후 눈, 코, 입, 얼굴의 모습에 따른 감정을 분석해야 한다.
<img width="592" alt="BMoqsvnGTz_V7D1sVeFgVn5Cmq6_Pzjw9spX38cb3S7SfgcmcL4-RkGTIbFDzzrpSiMvzi-CAm4y5a_3SGii7ckzz2IqsBVxlwbweSE0Ol9H9DytHTetLbGutjOzXs4XlT9j8l7f-cBOkliV6-7eQx4" src="https://github.com/oIfloraIo/opencv_emotion_recognition/assets/102645357/d1d02901-27ae-415f-a26b-b6e73235ea69">

OpenCV의 CascadeClassifier를 이용하여 Haar Cascade 기반 얼굴 감지를 수행한다.

![jIKuFPe5cBJzUKyNQXST39ns6h8j3fJQNMPaNmMUdDrvFnm0cmY1KfyRGQ60o4Or0ZVu3gvTmGdjFs_jPUbYN73DhDfan-2srlBwm70ZyrYtsldP4L_vLJOdOvjotkMckdhWf4byT88FF6sbvmYgQ9Y](https://github.com/oIfloraIo/opencv_emotion_recognition/assets/102645357/f9ed2941-d206-4dfb-b856-39b41c6d4a6b)

Haar Cascade는 여러 개의 Haar-like 특징으로 표현한다. Haar-like 특징은 사각형 영역의 밝기 차이를 계산하여 특징 값을 생성하는 방식이며 Haar Cascade는 각 단계마다 다른 크기의 Haar-like 특징을 사용하여 눈,코,입 등을 필터링하고, 인식된 물체의 특징을 감지해 얼굴 인식이 가능한 알고리즘이다.

### 2. CNN
얼굴 감정 인식은 합성곱 신경망(CNN, Convolutional Neural Network) 딥러닝 모델을 활용하고 있다. CNN은 이미지 처리 작업에 주로 사용되는 딥러닝 아키텍처 중 하나이며 이미지 내 특징을 추출하고 패턴을 학습하여 이미지를 분류하거나 예측하는 데 사용된다. 

![FeoXEcodIKxIV5bwRZ_sJysFYzwMcstFuYcp9EaoT3iN4YQQYLYMr123RD3Kwd-Goptm-t80tV2BuStlxw9uPJns_VJXMlgGj6TFVFUUFOr3yhnQgKoN1UNdyvNDQcHZ19NGZRrkSTGLANLd012CS0w](https://github.com/oIfloraIo/opencv_emotion_recognition/assets/102645357/4f480925-5af2-4cd4-8ffd-3bb6a2b3b0da)

합성곱 신경망의 다양한 종류 중 mini_XCEPTION은 CNN의 구조 중 하나로 Inception 모듈과 Depthwise Separable Convolution을 조합하여 작은 크기의 모델을 만들어 이미지 분류 작업에 활용하며 주로 작은 이미지 크기와 작은 모델 크기가 필요한 얼굴 표정 인식을 통한 감정 분류나 비슷한 이미지 분류 작업에 쓰이는 구조이다.

### 3. mini XCEPTION을 활용한 감정 인식 모델 학습
#### 가. FER2013 데이터셋 전처리
감정 인식 모델 학습 과정에서 사용된 “FER2013” 데이터셋은 공개 데이터셋으로, 총 7개의 감정을 기준으로 감정 표현을 나타내는 얼굴 이미지와 데이터 셋에 대한 감정 레이블로 구성되어 있다.

![CzA56m8-inIzDZ9lds7AXpiTmu-rW6pETnvOLFvKpGD1wmkpfmXOC6U2D9v1DzlheSLW98-sgCFKasfqt18kQFlkOibh7dmaMj0tUiop9mTnEOB9gd8_ZRlAcVgg3N5CAkNqo4Bljkac_4dDmuL1kvk](https://github.com/oIfloraIo/opencv_emotion_recognition/assets/102645357/1b9b235f-4542-4540-b96a-83bf80598ac6)

FER2013 데이터셋 파일을 읽어와 각 이미지의 픽셀 값을 문자열로 추출해 배열로 변환하고, 크기를 동일하게 변환한다.

각 이미지를 실수형으로 변환하고 원-핫 인코딩을 통해 수치형 데이터로 변환하여 모델이 활용 가능하도록 한다. 이미지 데이터를 0-1 사이의 값으로 정규화 한 후, 조건 만족 시 데이터를 -1에서 1 사이의 값으로 재조정하여 모델 적용에 용이하게 한다.

#### 나. mini XCEPTION 정의
mini XCEPTION 모델은 주어진 입력에 맞게 컨볼루션과 풀링 레이어를 나타내고 있다. 마지막 출력 레이어에서는 클래스 수에 맞게 뉴런 수가 설정되고 소프트맥스 활성화 함수가 사용되어 클래스별 확률을 출력한다. 

이 모델은 주로 작은 이미지에서의 감정 인식에 사용되며, 실제로 데이터를 이용해 학습하고 평가하는 등의 과정을 거쳐 모델을 훈련시킨다.

구성된 mini_XCEPTION 모델은 입력 이미지의 특징을 추출하고, 이를 기반으로 각 클래스(감정)에 대한 확률 분포를 출력하는 분류 모델이다. 이 모델은 얼굴 감정 분류 작업에 사용되며, 각 모듈에서의 특징 추출과 잔차 연결은 모델의 성능을 향상시킨다. 

#### 다. mini XCEPTION을 활용한 모델 학습

<img width="657" alt="179NoA5sNcrUlokBfI99OSf-lJ6F9I34lTqoqvhlfkxY_CK1mvO5hxzBPNqSIGPC0KCbB1Ttu4Bh0nQH0hk4j0faexfTygNmExGNjVgl_GOlVGQZXcyZJrZRegn1Gwh8H59XJZif8ZDNyxhFtc0T0GM" src="https://github.com/oIfloraIo/opencv_emotion_recognition/assets/102645357/c68adcf0-dbe1-4644-a788-e1d78a3b73ee">

ImageDataGenerator 클래스를 사용하여 데이터 생성기를 하고, 데이터 생성기는 이미지 데이터를 적절한 형식으로 읽고 데이터 증강 기능을 적용한다. 

데이터 증강 기능은 이미지를 무작위로 변환하거나 반전시켜서 학습 데이터의 다양성을 높인다.

<img width="410" alt="TZ9W23HRhqz_IhLxvWQpravXkcQitJVkYfOw8Fn7d8i3WIjjOUeATWkZ2NTnzyuLDSYKwP4RoZEqINMxW-E4LGk1_2yYn421qgnRE7AdXvzXJLP7lA9c5mPq3m4zrZEKV-YSS1N6Hz0SzOoKnuUpvVY" src="https://github.com/oIfloraIo/opencv_emotion_recognition/assets/102645357/7391336d-4a8c-44dd-a37a-142f1c4734bb">

모델 컴파일 과정에서는 가중치, 손실 함수, 평가 지표 등을 설정하여 컴파일이 이뤄진다. 가중치를 업데이트하는 방식으로 'Adam' 옵티마이저, 모델이 얼마나 정확하게 예측하는 지를 평가하는 손실 함수는 'categorical_crossentropy', 모델의 평가 지표로는 정확도를 측정한다.


참고
https://arxiv.org/pdf/2003.01791.pdf
https://github.com/omar-aymen/Emotion-recognition/tree/master
