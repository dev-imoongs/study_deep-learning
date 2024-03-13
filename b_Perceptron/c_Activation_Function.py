#!/usr/bin/env python
# coding: utf-8

# ### Activation Function, 활성화 함수
# - 인공 신경망에서 입력 값에 가중치를 곱한 뒤 적용하는 함수이다.
# ---
# 1. 시그모이드 함수
# - 은닉층이 아닌 최종 활성화 함수(출력층)에서 사용된다. 
# - 은닉층에서 사용 시, 입력 값이 양의 방향으로 큰 값일 경우 출력 값의 변화가 없으며, 음의 방향도 마찬가지이다.  
# 평균이 0이 아니기 때문에 정규 분포 형태가 아니고, 이는 방향에 따라 기울기가 달라져서 탐색 경로가 비효율적(지그재그)이 된다.
# <img src="./images/sigmoid.png" width="500" style="margin-bottom:30px; margin-left: 0">
# 
# ---
# 2. 소프트맥스 함수
# - 은닉층이 아닌 최종 활성화 함수(출력층)에서 사용된다.
# - 시그모이드와 유사하게 0 ~ 1 사이의 값을 출력하지만, 이진 분류가 아닌 다중 분류를 통해 모든 확률의 합이 1이 되도록 해준다.
# - 여러 개의 타겟 데이터를 분류하는 다중 분류의 최종 활성화 함수(출력층)로 사용된다.
# <img src="./images/softmax.png" width="400" style="margin-top:20px; margin-left: 0">
# 
# ---
# 3. 탄젠트 함수
# - 은닉층이 아닌 최종 활성화 함수(출력층)에서 사용된다.
# - 은닉층에서 사용 시, 시그모이드와 달리 -1 ~ 1 사이의 값을 출력해서 평균이 0이 될 수 있지만,  
# 여전히 입력 값이 양의 방향으로 큰 값일 경우 출력값의 변화가 미비하며, 음의 방향도 마찬가지이다
# <img src="./images/tanh.png" width="610" style="margin-bottom:30px; margin-left: 0">
# 
# ---
# 4. 렐루 함수
# - 대표적인 은닉층의 활성함수이다.
# - 입력 값이 0보다 작으면 출력은 0, 0보다 크면 입력값을 출력하게 된다.
# <img src="./images/relu.png" width="440" style="margin-bottom:30px; margin-left: 0">

# ### Cross Entropy
# - 실제 데이터의 확률 분포와, 학습된 모델이 계산한 확률 분포의 차이를 구하는데 사용된다.
# - 분류(classification) 문제에서 원-핫 인코딩(one-hot encoding)을 통해 사용할 수 있는 오차 계산법이다.
# <img src="./images/cross_entropy01.png" width="350" style="margin:20px; margin-left:0">
# <img src="./images/cross_entropy03.png" width="700" style="margin-left:0">
