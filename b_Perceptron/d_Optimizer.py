#!/usr/bin/env python
# coding: utf-8

# ### Optimizer, 최적화
# - 최적의 경사 하강법을 적용하기 위해 필요하며, 최소값을 찾아가는 방법들을 의미한다.
# - loss를 줄이는 방향으로 최소 loss를 보다 빠르고 안정적으로 수렴할 수 있어야 한다.
# <img src="./images/optimizer.png" width="600" style="margin-top: 20px; margin-left: 0">
# 
# #### Momentum
# - 가중치를 계속 업데이트할 때마다 이전의 값을 일정 수준 반영시키면서 새로운 가중치로 업데이트한다.
# - 지역 최소값에서 벗어나지 못하는 문제를 해결할 수 있으며, 진행했던 방향만큼 추가적으로 더해주어, 관성처럼 빠져나올 수 있게 해준다.
# <img src="./images/momentum.png" width="600" style="margin-top: 20px; margin-left: 0">
# 
# #### AdaGrad (Adaptive Gradient)
# - 가중치 별로 서로 다른 학습률을 동적으로 적용한다.
# - 적게 변화된 가중치는 보다 큰 학습률을 적용하고, 많이 변화된 가중치는 보다 작은 학습률을 적용시킨다.
# - 처음에는 큰 보폭으로 이동하다가 최소값에 가까워질 수록 작은 보폭으로 이동하게 된다.
# - 과거의 모든 기울기를 사용하기 때문에 학습률이 급격히 감소하여, 분모가 커짐으로써 학습률이 0에 가까워지는 문제가 있다.
# <div style="display: flex">
#     <div>
#         <img src="./images/adagrad01.png" width="100" style="margin-top: 20px; margin-left: 0">
#     </div>
#     <div>
#         <img src="./images/adagrad02.png" width="400" style="margin-top: 20px; margin-left: 80px">
#     </div>
# </div>
# 
# #### RMSProp (Root Mean Sqaure Propagation)
# -  AdaGrad의 단점을 보완한 기법으로서, 학습률이 지나치게 작아지는 것을 막기 위해 지수 가중 평균법(Exponentially Weighted Average)로 구한다.
# > - 📌 지수 가중 평균법이란, 데이터의 이동 평균을 구할 때, 오래된 데이터가 미치는 영향을 지수적으로 감쇠하도록 하는 방법이다.
# - 이전의 기울기들을 똑같이 더해가는 것이 아니라 훨씬 이전의 기울기는 조금 반영하고 최근의 기울기를 많이 반영한다.
# - feature마다 적절한 학습률을 적용하여 효율적인 학습을 진행할 수 있고, AdaGrad보다 학습을 오래 할 수 있다.
# <img src="./images/rmsprop.png" width="250" style="margin-top: 20px; margin-left: 0">
# 
# #### Adam (Adaptive Moment Esimation)
# - Momentum과 RMSProp 두 가지 방식을 결합한 형태로서, 진행하던 속도에 관성을 주고, 지수 가중 평균법을 적용한 알고리즘이다.
# - 최적화 방법 중에서 가장 많이 사용되는 알고리즘이며, 수식은 아래와 같다.
# <div style="display: flex">
#     <div>
#         <img src="./images/adam01.png" width="300" style="margin-top: 20px; margin-left: 0">
#     </div>
#     <div>
#         <img src="./images/adam02.png" width="200" style="margin-top: 20px; margin-left: 80px">
#     </div>
# </div>
