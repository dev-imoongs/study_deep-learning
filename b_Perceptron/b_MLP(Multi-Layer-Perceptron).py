#!/usr/bin/env python
# coding: utf-8

# ### Multi Layer Perceptron, 다층 퍼셉트론, 다중 퍼셉트론
# - 보다 복잡한 문제의 해결을 위해서 입력층(Input Layer)과 출력층(Output Layer) 사이에 은닉층(Hidden Layer)이 포함되어 있다.
# - 퍼셉트론을 여러층 쌓은 인공 신경망으로서, 각 층에서는 활성함수를 통해 입력을 처리한다.
# - 층이 깊어질수록 정확한 분류가 가능해지지만, 너무 깊어지면 Overfitting이 발생한다.
# <div style="display: flex">
#     <div>
#         <img src="./images/MLP01.png" width="500" style="margin-left: -30px">
#     </div>
#     <div>
#         <img src="./images/MLP02.png" width="600" style="margin-top:50px; margin-left: -30px">
#     </div>
# </div>

# ### ANN (Artificial Neural Network), 인공 신경망
# - 은닉층(Hidden Layer)이 1개일 경우 이를 인공 신경망(Artificial Neural Network)이라고 한다.

# In[2]:


import mglearn
display(mglearn.plots.plot_single_hidden_layer_graph())


# ### DNN (Deep Neural Network), 심층 신경망
# - 은닉층(Hidden Layer)이 2개 이상일 경우 이를 인공 신경망(Artificial Neural Network)이라고 한다.
# - 다층 퍼셉트론 뿐만 아니라, 여러 변형된 다양한 신경망들도 은닉층이 2개 이상이 되면 심층 신경망이라고 부른다.

# In[6]:


display(mglearn.plots.plot_two_hidden_layer_graph())


# ### BackPropagation, 역전파
# - 심층 신경망에서 최종 출력(예측)을 하기 위한 식이 생기지만 식이 너무 복잡해지기 때문에 편미분을 진행하기에 한계가 있다.
# - 즉, 편미분을 통해 가중치 값을 구하고, 경사 하강법을 통해 가중치 값을 업데이트하며 손실함수의 최소값(MSE)을 찾아야 하는데,  
# 순방향으로는 복잡한 미분식을 계산할 수가 없다. 따라서 미분의 연쇄 법칙(Chain Rule)을 사용하여 역방향으로 편미분을 진행한다.
# 
# ##### 합성 함수의 미분
# <img src="./images/chain_rule01.png" width="150" style="margin-left: 0">  
# 
# ---
# <img src="./images/chain_rule02.png" width="550" style="margin-left: 0">
# 
# ##### 미분의 연쇄 법칙(Chain Rule)
# <div style="display: flex; margin-top:20px">
#     <div>
#         <img src="./images/chain_rule03.png" width="170" style="margin-left: 0">
#     </div>
#     <div>
#         <img src="./images/chain_rule04.png" width="500" style="margin-left: 50px">
#     </div>
# </div>
# 
# ---
# <img src="./images/backpropagation01.png" width="800" style="margin-left: 0">  
# <img src="./images/backpropagation02.png" width="800" style="margin-left: 0">  
# <img src="./images/backpropagation03.png" width="500" style="margin-left: 0">  
