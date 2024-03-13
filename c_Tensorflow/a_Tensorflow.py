#!/usr/bin/env python
# coding: utf-8

# ### Tensorflow, 텐서플로우
# - 구글이 개발한 오픈소스 소프트웨어 라이브러리이며, 머신러닝과 딥러닝을 쉽게 사용할 수 있도록 다양한 기능을 제공한다.
# - 주로 이미지 인식이나 반복 신경망 구성, 기계 번역, 필기 숫자 판별 등을 위한 각종 신경망 학습에 사용된다.
# - 딥러닝 모델을 만들 때, 기초부터 세세하게 작업해야 하기 때문에 진입장벽이 높다.
# <img src="./images/tensorflow.png" width="600" style="margin-left: -20px">
# 
# ### Keras, 케라스
# - 일반 사용 사례에 최적화된 간단하고 일관된, 단순화된 인터페이스를 제공한다.
# - 손쉽게 딥러닝 모델을 개발하고 활용할 수 있도록 직관적인 API를 제공한다.
# - 텐서플로우 2버전 이상부터 케라스가 포함되었기 때문에 텐서플로우를 통해 케라스를 사용한다.
# - 기존 Keras 패키지 보다는 이제 Tensorflow에 내장된 Keras 사용이 더 권장된다.
# <img src="./images/keras.png" width="600" style="margin-left: -20px">  
# 
# > - Keras API  
# > https://keras.io/api/
# 
# ### Grayscale, RGB
# - 흑백 이미지와 컬러 이미지는 각 2차원과 3차원으로 표현될 수 있다.
# - 흑백 이미지는 0 ~ 255를 갖는 2차원 배열(Height x Width)이고,  
# 컬러 이미지는 0 ~ 255를 갖는 R, G, B 2차원 배열 3개를 갖는 3차원 배열(Height x Width x Channel)이다.
# <div style="display: flex; margin-top:20px;">
#     <div>
#         <img src="./images/grayscale.png" width="300" style="margin-left: -20px">
#     </div>
#     <div>
#         <img src="./images/rgb.png" width="280" style="margin-left: 50px">
#     </div>
# </div>
# 
# ### Grayscale Image Matrix
# - 검은색에 가까운 색은 0에 가깝고 흰색에 가까우면 255에 가깝다.
# - 모든 픽셀이 feature이다.
# <img src="./images/matrix.png" width="500" style="margin-top: 20px; margin-left: 0">

# In[2]:


from tensorflow.keras.datasets import fashion_mnist

(train_images, train_targets), (test_images, test_targets) = fashion_mnist.load_data()

print("train dataset shape", train_images.shape, train_targets.shape)
print("test dataset shape", test_images.shape, test_targets.shape)


# In[5]:


import matplotlib.pyplot as plt

plt.imshow(train_images[0], cmap='gray')
plt.title(train_targets[0])


# In[12]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

figure, axs = plt.subplots(1, 8, figsize=(22, 6))
for i in range(8):
    axs[i].imshow(train_images[:8][i], cmap='gray')
    axs[i].set_title(class_names[train_targets[:8][i]])

plt.show()
    
figure, axs = plt.subplots(1, 8, figsize=(22, 6))
for i in range(8):
    axs[i].imshow(train_images[8:16][i], cmap='gray')
    axs[i].set_title(class_names[train_targets[8:16][i]])

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




