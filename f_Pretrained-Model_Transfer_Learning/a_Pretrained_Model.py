#!/usr/bin/env python
# coding: utf-8

# ### Pretrained Model
# - 모델을 처음부터 학습하면 오랜시간 학습을 해야한다. 이를 위해 대규모 학습 데이터 기반으로 사전에 훈련된 모델(Pretrained Model)을 활용한다.
# - 대규모 데이터 세트에서 훈련되고 저장된 네트워크로서, 일반적으로 대규모 이미지 분류 작업에서 훈련된 것을 의미한다.
# - 입력 이미지는 대부분 244 * 244 크기이며, 모델 별로 다를 수 있다.
# - 자동차나 고양이를 포함한 1000개의 클래스, 총 1400만개의 이미지로 구성된 ImageNet 데이터 세트로 사전 훈련되었다.
# <img src="./images/pretrained_model.png" style="margin-left: 0">  
# 
# ### ImageNet Large Scale Visual Recognition Challenge (ILSVRC)
# - 2017년까지 대회가 주최되었으며, 이후에도 좋은 모델들이 등장했고, 앞으로도 계속 등장할 것이다.
# - 메이저 플레이어들(구글, 마이크로소프트, 페이스북)이 만들어놓은 모델들도 등장했다.
# <img src="./images/ILSVRC.png" style="margin:20px; margin-left: 0">

# In[3]:


from tensorflow.keras.applications.vgg16 import VGG16

model = VGG16(include_top=False, weights='imagenet')
model.summary()


# In[13]:


import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import decode_predictions

model = VGG16(weights='imagenet')
image = load_img('./datasets/hamster.jpg', target_size=(224, 224))
image = img_to_array(image)

# 4차원으로 변경
image = np.expand_dims(image, axis=0)
pred = model.predict(image)

target = decode_predictions(pred)
print('{}: {:.4f}%'.format(target[0][0][1], target[0][0][2] * 100))


# In[15]:


import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import decode_predictions

model = VGG16(weights='imagenet')
image = load_img('./datasets/dog.jpg', target_size=(224, 224))
image = img_to_array(image)

# 4차원으로 변경
image = np.expand_dims(image, axis=0)
pred = model.predict(image)

target = decode_predictions(pred)

print(target)

print('{}: {:.4f}%'.format(target[0][0][1], target[0][0][2] * 100))

