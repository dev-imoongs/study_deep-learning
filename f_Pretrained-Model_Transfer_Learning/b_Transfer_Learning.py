#!/usr/bin/env python
# coding: utf-8

# ### Transfer Learning, 전이 학습
# - 이미지 분류 문제를 해결하는데 사용했던 모델을 다른 데이터세트 혹은 다른 문제에 적용시켜 해결하는 것을 의미한다.
# - 즉, 사전에 학습된 모델(Pretrained model)을 다른 작업에 이용하는 것을 의미한다.
# - Pretrained model의 Convolutional Base구조(Conv2D + Pooling)를 그대로 두고 분류기(FC,Fully Connected Layer)를 붙여서 학습시킨다.
# <div style="display: flex; margin-left:-10px">
#     <div>
#         <img src="./images/transfer_learning01.png" width="150">
#     </div>
#     <div>
#         <img src="./images/fc.png" width="600" style="margin-top: 10px; margin-left: 50px">
#     </div>
# </div>
# 
# - 사전 학습된 모델의 용도를 변경하기 위한 층별 미세 조정은 데이터 세트의 크기와 유사성을 기반으로 고민하여 조정한다.
# - 층별로 동결 혹은 학습 결정을 위해 미세 조정을 진행할 때, 학습률이 높으면 이전 지식을 잃을 위험이 높아지므로 작은 학습률을 사용하는 것이 좋다.
# <div style="display: flex; margin-left:-30px; margin-bottom: 20px">
#     <div>
#         <img src="./images/transfer_learning02.png" width="600">
#     </div>
#     <div>
#         <img src="./images/transfer_learning03.png" width="500" style="margin-left: -80px">
#     </div>
# </div>
# 
# - 2018년 FAIR(Facebook AI Research)논문에서 실험을 통해 '전이학습이 학습 속도 면에서 효과가 있다'라는 것을 밝혀냈다.
# <img src="./images/transfer_learning04.png" width="400" style="margin-left: -30px">

# In[1]:


import numpy as np
import pandas as pd
import os


# In[2]:


from tensorflow.keras.applications import VGG16


# In[3]:


model = VGG16()
model.summary()


# In[4]:


print('model: ', model)
print('model output: ', model.output)


# In[5]:


IMAGE_SIZE = 32
BATCH_SIZE = 64


# In[6]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense , Conv2D , Dropout , Flatten , Activation, MaxPooling2D , GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau , EarlyStopping , ModelCheckpoint

base_model = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(50, activation='relu')(x)
output = Dense(10, activation='softmax', name='output')(x)

model = Model(inputs=base_model.input, outputs=output)
model.summary()


# In[7]:


import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10

# 0 ~ 1사이값의 float32로 변경하는 함수
def get_preprocessed_data(images, targets, scaling=True):
    
    # 학습과 테스트 이미지 array를 0~1 사이값으로 scale 및 float32 형 변형. 
    if scaling:
        images = np.array(images/255.0, dtype=np.float32)
    else:
        images = np.array(images, dtype=np.float32)
        
    targets = np.array(targets, dtype=np.float32)
    
    return images, targets

# 0 ~ 1사이값 float32로 변경하는 함수 호출 한 뒤 OHE 적용 
def get_preprocessed_ohe(images, targets):
    images, targets = get_preprocessed_data(images, targets, scaling=False)
    # OHE 적용 
    oh_targets = to_categorical(targets)
    return images, oh_targets

# 학습/검증/테스트 데이터 세트에 전처리 및 OHE 적용한 뒤 반환 
def get_train_valid_test_set(train_images, train_targets, test_images, test_targets, validation_size=0.2, random_state=124):
    # 학습 및 테스트 데이터 세트를  0 ~ 1사이값 float32로 변경 및 OHE 적용. 
    train_images, train_oh_targets = get_preprocessed_ohe(train_images, train_targets)
    test_images, test_oh_targets = get_preprocessed_ohe(test_images, test_targets)
    
    # 학습 데이터를 검증 데이터 세트로 다시 분리
    train_train_images, validation_images, train_train_oh_targets, validation_oh_targets = train_test_split(train_images, train_oh_targets, test_size=validation_size, random_state=random_state)
    
    return (train_train_images, train_train_oh_targets), (validation_images, validation_oh_targets), (test_images, test_oh_targets) 


# CIFAR10 데이터 재 로딩 및 Scaling/OHE 전처리 적용하여 학습/검증/데이터 세트 생성. 
(train_images, train_targets), (test_images, test_targets) = cifar10.load_data()
print(train_images.shape, train_targets.shape, test_images.shape, test_targets.shape)

(train_train_images, train_train_oh_targets), (validation_images, validation_oh_targets), (test_images, test_oh_targets) = \
    get_train_valid_test_set(train_images, train_targets, test_images, test_targets, validation_size=0.2, random_state=124)

print(train_train_images.shape, train_train_oh_targets.shape, validation_images.shape, validation_oh_targets.shape, test_images.shape, test_oh_targets.shape)


# In[8]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_generator = ImageDataGenerator(horizontal_flip=True, rescale=1./255)
validation_generator = ImageDataGenerator(rescale=1./255)

train_flow = train_generator.flow(train_train_images, train_train_oh_targets, batch_size=BATCH_SIZE, shuffle=True)
validation_flow = validation_generator.flow(validation_images, validation_oh_targets, batch_size=BATCH_SIZE, shuffle=False)


# In[9]:


def create_model(verbose=False):
    input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base_model = VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet')
    base_model_output = base_model.output
    
    x = GlobalAveragePooling2D()(base_model_output)
    x = Dense(50, activation='relu')(x)
    output = Dense(10, activation='softmax', name='output')(x)
    
    model = Model(inputs=input_tensor, outputs=output)
    
    if verbose:
        model.summary()
    else:
        pass

    return model


# In[10]:


from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

mcp_cb = ModelCheckpoint(
    filepath='./callback_files/weights.{epoch:03d}-{val_loss:.4f}.h5', 
    monitor='val_loss', 
    save_best_only=True, 
    save_weights_only=True, 
    mode='min', 
    verbose=1)

rlr_cb = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.1,
    patience=2, 
    mode='min', 
    verbose=1)

ely_cb = EarlyStopping(
    monitor='val_loss', 
    patience=4, 
    mode='min', 
    verbose=1)


# In[11]:


from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
model = create_model(verbose=True)
model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['acc'])


# In[12]:


history = model.fit(train_flow, batch_size=BATCH_SIZE, epochs=10, validation_data=validation_flow, callbacks=[mcp_cb, rlr_cb, ely_cb])


# In[ ]:


test_generator = ImageDataGenerator(rescale=1./255)
test_flow = test_generator.flow(test_images, test_oh_targets, batch_size=BATCH_SIZE, shuffle=False)
model.evaluate(test_flow)


# In[ ]:


import matplotlib.pyplot as plt

def show_history(history):
    plt.figure(figsize=(6, 6))
    plt.yticks(np.arange(0, 1, 0.05))
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='validation')
    plt.legend()
    
show_history(history)


# ##### 모듈화

# In[ ]:


import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense , Conv2D , Dropout , Flatten , Activation, MaxPooling2D , GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# 0 ~ 1사이값의 float32로 변경하는 함수
def get_preprocessed_data(images, targets, scaling=True):
    
    # 학습과 테스트 이미지 array를 0~1 사이값으로 scale 및 float32 형 변형. 
    if scaling:
        images = np.array(images/255.0, dtype=np.float32)
    else:
        images = np.array(images, dtype=np.float32)
        
    targets = np.array(targets, dtype=np.float32)
    
    return images, targets

# 0 ~ 1사이값 float32로 변경하는 함수 호출 한 뒤 OHE 적용 
def get_preprocessed_ohe(images, targets):
    images, targets = get_preprocessed_data(images, targets, scaling=False)
    # OHE 적용 
    oh_targets = to_categorical(targets)
    return images, oh_targets

# 학습/검증/테스트 데이터 세트에 전처리 및 OHE 적용한 뒤 반환 
def get_train_valid_test_set(train_images, train_targets, test_images, test_targets, validation_size=0.2, random_state=124):
    # 학습 및 테스트 데이터 세트를  0 ~ 1사이값 float32로 변경 및 OHE 적용. 
    train_images, train_oh_targets = get_preprocessed_ohe(train_images, train_targets)
    test_images, test_oh_targets = get_preprocessed_ohe(test_images, test_targets)
    
    # 학습 데이터를 검증 데이터 세트로 다시 분리
    train_train_images, validation_images, train_train_oh_targets, validation_oh_targets = train_test_split(train_images, train_oh_targets, test_size=validation_size, random_state=random_state)
    
    return (train_train_images, train_train_oh_targets), (validation_images, validation_oh_targets), (test_images, test_oh_targets) 

def get_resized_images(images, resize=64):
    image_count = images.shape[0]
    resized_images = np.zeros((image_count, resize, resize, 3))
    for i in range(image_count):
        resized_images[i] = cv2.resize(images[i], (resize, resize))
    
    return resized_images

def create_model(verbose=False):
    input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base_model = VGG16(input_tensor=input_tensor, include_top=False, weights='imagenet')
    base_model_output = base_model.output
    
    x = GlobalAveragePooling2D()(base_model_output)
    x = Dense(50, activation='relu')(x)
    output = Dense(10, activation='softmax', name='output')(x)
    
    model = Model(inputs=input_tensor, outputs=output)
    if verbose:
        model.summary()
    else:
        pass
    
    return model


# In[ ]:


IMAGE_SIZE = 32
BATCH_SIZE = 64

def train_and_evaluation(image_size=IMAGE_SIZE):
    (train_images, train_targets), (test_images, test_targets) = cifar10.load_data()
    (train_train_images, train_train_oh_targets), (validation_images, validation_oh_targets), (test_images, test_oh_targets) = \
    get_train_valid_test_set(train_images, train_targets, test_images, test_targets)
    
    print(train_train_images.shape, train_train_oh_targets.shape, validation_images.shape, validation_oh_targets.shape, test_images.shape, test_oh_targets.shape)
    
    if image_size >= 32:
        train_train_images = get_resized_images(train_train_images)
        validation_images = get_resized_images(validation_images)
        test_images = get_resized_images(test_images)
    
    else:
        pass
    
    train_generator = ImageDataGenerator(horizontal_flip=True, rescale=1./255)
    validation_generator = ImageDataGenerator(rescale=1./255)
    test_generator = ImageDataGenerator(rescale=1./255)
    
    train_flow = train_generator.flow(train_train_images, train_train_oh_targets, batch_size=BATCH_SIZE)
    validation_flow = validation_generator.flow(validation_images, validation_oh_targets, batch_size=BATCH_SIZE, shuffle=False)
    test_flow = test_generator.flow(test_images, test_oh_targets, batch_size=BATCH_SIZE, shuffle=False)
    
    model = create_model(verbose=True)
    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['acc'])
    
    mcp_cb = ModelCheckpoint(
        filepath='./callback_files/weights.{epoch:03d}-{val_loss:.4f}.h5', 
        monitor='val_loss', 
        save_best_only=True, 
        save_weights_only=True, 
        mode='min', 
        verbose=1)

    rlr_cb = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.1,
        patience=2, 
        mode='min', 
        verbose=1)

    ely_cb = EarlyStopping(
        monitor='val_loss', 
        patience=4, 
        mode='min', 
        verbose=1)
    
    history = model.fit(train_flow, batch_size=BATCH_SIZE, epochs=10, validation_data=validation_flow, callbacks=[mcp_cb, rlr_cb, ely_cb])
    
    evaluation_result = model.evaluate(test_flow)
    
    return history, evaluation_result


# In[13]:


import gc

# 불필요한 오브젝트를 지우는 작업
gc.collect()


# In[ ]:


history, evaluation_result = train_and_evaluation(image_size=64)


# In[ ]:


print('result:', evaluation_result)


# In[ ]:


import matplotlib.pyplot as plt

def show_history(history):
    plt.figure(figsize=(6, 6))
    plt.yticks(np.arange(0, 1, 0.05))
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='validation')
    plt.legend()
    
show_history(history) 

