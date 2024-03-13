#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os


# In[2]:


from tensorflow.keras.datasets import cifar10

(train_images, train_targets), (test_images, test_targets) = cifar10.load_data()

print(train_images.shape, train_targets.shape)
print(test_images.shape, test_targets.shape)


# In[3]:


train_images[0, :, :, :], train_targets[0, :]


# In[4]:


import matplotlib.pyplot as plt

def show_images(images, targets, ncols=8):
    class_names = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
    
    figure, axs = plt.subplots(1, ncols, figsize=(22, 6))
    for i in range(ncols):
        axs[i].imshow(images[i])
#       squeeze(): 마지막 axis의 차원을 제거할 때 사용한다
        target = targets[i].squeeze()
        axs[i].set_title(class_names[target])

show_images(train_images[:8], train_targets[:8])
show_images(train_images[8:16], train_targets[8:16])


# In[5]:


from tensorflow.keras.datasets import cifar10

(train_images, train_targets), (test_images, test_targets) = cifar10.load_data()

def get_preprocessed_data(images, targets):
    images = np.array(images / 255.0, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
    return images, targets

train_images, train_targets = get_preprocessed_data(train_images, train_targets)
test_images, test_targets = get_preprocessed_data(test_images, test_targets)


# In[6]:


train_images[0, :, :, :]


# In[7]:


print(train_images.shape, train_targets.shape)


# In[8]:


train_targets = train_targets.squeeze()
test_targets = test_targets.squeeze()


# In[9]:


print(train_targets.shape, test_targets.shape)


# In[10]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense , Conv2D , Dropout , Flatten , Activation, MaxPooling2D
from tensorflow.keras.optimizers import Adam , RMSprop 
from tensorflow.keras.callbacks import ReduceLROnPlateau , EarlyStopping , ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy

IMAGE_SIZE = 32

input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

x = Conv2D(filters=32, kernel_size=5, activation='relu')(input_tensor)
x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPooling2D(2)(x)

x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(filters=64, kernel_size=3, padding='same')(x)
x = Activation('relu')(x)
x = MaxPooling2D(2)(x)

x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
x = MaxPooling2D(2)(x)

x = Flatten()(x)
x = Dropout(rate=0.5)(x)
x = Dense(300, activation='relu')(x)
x = Dropout(rate=0.3)(x)
output = Dense(10, activation='softmax')(x)

model = Model(inputs=input_tensor, outputs=output)
model.summary()


# In[11]:


from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy

model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['acc'])


# In[12]:


from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

mcp_cb = ModelCheckpoint(
    filepath='./callback_files/weights.{epoch:02d}-{val_loss:.4f}.h5', 
    monitor='val_loss',
    save_best_only=True, 
    save_weights_only=True,
    mode='min'
)

rlr_cb = ReduceLROnPlateau(
    factor=0.2,
    patience=4, 
    monitor='val_loss',
    mode='min'
)

ely_cb = EarlyStopping(
    patience=4, 
    monitor='val_loss',
    mode='min'
)

history = model.fit(x=train_images, y=train_targets, batch_size=64, epochs=30, validation_split=0.2, callbacks=[mcp_cb, rlr_cb, ely_cb])


# In[ ]:


def show_history(history):
    plt.figure(figsize=(6, 6))
    plt.yticks(np.arange(0, 1, 0.05))
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='validation')
    plt.legend()
    
show_history(history)

# 테스트 데이터로 성능 평가
model.evaluate(test_images, test_targets, batch_size=64)


# In[ ]:


pred_probas = model.predict(test_images[8:16], batch_size=64)
print('softmax outputs: ', pred_probas)

preds = np.argmax(np.squeeze(pred_probas))
print('prediected target values: ', preds)

prediected_class = np.argmax(pred_probas)
print(prediected_class)


# In[ ]:


import matplotlib.pyplot as plt

show_images(test_images[8:16], prediected_class[:8])
show_images(test_images[8:16], test_targets[8:16])

