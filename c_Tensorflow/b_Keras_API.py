#!/usr/bin/env python
# coding: utf-8

# ### Sequential API, Functional API
# 
# #### Sequential API
# - Sequential API는 간단한 모델을 구현하기에 적합하고 단순하게 층을 쌓는 방식으로 쉽고 사용하기가 간단하다.
# - Sequential API는 단일 입력 및 출력만 있으므로 레이어를 공유하거나 여러 입력 또는 출력을 가질 수 있는 모델을 생성할 수 없다.
# 
# #### Functional API
# - Functional API는 Sequential API로는 구현하기 어려운 복잡한 모델들을 구현할 수 있다.
# - 여러 개의 입력(multi-input) 및 출력(multi-output)을 가진 모델을 구현하거나,  
# 층 간의 연결이나 연산을 하는 모델을 구현할 때에는 Functional API를 사용해야 한다.
# 
# <img src="./images/functional_api.png" width="400" style="margin-left: 0">
# 

# ##### Sequential API

# In[91]:


# pip install tensorflow
# conda install -c conda-forge wrapt


# In[92]:


from tensorflow.keras.datasets import fashion_mnist

(train_images, train_targets), (test_images, test_targets) = fashion_mnist.load_data()

print(train_images.shape, train_targets.shape)
print(test_images.shape, test_targets.shape)


# In[93]:


import numpy as np

(train_images, train_targets), (test_images, test_targets) = fashion_mnist.load_data()

def get_preprocessed_data(images, targets):
    images = np.array(images / 255.0, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
    return images, targets

train_images, train_targets = get_preprocessed_data(train_images, train_targets)
test_images, test_targets = get_preprocessed_data(test_images, test_targets)

print(train_images.shape, train_targets.shape)
print(test_images.shape, test_targets.shape)


# In[94]:


INPUT_SIZE = 28

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

# 전부 Dense layer라고 부른다.
# 학습할 데이터 많으면 neuron의 개수를 더 많이 지정(예, 64, 128 등)하고 은닉층(hidden layer)수를 더 늘려도 되고,
# 학습할 데이터가 적으면 neuron 수와 은닉층 수를 줄이는 것이 좋다.
model = Sequential([
#     전체 28 * 28(784)개의 feature로 flatten 진행
#     첫 번째 Input layer
    Flatten(input_shape=(INPUT_SIZE, INPUT_SIZE)),
#     두 번째 hidden layer
    Dense(64, activation='relu'),
#      세 번째 hidden layer
    Dense(128, activation='relu'),
#     네 번째 output layer
    Dense(10, activation='softmax')
])

model.summary()


# In[95]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy

model.compile(optimizer=Adam(0.001), loss=CategoricalCrossentropy(), metrics=['accuracy'])


# In[96]:


from tensorflow.keras.utils import to_categorical

train_oh_targets = to_categorical(train_targets)
test_oh_targets = to_categorical(test_targets)

print(train_oh_targets.shape, test_oh_targets.shape)


# In[97]:


train_images.shape


# In[98]:


history = model.fit(x=train_images, y=train_oh_targets, batch_size=32, epochs=20, verbose=1)


# In[99]:


print(history.history['loss'])
print("=" * 80)
print(history.history['accuracy'])


# In[100]:


pred_proba = model.predict(np.expand_dims(test_images[0], axis=0))
print('softmax output:', pred_proba)
pred = np.argmax(np.squeeze(pred_proba))
print('predicted target value:', pred)


# In[101]:


import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.imshow(test_images[0], cmap='gray')
plt.title(str(int(test_targets[0])) + '. ' + class_names[int(test_targets[0])])


# In[102]:


model.evaluate(test_images, test_oh_targets, batch_size=32)


# ##### Validation Data
# - 훈련 데이터를 잘 맞추는 모델이 아니라, 학습에 사용하지 않은 테스트 데이터를 얼마나 잘 맞추는지가 목적이다.
# - 훈련 데이터로 모델을 만들고 검증 데이터로 성능을 평가한다.
# - 성능이 만족스럽다면, 해당 모델에 훈련 데이터와 검증 데이터를 합쳐서 학습 시킨 후 테스트 데이터를 넣어 확인한다.
# 
# <img src="./images/validation.jpg" width="500" style="margin-left: 0">

# In[103]:


import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(train_images, train_targets), (test_images, test_targets) = fashion_mnist.load_data()

def get_preprocessed_data(images, targets):
    images = np.array(images / 255.0, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
    return images, targets

train_images, train_targets = get_preprocessed_data(train_images, train_targets)
test_images, test_targets = get_preprocessed_data(test_images, test_targets)


# In[104]:


from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

train_train_images, validation_images, train_train_targets, validation_targets = train_test_split(train_images, train_targets, test_size=0.2, stratify=train_targets, random_state=124)
print(train_train_images.shape, train_train_targets.shape, validation_images.shape, validation_targets.shape)

train_train_oh_targets = to_categorical(train_train_targets)
validation_oh_targets = to_categorical(validation_targets)

print(train_train_oh_targets.shape, validation_oh_targets.shape)


# In[105]:


from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

INPUT_SIZE = 28

model = Sequential([
    Flatten(input_shape=(INPUT_SIZE, INPUT_SIZE)),
    Dense(64, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(0.001), loss=CategoricalCrossentropy(), metrics=['acc'])


# In[106]:


history = model.fit(x=train_train_images, y=train_train_oh_targets, batch_size=32, validation_data=(validation_images, validation_oh_targets), epochs=20, verbose=2)


# In[107]:


print(history.history['loss'])
print("=" * 80)
print(history.history['acc'])
print("=" * 80)
print(history.history['val_loss'])
print("=" * 80)
print(history.history['val_acc'])


# In[108]:


import matplotlib.pyplot as plt

plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='validation')
plt.legend()


# In[109]:


pred_proba = model.predict(np.expand_dims(test_images[8], axis=0))
print('softmax output: ', pred_proba)
pred = np.argmax(np.squeeze(pred_proba))
print('predicted target value: ', pred)


# In[110]:


import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.imshow(test_images[8], cmap='gray')
plt.title(str(int(test_targets[8])) + '. ' + class_names[int(test_targets[8])])


# In[111]:


model.evaluate(test_images, test_oh_targets, batch_size=32)


# ##### Functional API

# In[112]:


class Test:
    def __call__(self, data):
        return data + 10

print(Test()(20))


# In[113]:


from tensorflow.keras.layers import Layer, Input, Dense, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf

INPUT_SIZE = 28

def create_model():
    input_tensor = Input(shape=(INPUT_SIZE, INPUT_SIZE))
    x = Flatten()(input_tensor)
    x = Dense(64, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=input_tensor, outputs=output)
    return model


# In[114]:


import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist

(train_images, train_targets), (test_images, test_targets) = fashion_mnist.load_data()

def get_preprocessed_data(images, targets):
    images = np.array(images / 255.0, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
    return images, targets

def get_preprocessed_ohe(images, targets):
    images, targets = get_preprocessed_data(images, targets)
    oh_targets = to_categorical(targets)
    return images, oh_targets

def get_train_valid_test(train_images, train_targets, test_images, test_targets, validation_size=0.2, random_state=124):
    train_images, train_oh_targets = get_preprocessed_ohe(train_images, train_targets)
    test_images, test_oh_targets = get_preprocessed_ohe(test_images, test_targets)
    
    train_train_images, validation_images, train_train_oh_targets, validation_oh_targets = \
    train_test_split(train_images, train_oh_targets, stratify=train_oh_targets, test_size=validation_size, random_state=random_state)
    
    return (train_train_images, train_train_oh_targets), (validation_images, validation_oh_targets), (test_images, test_oh_targets)


# In[115]:


from tensorflow.keras.datasets import fashion_mnist

(train_images, train_targets), (test_images, test_targets) = fashion_mnist.load_data()

(train_train_images, train_train_oh_targets), (validation_images, validation_oh_targets), (test_images, test_oh_targets) = \
get_train_valid_test(train_images, train_targets, test_images, test_targets)

print(train_train_images.shape, train_train_oh_targets.shape)
print(validation_images.shape, validation_oh_targets.shape)
print(test_images.shape, test_oh_targets.shape)


# In[116]:


from tensorflow.keras.optimizers import Adam

model = create_model()
model.summary()

model.compile(optimizer=Adam(0.001), loss=CategoricalCrossentropy(), metrics=['acc'])


# In[117]:


history = model.fit(
    x=train_train_images, 
    y=train_train_oh_targets, 
    batch_size=64, 
    epochs=20, 
    validation_data=(validation_images, validation_oh_targets))


# In[118]:


import matplotlib.pyplot as plt

def show_history(history):
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='validation')
    plt.legend()

show_history(history)


# In[121]:


pred_proba = model.predict(np.expand_dims(test_images[124], axis=0))
print('softmax output: ', pred_proba)
pred = np.argmax(np.squeeze(pred_proba))
print('predicted target value: ', pred)


# In[120]:


import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.imshow(test_images[124], cmap='gray')
plt.title(str(int(test_targets[124])) + '. ' + class_names[int(test_targets[124])])


# In[122]:


model.evaluate(test_images, test_oh_targets, batch_size=64)

