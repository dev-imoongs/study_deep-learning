#!/usr/bin/env python
# coding: utf-8

# ### Functional API Task

# In[17]:


from tensorflow.keras.datasets import mnist

(train_images, train_targets), (test_images, test_targets) = mnist.load_data()

print(train_images.shape, train_targets.shape)
print(test_images.shape, test_targets.shape)


# In[18]:


import matplotlib.pyplot as plt

plt.imshow(train_images[0], cmap='gray')
plt.title(train_targets[0])


# In[19]:


import matplotlib.pyplot as plt

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def show_images(images, targets, ncols=8):
    figure, axs = plt.subplots(figsize=(22, 6), nrows=1, ncols=ncols)
    for i in range(ncols):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].set_title(class_names[targets[i]])


# In[20]:


show_images(train_images[:8], train_targets[:8], ncols=8)
show_images(train_images[8:16], train_targets[8:16], ncols=8)


# In[21]:


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
    
    model = Model(inputs=input_tensor, outputs = output)
    return model


# In[22]:


import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def get_preprocessed_data(images, targets):
    images = np.array(images / 255.0, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    
    return images, targets

def get_preprocessed_ohe(images, targets):
    images, targets = get_preprocessed_data(images, targets)
    oh_targets = to_categorical(targets)
    return images, oh_targets

def get_train_valid_test(train_images, train_targets, test_images, test_targets, valid_size=0.2, random_state=124):
    train_images, train_oh_targets = get_preprocessed_ohe(train_images, train_targets)
    test_images, test_oh_targets = get_preprocessed_ohe(test_images, test_targets)
    
    train_train_images, validation_images, train_train_oh_targets, validation_oh_targets = \
    train_test_split(train_images, train_oh_targets, stratify=train_oh_targets, test_size=valid_size, random_state=random_state)

    return (train_train_images, train_train_oh_targets), (validation_images, validation_oh_targets),(test_images, test_oh_targets)


# In[23]:


from tensorflow.keras.datasets import mnist

(train_images, train_targets), (test_images, test_targets) = mnist.load_data()
print(train_images.shape, train_targets.shape, test_images.shape, test_targets.shape)

(train_train_images, train_train_oh_targets), (validation_images, validation_oh_targets),(test_images, test_oh_targets) = \
get_train_valid_test(train_images, train_targets, test_images, test_targets)
print(train_train_images.shape, train_train_oh_targets.shape, validation_images.shape, validation_oh_targets.shape, test_images.shape, test_oh_targets.shape)


# In[24]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy

model = create_model()
model.summary()

model.compile(optimizer=Adam(0.001), loss=CategoricalCrossentropy(), metrics=['acc'])


# In[25]:


history = model.fit(x=train_train_images, y=train_train_oh_targets, batch_size=32, epochs=10, validation_data=(validation_images, validation_oh_targets))


# In[28]:


import matplotlib.pyplot as plt

def show_history(history):
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='validation')
    plt.legend()

show_history(history)


# In[30]:


model.evaluate(test_images, test_oh_targets, batch_size=32)


# In[34]:


pred_proba = model.predict(np.expand_dims(test_images[326], axis=0))
print('softmax output:', pred_proba)
pred = np.argmax(np.squeeze(pred_proba))
print('predicted target value:', pred)


# In[36]:


import matplotlib.pyplot as plt

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
plt.imshow(test_images[326], cmap='gray')
plt.title(class_names[int(test_targets[326])])

