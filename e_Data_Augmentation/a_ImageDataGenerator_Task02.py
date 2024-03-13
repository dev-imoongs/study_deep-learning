#!/usr/bin/env python
# coding: utf-8

# ### ImageDataGenerator Task02
# 
# ##### 표정 분류
# 
# https://drive.google.com/file/d/1lpwQNwijBfaSr8knSNHKWu5KUmzYWU9d/view?usp=drive_link

# In[1]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

root = './datasets/face/original/'

image_data_generator = ImageDataGenerator(rescale=1./255)
generator = image_data_generator.flow_from_directory(root, target_size=(150, 150), batch_size=20, class_mode='categorical')

print(generator.class_indices)
print(generator.classes)


# In[2]:


import pandas as pd

face_df = pd.DataFrame({'file_paths': generator.filepaths, 'targets': generator.classes})
face_df.file_paths = face_df.file_paths.apply(lambda file_path: file_path.replace('\\', '/'))
face_df


# In[3]:


from sklearn.model_selection import train_test_split

train_images, validation_images, train_targets, validation_targets = train_test_split(face_df.file_paths, face_df.targets, stratify=face_df.targets, test_size=0.2, random_state=124)

print(train_targets.value_counts())
print(validation_targets.value_counts())


# In[4]:


import shutil
import os.path

root = './datasets/face/'


for filepath in train_images:
    face_dir = filepath[filepath.find('original/') + 9:filepath.rindex('/')] + '/'
    destination = root + 'train/' + face_dir
    
    if not os.path.exists(destination):
        os.mkdir(destination)
    shutil.copy2(filepath, destination)


# In[5]:


import shutil
import os.path

root = './datasets/face/'


for filepath in validation_images:
    face_dir = filepath[filepath.find('original/') + 9:filepath.rindex('/')] + '/'
    destination = root + 'validation/' + face_dir
    
    if not os.path.exists(destination):
        os.mkdir(destination)
    shutil.copy2(filepath, destination)


# In[9]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 244
BATCH_SIZE = 64

train_dir = './datasets/face/train'
validation_dir = './datasets/face/validation/'
test_dir = './datasets/face/test/'

train_generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, brightness_range=(0.7, 1.3), horizontal_flip=True, vertical_flip=True, rescale=1./255)
validation_generator = ImageDataGenerator(rescale=1./255)
test_generator = ImageDataGenerator(rescale=1/255.0)

train_flow = train_generator.flow_from_directory(train_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, class_mode='categorical')
validation_flow = validation_generator.flow_from_directory(validation_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, class_mode='categorical')
test_flow = test_generator.flow_from_directory(test_dir, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, class_mode='categorical')


# In[11]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l1, l2

IMAGE_SIZE = 244

input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

x = Conv2D(filters=64, kernel_size=3, padding='same', kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(input_tensor)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(filters=64, kernel_size=3, padding='same', kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=2)(x)

x = Conv2D(filters=128, kernel_size=3, padding='same', kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(filters=128, kernel_size=3, padding='same', kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=2)(x)

x = Conv2D(filters=256, kernel_size=3, padding='same', kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = Conv2D(filters=256, kernel_size=3, padding='same', kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=2)(x)

x = Conv2D(filters=512, kernel_size=3, padding='same', kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

x = GlobalAveragePooling2D()(x)
x = Dropout(rate=0.5)(x)
x = Dense(300, activation='relu', kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')(x)
x = Dropout(rate=0.2)(x)
output = Dense(7, activation='softmax', name='output', kernel_initializer='glorot_normal')(x)

model = Model(inputs=input_tensor, outputs=output)
model.summary()


# In[13]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy

model.compile(optimizer=Adam(0.001), loss=CategoricalCrossentropy(), metrics=['acc'])


# In[14]:


from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

mcp_cb = ModelCheckpoint(filepath='./callback_files/weights.{epoch:03d}-{val_loss:.4f}.h5', monitor='val_loss', 
                         save_best_only=True, save_weights_only=True, mode='min', verbose=1)
rlr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, mode='min', verbose=1)
ely_cb = EarlyStopping(monitor='val_loss', patience=4, mode='min', verbose=1)


# In[15]:


history = model.fit_generator(
    train_flow,
    epochs=20, 
    validation_data=validation_flow,
    callbacks=[mcp_cb, rlr_cb, ely_cb])


# In[ ]:


model.evaluate(test_flow)


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='validation')
plt.legend()

