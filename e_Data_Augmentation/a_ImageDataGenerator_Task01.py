#!/usr/bin/env python
# coding: utf-8

# ### ImageDataGenerator Task
# 
# ##### 동물 분류
# https://drive.google.com/file/d/1_d8RcCM21XneorFe_m4939erMkCnccS1/view?usp=drive_link
# 
# ### Preprocessing, Data Loading
# - Preprocessing과 Data Loading은 ImageDataGenerator객체와 model의 fit_generator()가 연결되어 수행된다.
# - model의 fit_generator()가 ImageDataGenerator 객체를 전달 받은 뒤 Pipeline Stream으로 이어지게 구성된다.
# 1. 이미지 파일을 이미지 배열로 불러오기
# 2. 전처리(preprocessing) 적용
# > - Augmentation
# > - 이미지 배열 값 scale 조정 (0 ~ 1,  float32로 형변환)
# > - 이미지 배열 크기 재조정
# > - Normalization(평균과 표준 편차 재조정)
# > - 이진 분류, 다중 분류에 따라 Encoding 진행
# > - 다중 분류일 경우 One-hot encoding, Lavbel Encoding 중 한 가지 진행
# 3. 이미지 파일을 메모리에 이미지 배열 형태로 로딩
# 4. 이미지 파일을 메모리에 로딩할 때 메모리가 부족할 수 있기 때문에 일정 크기 단위(batch)로 작업 진행
# 5. fit_generator() 호출 시, 이전 CPU 작업이 GPU로 이동 후 작업 진행
# - 실제 Preprocessing과 Data Loading은 Model을 통해 fit_generator()를 호출하기 전까지는 수행되지 않는다(대기 상태).  
# 
# 📌 CPU는 컴퓨터의 뇌로서, 연산이 어려운 일련의 작업을 처리하는 작업, 컴퓨터가 돌아가는 데에 있어서 중요한 작업을 처리하는 데에 어울리고, GPU는 양이 굉장히 많은 단순한 연산(내적, 벡터 등)을 빠르게 처리하는 데 특화되어 있는 CPU를 돕는 장치이다. 신경망 계층 또는 2D 이미지와 같은 대규모의 특정 데이터 세트에 대한 딥 러닝 훈련 작업에 어울린다. 픽셀로 이루어진 영상을 처리하는 용도로 제작되었으며, 반복적이고 비슷한, 대량의 연산을 수행하며 이를 병렬적으로 나누어 작업하기 때문에 CPU에 비해 속도가 대단히 빠르다. 그래픽 작업의 경우 CPU가 GPU로 데이터를 보내 빠르게 처리한다. CPU는 순차적인 작업, GPU는 병렬적인 작업에 특화되어 있다. CPU와 GPU의 조합, 거기에 충분한 RAM을 더하면 딥 러닝 및 AI에 알맞은 환경을 구축할 수 있다.

# In[1]:


file = open('./datasets/animals/translate.py', 'r')
content = file.readline()
content = content[content.index('{'): content.index('}') + 1]
content1 = eval(content)
content2 = {v : k for k, v in content1.items()}

file.close()

print(content1, content2, sep='\n')


# In[2]:


import os
from glob import glob

root = './datasets/animals/original/'
directories = glob(os.path.join(root, '*'))
print(directories)

for directory in directories:
    try:
        os.rename(directory, os.path.join(root, content1[directory[directory.rindex('\\') + 1:]]))
    except KeyError as e:
        os.rename(directory, os.path.join(root, content2[directory[directory.rindex('\\') + 1:]]))


# In[3]:


directories = glob(os.path.join(root, '*'))
print(directories)


# In[4]:


# 디렉토리 이름을 모두 가져오기(list 타입으로 변환)
directory_names = list(map(lambda directory: directory[directory.rindex("\\") + 1:], directories))
directory_names


# In[5]:


# 전체 파일명을 디렉토리명과 일치하게 바꾸자!(예: dog1.png, dog2.png, ...)
# os.rename(old, new)
# 1. directory_names
# 2. os.listdir()
# os.listdir(os.path.join(root, 'dog'))

root = './datasets/animals/original/'

for name in directory_names:
    for i, file_name in enumerate(os.listdir(os.path.join(root, name))):
        old_file = os.path.join(root + name + '/', file_name)
        new_file = os.path.join(root + name + '/', name + str(i + 1) + '.png')
        
        os.rename(old_file, new_file)


# In[6]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_data_generator = ImageDataGenerator(rescale=1./255)

generator = image_data_generator.flow_from_directory(
    root,
    target_size=(150, 150),
#     CPU에 나눠서 작업을 수행할 단위 작성(batch_size)
    batch_size = 20,
    class_mode = 'categorical'
)

print(generator.class_indices)


# In[7]:


import pandas as pd

animal_df = pd.DataFrame({'file_paths': generator.filepaths, 'targets': generator.classes})
animal_df.file_paths = animal_df.file_paths.apply(lambda x: x.replace('\\', '/'))
animal_df


# In[8]:


from sklearn.model_selection import train_test_split

train_images, test_images, train_targets, test_targets = train_test_split(animal_df.file_paths, animal_df.targets, stratify=animal_df.targets, test_size=0.2, random_state=124)

print(train_targets.value_counts())
print(test_targets.value_counts())


# In[9]:


# 검증 데이터 분리
train_images, validation_images, train_targets, validation_targets = train_test_split(train_images, train_targets, stratify=train_targets, test_size=0.2, random_state=124)

print(train_targets.value_counts())
print(validation_targets.value_counts())


# In[10]:


import shutil
import os.path

root = './datasets/animals/'


for filepath in train_images:
    animal_dir = filepath[filepath.find('original/') + 9:filepath.rindex('/')] + '/'
    destination = root + 'train/' + animal_dir
    
    if not os.path.exists(destination):
        os.mkdir(destination)
    shutil.copy2(filepath, destination)


# In[11]:


import shutil
import os.path

root = './datasets/animals/'


for filepath in validation_images:
    animal_dir = filepath[filepath.find('original/') + 9:filepath.rindex('/')] + '/'
    destination = root + 'validation/' + animal_dir
    
    if not os.path.exists(destination):
        os.mkdir(destination)
    shutil.copy2(filepath, destination)


# In[12]:


import shutil
import os.path

root = './datasets/animals/'


for filepath in test_images:
    animal_dir = filepath[filepath.find('original/') + 9:filepath.rindex('/')] + '/'
    destination = root + 'test/' + animal_dir
    
    if not os.path.exists(destination):
        os.mkdir(destination)
    shutil.copy2(filepath, destination)


# In[13]:


IMAGE_SIZE = 244
BATCH_SIZE = 20

train_dir = './datasets/animals/train/'
validation_dir = './datasets/animals/validation/'
test_dir = './datasets/animals/test'

train_generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, brightness_range=(0.7, 1.3), horizontal_flip=True, vertical_flip=True, rescale=1./255)
validation_generator = ImageDataGenerator(rescale=1./255)
test_generator = ImageDataGenerator(rescale=1/255.0)

train_flow = train_generator.flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_flow = validation_generator.flow_from_directory(
    validation_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_flow = test_generator.flow_from_directory(
    test_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


# In[25]:


image_array, image_target = next(train_flow)
print(image_array.shape)
print(image_target.shape)


# In[28]:


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
output = Dense(10, activation='softmax', name='output', kernel_initializer='glorot_normal')(x)

model = Model(inputs=input_tensor, outputs=output)
model.summary()


# In[29]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy

model.compile(optimizer=Adam(0.001), loss=CategoricalCrossentropy(), metrics=['acc'])


# In[30]:


from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

mcp_cb = ModelCheckpoint(filepath='./callback_files/weights.{epoch:03d}-{val_loss:.4f}.h5', monitor='val_loss', 
                         save_best_only=True, save_weights_only=True, mode='min', verbose=1)
rlr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, mode='min', verbose=1)
ely_cb = EarlyStopping(monitor='val_loss', patience=4, mode='min', verbose=1)


# In[32]:


history = model.fit_generator(train_flow, epochs=10, validation_data=validation_flow, callbacks=[mcp_cb, rlr_cb, ely_cb])


# In[ ]:


model.evaluate(test_flow)


# In[ ]:


import matplotlib.pyplot as plt

plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='validation')
plt.legend()


# 1. 이미지 데이터 불러오기
# 2. 이미지 배열로 변경
# 3. 이미지 증강(선택)
# 4. flow 구성
# 5. 모델 제작
# 6. fit_generator()
# 7. evaluate
