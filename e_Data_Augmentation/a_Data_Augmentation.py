#!/usr/bin/env python
# coding: utf-8

# ### Data Augmentation, 데이터 증강
# - 이미지의 종류와 개수가 적으면, CNN 모델의 성능이 떨어질 수 밖에 없다. 또한 몇 안되는 이미지로 훈련시키면 해당 이미지에 과적합이 발생한다.
# - CNN 모델의 성능을 높이고 과적합을 개선하기 위해서는 훈련 이미지가 다양하고 많이 있어야 한다. 즉, 데이터의 양을 늘려야 한다.
# - 이미지 데이터는 학습 데이터를 수집하여 양을 늘리기 쉽지 않기 때문에, 원본 이미지를 변형 시켜서 양을 늘릴 수 있다.
# - Data Augmentation을 통해 원본 이미지에 다양한 변형을 주어서 학습 이미지 데이터를 늘리는 것과 유사한 효과를 볼 수 있다.
# - 원본 학습 이미지의 개수를 늘리는 것이 아닌 매 학습 마다 개별 원본 이미지를 변형해서 학습을 수행한다.
# 
# <img src="./images/data_augmentation.png" width="400" style="margin-left: 30px">
# 
# #### 공간 레벨 변형
# - 좌우 또는 상하 반전, 특정 영역만큼 확대, 회전 등으로 변형시킨다.
# <img src="./images/spatial.png" width="500" style="margin-left: -10px">
# 
# #### 픽셀 레벨 변형
# - 밝기, 명암, 채도, 색상 등을 변형시킨다.
# <img src="./images/pixel.png" width="200" style="margin-left: 0">

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


# In[51]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


root = './datasets/nangman22/'

image_data_generator = ImageDataGenerator(rescale=1./255)


generator = image_data_generator.flow_from_directory(
    root,
    target_size=(150, 150),
    batch_size = 20,
    class_mode = 'categorical'
)

print(generator.class_indices)


# In[52]:


generator.filepaths


# In[53]:


# conda install -c conda-forge opencv
import cv2
import matplotlib.pyplot as plt

image = cv2.cvtColor(cv2.imread(generator.filepaths[0]), cv2.COLOR_BGR2RGB)

def show_image(image):
    plt.figure(figsize=(3, 3))
    plt.imshow(image)
    plt.axis('off')
    
show_image(image)


# ##### Data Augmentation, 위치 반전(상하 좌우)

# In[8]:


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Horizontal Filp(좌우 반전)을 적용
# horizontal_flip=True로 적용했지만 반드시 변환되는 것이 아니다!
# 랜덤하게 원본데이터를 유지하거나 반전된 데이터로 사용된다.
data_generator = ImageDataGenerator(horizontal_flip=True)


# In[54]:


image_batch = np.expand_dims(image, axis=0)
print(image_batch.shape)

data_generator.fit(image_batch)
data_gen_iter = data_generator.flow(image_batch)

aug_image_batch = next(data_gen_iter)
aug_image = np.squeeze(aug_image_batch)
print(aug_image.shape)


# In[55]:


aug_image.astype('int')


# In[56]:


#augmentation이 적용된 image들을 시각화 해주는 함수
def show_aug_image(image, generator, n_images=4):
	
    # ImageDataGenerator는 여러개의 image를 입력으로 받기 때문에 4차원으로 입력 해야함.
    image_batch = np.expand_dims(image, axis=0)
	
    # featurewise_center or featurewise_std_normalization or zca_whitening 가 True일때만 fit 해주어야함
    generator.fit(image_batch) 
    # flow로 image batch를 generator에 넣어주어야함.
    data_gen_iter = generator.flow(image_batch)

    fig, axs = plt.subplots(nrows=1, ncols=n_images, figsize=(24, 8))

    for i in range(n_images):
    	#generator에 batch size 만큼 augmentation 적용(매번 적용이 다름)
        aug_image_batch = next(data_gen_iter)
        aug_image = np.squeeze(aug_image_batch)
        aug_image = aug_image.astype('int')
        axs[i].imshow(aug_image)
        axs[i].axis('off')
        
data_generator = ImageDataGenerator(horizontal_flip=True)
show_aug_image(image, data_generator, n_images=4)


# In[62]:


data_generator = ImageDataGenerator(shear_range=60)
show_aug_image(image, data_generator, n_images=4)


# In[58]:


data_generator = ImageDataGenerator(width_shift_range=0.4, fill_mode='nearest')
show_aug_image(image, data_generator, n_images=4)


# ### Data Augmentation, 회전

# In[59]:


data_generator = ImageDataGenerator(rotation_range=45)
show_aug_image(image, data_generator, n_images=4)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:




