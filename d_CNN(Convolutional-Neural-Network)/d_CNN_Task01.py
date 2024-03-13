#!/usr/bin/env python
# coding: utf-8

# ### CNN Task01
# 
# ##### 동물 분류
# 
# https://drive.google.com/file/d/1_d8RcCM21XneorFe_m4939erMkCnccS1/view?usp=drive_link

# In[1]:


file = open('./datasets/animals/translate.py', 'r')
content = file.readline()
content = content[content.index('{'): content.index('}') + 1]
content1 = eval(content)
content2 = {v : k for k, v in content1.items()}

file.close()

print(content1, content2, sep='\n')


# In[3]:


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


# In[4]:


directories = glob(os.path.join(root, '*'))
print(directories)


# In[8]:


# 디렉토리 이름을 모두 가져오기(list 타입으로 변환)
directory_names = list(map(lambda directory: directory[directory.rindex("\\") + 1:], directories))
directory_names


# In[12]:


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


# In[13]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_data_generator = ImageDataGenerator(rescale=1./255)

generator = image_data_generator.flow_from_directory(
    root,
    target_size=(150, 150),
    batch_size = 20,
    class_mode = 'categorical'
)

print(generator.class_indices)


# In[24]:


import pandas as pd

animal_df = pd.DataFrame({'file_paths': generator.filepaths, 'targets': generator.classes})
animal_df.file_paths = animal_df.file_paths.apply(lambda x : x.replace('\\', '/'))
animal_df.file_paths[0]


# In[25]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(animal_df.file_paths, animal_df.targets, stratify=animal_df.targets, test_size=0.2, random_state=124)

print(y_train.value_counts())
print(y_test.value_counts())


# In[27]:


train_images, validation_images, train_targets, validation_targets = \
train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=124)

print(y_train.value_counts())
print(train_targets.value_counts())
print(validation_targets.value_counts())


# In[28]:


import shutil
import os.path

root = './datasets/animals/'

for filepath in train_images:
#     './datasets/animals/original/butterfly/butterfly1.png'
    animal_dir = filepath[filepath.find('original/') + 9:filepath.rindex('/') + 1]
    destination = os.path.join(root, 'train/' + animal_dir)
    
    if not os.path.exists(destination):
        os.mkdir(destination)
    
    shutil.copy2(filepath, destination)


# In[37]:


# valid copy
import shutil
import os.path

root = './datasets/animals/'

for filepath in validation_images:
#     './datasets/animals/original/butterfly/butterfly1.png'
    animal_dir = filepath[filepath.find('original/') + 9:filepath.rindex('/') + 1]
    destination = os.path.join(root, 'validation/' + animal_dir)
    
    if not os.path.exists(destination):
        os.mkdir(destination)
    
    shutil.copy2(filepath, destination)


# In[38]:


# test copy
# valid copy
import shutil
import os.path

root = './datasets/animals/'

for filepath in X_test:
#     './datasets/animals/original/butterfly/butterfly1.png'
    animal_dir = filepath[filepath.find('original/') + 9:filepath.rindex('/') + 1]
    destination = os.path.join(root, 'test/' + animal_dir)
    
    if not os.path.exists(destination):
        os.mkdir(destination)
    
    shutil.copy2(filepath, destination)

