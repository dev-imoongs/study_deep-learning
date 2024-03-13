#!/usr/bin/env python
# coding: utf-8

# ### CNN_Task02
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

for file_path in train_images:
    face_dir = file_path[file_path.find('original/') + 9: file_path.rindex('/') + 1]
    destination = os.path.join(root + 'train/', face_dir)
    
    if not os.path.exists(destination):
        os.mkdir(destination)
    
    shutil.copy2(file_path, destination)


# In[5]:


import shutil
import os.path

root = './datasets/face/'

for file_path in validation_images:
    face_dir = file_path[file_path.find('original/') + 9: file_path.rindex('/') + 1]
    destination = os.path.join(root + 'validation/', face_dir)
    
    if not os.path.exists(destination):
        os.mkdir(destination)
    
    shutil.copy2(file_path, destination)

