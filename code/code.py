#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[5]:


classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))


# In[8]:


classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(activation = 'relu',units=128)) 
classifier.add(Dense(activation = 'sigmoid',units=1))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.summary()


# In[10]:


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[11]:


training_set = train_datagen.flow_from_directory('C:/Users/saidivya/Desktop/NA1/ass 5/Deep Learning Applications in/Deep Learning Applications in/code/Brain_tumor/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('C:/Users/saidivya/Desktop/NA1/ass 5/Deep Learning Applications in/Deep Learning Applications in/code/Brain_tumor/test/',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# In[12]:


classifier.fit_generator(training_set, steps_per_epoch=1, epochs=100, verbose=1, callbacks=None, validation_data=test_set, validation_steps=1, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)


# In[13]:


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:/Users/saidivya/Desktop/NA1/ass 5/Deep Learning Applications in/Deep Learning Applications in/code/TestImages/ring-enhancing-tumor.jpg', target_size = (64, 64))
test_image


# In[14]:


test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)
test_image


# In[15]:


result = classifier.predict(test_image)
result


# In[16]:


training_set.class_indices


# In[17]:


if result[0][0] == 0:
    prediction = 'Benign'
else:
    prediction = 'Malignent'
print("Detected tumor type is %s"%prediction)

