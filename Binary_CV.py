# -*- coding: utf-8 -*-
"""
Created on December 2024

@author: Yu Gan
"""
import os 
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf 
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet import ResNet50,ResNet101
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from tensorflow.keras.layers import Conv2D, BatchNormalization , Activation, MaxPool2D , Dropout, Flatten, Dense,Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
np.set_printoptions(threshold= np.inf)
#%%
# def generateds(path, txt):
#     f  = open(txt, 'r')   
#     contents = f.readlines()
#     f.close()
#     x, y = [], []
#     for content in tqdm(contents,desc = 'loading data'):
#         value = content.split()
#         img_path = path + value[0]
#         img = Image.open(img_path)
#         img = img.resize((224,224),Image.ANTIALIAS)
#         img = np.array(img)# 变为np.array格式
#         img = img / 255
#         x.append(img)
#         y.append(float(value[1]))
        
    
#     x = np.array(x)
#     y = np.array(y)
#     # y = np.reshape(y, (len(y),1))
#     return x, y

# train_path = '/public/home/ganyu/deep_learning/data/binnary_classification/total_image_cv/'
# train_txt = '/public/home/ganyu/deep_learning/data/binnary_classification/total_image_cv/label.txt'
# x_train_savepath = '/public/home/ganyu/deep_learning/data/binnary_classification/total_image_cv/x_train_cv.npy' 
# y_train_savepath = '/public/home/ganyu/deep_learning/data/binnary_classification/total_image_cv/y_train_cv.npy'
# test_path = '/public/home/ganyu/deep_learning/data/binnary_classification/test/'
# test_txt = '/public/home/ganyu/deep_learning/data/binnary_classification/test/label.txt'
# x_test_savepath = '/public/home/ganyu/deep_learning/data/binnary_classification/test/x_test.npy'
# y_test_savepath = '/public/home/ganyu/deep_learning/data/binnary_classification/test/y_test.npy'
# if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(x_test_savepath) and os.path.exists(y_test_savepath):
#     print('--------------------------Load Datasets--------------------------')
#     x_train = np.load(x_train_savepath)
#     y_train = np.load(y_train_savepath)
#     x_test = np.load(x_test_savepath)
#     y_test = np.load(y_test_savepath)
#     n_samples = x_train.shape[0] + y_train.shape[0] + x_test.shape[0] + y_test.shape[0]
#     for i in range(n_samples):
#         progress = (i+1)/n_samples*100
#         bar = '='*int(progress/2) + '>' + '-'*(50-int(progress/2))
#         print('\r[{:.2f}%]  '.format(progress) + bar, end='', flush=True)
# else:
#     print('-----------------------Generate Datasets-------------------------')
#     x_train, y_train = generateds(train_path,train_txt)
#     x_test,y_test = generateds(test_path, test_txt)
    
#     print('------------------------Save Datasets----------------------------')
#     np.save(x_train_savepath,x_train)
#     np.save(y_train_savepath,y_train)
#     np.save(x_test_savepath,x_test)
#     np.save(y_test_savepath,y_test)

# x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
#     x_train, y_train, test_size=0.2, random_state=577)

# image_gen_train = ImageDataGenerator(
#     rescale = 1. / 1., # 如为图像，分母为255时，可归至0~1
#     rotation_range = 45, # 随机45度旋转
#     width_shift_range = .15, # 宽度偏移
#     height_shift_range = .15, # 高度偏移
#     horizontal_flip = True, # 水平翻转
#     zoom_range = 0.5  # 将图像随机缩放阈量50%  
#     )
# train_generator = image_gen_train.flow(x_train_split, y_train_split, batch_size=64) # 32,64,128
# val_generator = image_gen_train.flow(x_val_split, y_val_split, batch_size=64) # 32,64,128
# print(f'the trains shape is {x_train.shape},the train label is {y_train.shape}')
# print(f'the test shape is {x_test.shape},the test label is {y_test.shape}')

#%% load data
x_train_savepath = '/public/home/ganyu/deep_learning/data/binnary_classification/total_image_cv/x_train_cv.npy'
y_train_savepath = '/public/home/ganyu/deep_learning/data/binnary_classification/total_image_cv/y_train_cv.npy'

x_train = np.load(x_train_savepath)
y_train = np.load(y_train_savepath)

# 数据增强
image_gen_train = ImageDataGenerator(
    rescale = 1. / 1., # 如为图像，分母为255时，可归至0~1
    rotation_range = 45, # 随机45度旋转
    width_shift_range = .15, # 宽度偏移
    height_shift_range = .15, # 高度偏移
    horizontal_flip = True, # 水平翻转
    zoom_range = 0.5  # 将图像随机缩放阈量50%  
    )
#%%
# VGG16 or VGG19
def create_model(model_type):
    inputs = Input(shape=(224,224,3))

    if model_type == "VGG16":
        base_model = VGG16(weights=None, include_top=False, input_tensor=inputs) # weight = 'imagenet'
    elif model_type == "DenseNet121":
        base_model = DenseNet121(weights=None, include_top=False, input_tensor=inputs)
    elif model_type == "InceptionV3":
        base_model = InceptionV3(weights=None, include_top=False, input_tensor=inputs)
    elif model_type == "ResNet50":
        base_model = ResNet50(weights=None, include_top=False, input_tensor=inputs)
    elif model_type == "ResNet101":
        base_model = ResNet101(weights=None, include_top=False, input_tensor=inputs)
    else:
        raise ValueError("Unsupported model type")

    # base_model.trainable = False  # 冻结卷积层

    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model
# Alex
def create_alex_model():
    model = tf.keras.models.Sequential(
        [
        Conv2D(filters = 96 ,kernel_size = (11,11), strides = 4 ,padding = 'valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D(pool_size = (3,3), strides = 2, padding = 'valid'),
        # tf.keras.layers.Dropout(0.2),
        
        Conv2D(filters = 256, kernel_size = (5,5), strides = 1,padding = 'valid' ),
        BatchNormalization(),
        Activation('relu'),
        MaxPool2D(pool_size= (3,3), strides = 2, padding= 'valid'),
        
        Conv2D(filters = 384 ,kernel_size = (3,3), strides = 1 ,padding = 'same'),
        Activation('relu'),
        
        Conv2D(filters = 384 ,kernel_size = (3,3), strides = 1 ,padding = 'same'),
        Activation('relu'),
        
        Conv2D(filters = 256 ,kernel_size = (3,3), strides = 1 ,padding = 'same'),
        Activation('relu'),
        MaxPool2D(pool_size=(3,3), strides= 2, padding='valid'),
         
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units = 4096, activation = 'relu' ),
        Dropout(0.5),
        tf.keras.layers.Dense(units = 4096, activation = 'relu'),
        Dropout(0.5),
        tf.keras.layers.Dense(units = 1 , activation = 'sigmoid')
        ]
        )  
    model.compile( optimizer = tf.keras.optimizers.Adam(1e-4),
                  loss = tf.keras.losses.BinaryCrossentropy(),
                  metrics= ['accuracy']
      )
    return model

early_stopping = EarlyStopping(monitor='val_accuracy', patience=40, restore_best_weights=True) # patience = 20,30,40,50
#%%

kf = KFold(n_splits=5,shuffle=True,random_state=577)

accuracys = []
precisions = []
recalls = []
f1s = []
results = []
threshold = 0.5
# 执行交叉验证
for train_index, val_index in kf.split(x_train):
    X_train, X_val = x_train[train_index], x_train[val_index]
    Y_train, Y_val = y_train[train_index], y_train[val_index]
    image_gen_train.fit(X_train)
    # 拟合模型
    model = create_model('VGG16')
    # model = create_alex_model()
    model.fit(
        image_gen_train.flow(X_train, Y_train, batch_size= 64), # 32,64,128
        validation_data = (X_val,Y_val),
        epochs = 200, 
        callbacks = [early_stopping],
        verbose = 1
    )
    y_pred_prob = model.predict(X_val)
    y_pred = np.where(y_pred_prob >= threshold, 1, 0)
    accuracy = accuracy_score(Y_val, y_pred) 
    precision = precision_score(Y_val, y_pred) 
    recall = recall_score(Y_val, y_pred) 
    f1 = f1_score(Y_val, y_pred) 
    # 将评估指标添加到列表中
    accuracys.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    results.append([accuracy,precision,recall,f1])
# 打印每折的评估结果和平均结果
df = pd.DataFrame(results, columns = ['acc','pre','recall','F1'])
df.to_csv('vgg16_cv_results_new.csv',index = False)
np.savez("vgg16_quota_new.npz", average_accuracy=np.mean(accuracys), average_macro_precisions=np.mean(precisions),
                              average_macro_recalls=np.mean(recalls), average_macro_f1s=np.mean(f1s))

