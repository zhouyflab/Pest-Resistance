# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 16:33:09 2023

@author: 86176
"""

from PIL import Image
import tensorflow as tf 
import numpy as np
import os 
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_absolute_error,mean_squared_error
# from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, BatchNormalization , Activation, MaxPool2D , Dropout, Flatten, Dense, GlobalAveragePooling2D,Input,Add
from tensorflow. keras.applications.vgg16 import VGG16
from tqdm import tqdm
np.set_printoptions(threshold= np.inf)
#%%
def generateds(path, txt):
    f  = open(txt, 'r')   
    contents = f.readlines()
    f.close()
    x, y = [], []
    for content in tqdm(contents,desc = 'loading data'):
        value = content.split()
        img_path = path + value[0]
        img = Image.open(img_path)
        img = img.resize((224,224),Image.ANTIALIAS)
        img = np.array(img)# 变为np.array格式
        img = img / 255
        x.append(img)
        y.append(float(value[1]))
        
    
    x = np.array(x)
    y = np.array(y)
    # y = np.reshape(y, (len(y),1))
    return x, y


x_train_savepath = '/public/home/ganyu/deep_learning/data/pest_damage_degree/train_0.8/x_train.npy' 
y_train_savepath = '/public/home/ganyu/deep_learning/data/pest_damage_degree/train_0.8/y_train.npy'
x_val_savepath = '/public/home/ganyu/deep_learning/data/pest_damage_degree/val_0.2/x_val.npy'
y_val_savepath = '/public/home/ganyu/deep_learning/data/pest_damage_degree/val_0.2/y_val.npy'
test_path = '/public/home/ganyu/deep_learning/data/pest_damage_degree/test_439/'
test_txt = '/public/home/ganyu/deep_learning/data/pest_damage_degree/test_439/label.txt'
x_test_savepath = '/public/home/ganyu/deep_learning/data/pest_damage_degree/test_439/x_test.npy'
y_test_savepath = '/public/home/ganyu/deep_learning/data/pest_damage_degree/test_439/y_test.npy'
x_train = np.load(x_train_savepath)
y_train = np.load(y_train_savepath)
x_val = np.load(x_val_savepath)
y_val = np.load(y_val_savepath)
x_test, y_test = generateds(test_path,test_txt)
np.save(x_test_savepath,x_test)
np.save(y_test_savepath,y_test)
n_samples = x_train.shape[0] + y_train.shape[0] + x_val.shape[0] + y_val.shape[0]
for i in range(n_samples):
    progress = (i+1)/n_samples*100
    bar = '='*int(progress/2) + '>' + '-'*(50-int(progress/2))
    print('\r[{:.2f}%]  '.format(progress) + bar, end='', flush=True)
image_gen_train = ImageDataGenerator(
    rescale = 1. / 1., # 如为图像，分母为255时，可归至0~1
    rotation_range = 45, # 随机45度旋转
    width_shift_range = .15, # 宽度偏移
    height_shift_range = .15, # 高度偏移
    horizontal_flip = True, # 水平翻转
    zoom_range = 0.5  # 将图像随机缩放阈量50%  
    )
image_gen_train.fit(x_train)
print(f'the trains shape is {x_train.shape},the train label is {y_train.shape}')
print(f'the val shape is {x_val.shape},the val label is {y_val.shape}')
print(f'the test shape is {x_test.shape},the test label is {y_test.shape}')
#%%
def residual_block(input_tensor, filters):
    x = Conv2D(filters, (3,3), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3,3), padding='same')(x)
    x = BatchNormalization()(x)

    if input_tensor.shape[-1] != filters:
        input_tensor = Conv2D(filters, (1, 1), padding='same')(input_tensor)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def create_model():
    base_model = VGG16(weights='/public/home/ganyu/deep_learning/model/regression_pestdamage/VGG16_imagenet.h5',include_top = False)
    output = base_model.output
    output = residual_block(output, 512)
    output = residual_block(output, 256)
    output = residual_block(output, 128)
    output = residual_block(output, 64)
    output = GlobalAveragePooling2D()(output)
    output = Dense(1024, activation = 'relu')(output) #1024,2048,4096
    output = Dropout(0.3)(output) # 0.3,0.4,0.5
    output = Dense(1024, activation = 'relu')(output) #1024,2048,4096
    output = Dropout(0.3)(output) # 0.3,0.4,0.5
    output = Dense(1024, activation = 'relu')(output) #1024,2048,4096
    output = Dropout(0.3)(output) # 0.3,0.4,0.5
    output = Dense(1, activation = 'sigmoid')(output) 
    pre = output*5
    
    model = Model(inputs = base_model.input, outputs = pre)
    
    # 只训练额外层，锁住VGG16的卷积层
    for layer in base_model.layers:
        layer.trainable = False

    model.compile( optimizer = tf.keras.optimizers.Adam(1e-3), #1e-3,1e-4,1e-5
                  loss = 'mean_squared_error' ,
                  metrics= ['mae']
     )
    return model
model = create_model()
#%%
model_path = '/public/home/ganyu/deep_learning/model/regression_pestdamage/result/'
save_path = model_path + 'my_regression_model_test.h5' 
history_path = model_path + 'history.npy'
if os.path.exists(history_path):
    print('Loading history...')
    history = np.load(history_path, allow_pickle=True).item()
    initial_epoch = len(history.get('loss', []))
else:
    history = {}
    initial_epoch = 0

# load model (if exit?)
if os.path.exists(save_path):
    print('Loading model...')
    model = tf.keras.models.load_model(save_path)
else:
    model = model
early_stopping = EarlyStopping(monitor='val_mae', patience=35, restore_best_weights=True)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath = save_path,
                save_weights_only = False, #只保留模型参数
                save_best_only = True) # 只保留最优模型
#%%
#train
history_new = model.fit(
    image_gen_train.flow(x_train, y_train, batch_size= 64), #32,64,128
    validation_data = (x_val,y_val),
    epochs = 300,
    validation_freq= 1,
    callbacks = [cp_callback,early_stopping],
    initial_epoch = initial_epoch,
    verbose = 1
)

for key in history_new.history.keys():
    if key not in history:
        history[key] = []
    history[key].extend(history_new.history[key])

# save history to .npy file
np.save(history_path, history)

pred = model.predict(x_test)
mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)



# print result
print("MSE:", mse)
print("MAE:", mae)
np.savez("result_quota_test.npz", mse = mse,mae = mae)


# draw
x_ticks = list(range(1,len(history['loss']) + 1))

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

# loss
ax[0].plot(x_ticks, history['loss'], label='Train')
ax[0].plot(x_ticks, history['val_loss'], label='Validation')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss_MSE')
ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].legend()

ax[1].plot(x_ticks, history['mae'], label='Train')
ax[1].plot(x_ticks, history['val_mae'], label='Validation')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('MAE')
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True)) 
ax[1].legend()

plt.tight_layout()
plt.savefig(model_path + 'learning_curve.jpg', dpi=400)

