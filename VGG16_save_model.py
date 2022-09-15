import cv2
import keras
from numpy.random import seed

seed(1)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import  random

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from glob import glob
import seaborn as sns
import pickle as pkl
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.utils import resample
from PIL import Image
import scipy.ndimage as  ndimage
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet152, Xception, VGG16, EfficientNetB4,ResNet50,ResNet101
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPool2D, AveragePooling2D, \
    GlobalMaxPooling2D, GlobalAveragePooling2D
from keras_drop_block import DropBlock2D
from tensorflow.keras import regularizers

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping,ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from tensorflow.keras.layers import BatchNormalization
from sklearn.utils.class_weight import compute_class_weight


lesion_type_dict = {'akiec': 'Actinic keratoses',
                    'bcc': 'Basal cell carcinoma',
                    'bkl': 'Benign keratosis-like lesions ',
                    'df': 'Dermatofibroma',
                    'nv': 'Melanocytic nevi',
                    'mel': 'Melanoma',
                    'vasc': 'Vascular lesions'}  #1,0,6,3
base_skin_dir = os.path.join('..', 'input')
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('HAM10000/', '*', '*.jpg'))}
skin_df = pd.read_csv('HAM10000/HAM10000_metadata.csv')
skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get)
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes
skin_df.groupby(['dx']).count()
print(skin_df)
print(skin_df.cell_type.value_counts())

fig = plt.figure(figsize=(12, 8))

ax1 = fig.add_subplot(221)
skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('Cell Type');

ax2 = fig.add_subplot(222)
skin_df['sex'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_ylabel('Count', size=15)
ax2.set_title('Sex');

ax3 = fig.add_subplot(223)
skin_df['localization'].value_counts().plot(kind='bar')
ax3.set_ylabel('Count', size=12)
ax3.set_title('Localization')

ax4 = fig.add_subplot(224)
sample_age = skin_df[pd.notnull(skin_df['age'])]
sns.distplot(sample_age['age'], fit=stats.norm, color='red');
ax4.set_title('Age')

plt.tight_layout()
plt.savefig('fig1.png')
plt.show()

def features_data():
    skin_df['image'] = skin_df['path'].map(lambda x:
                                           np.asarray(Image.open(x).resize(224, 224)))
    return skin_df['image']

images=[]
for x in skin_df['image_id']:
    images.append(x)
hair_images=images

def hair_remove(image) :
    #convert image to grayScale
    grayScale=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    #kernal for morphologyEx
    kernel=cv2.getStructuringElement(1,(10,10))#kernal shape :MORPH_CROSS
    #apply MORPH_BLACKHAT to grayScale image
    blackhat=cv2.morphologyEx(grayScale,cv2.MORPH_BLACKHAT,kernel)#orignal-(dilation than erosion)
    #applying threshould to blackhat
    _,threshold=cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)#10:value of threshold ,first value is retVal
    #inpaint with orignal image and threshold image
    final_image=cv2.inpaint(image,threshold,5,cv2.INPAINT_TELEA) #Fast Marching Method  5 is value of  inpaintRadius
    return final_image

# remove hair
features_list=[]
# for i,image_name in enumerate(hair_images[0:10015]):
#     image=cv2.imread(imageid_path_dict.get(image_name))
#     image_resize=cv2.resize(image,(224,224))
#     final_image=hair_remove(image_resize)
#     final_image=cv2.cvtColor(final_image,cv2.COLOR_BGR2RGB)  #change to RGB
#     features_list.append(final_image)

#skin_df['image_features']=features_list
# pkl.dump(skin_df['image_features'],open('image_features','wb'))
skin_df['image_features']=pkl.load(open('image_features','rb'))
skin_df['image_features']=ndimage.gaussian_laplace(skin_df['image_features'],sigma=0)

features = skin_df.drop(columns=['cell_type_idx'], axis=1)
target = skin_df['cell_type_idx']
print('features and target loaded')

# hair remove
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(features, target, test_size=0.2, random_state=666)
x_test = np.asarray(x_test_o['image_features'].tolist())
y_test = to_categorical(y_test_o, num_classes=7)

x_train, x_validate, y_train, y_validate = train_test_split(x_train_o, y_train_o, test_size=0.2, random_state=666)
x_validate = np.asarray(x_validate['image_features'].tolist())
y_validate=to_categorical(y_validate,num_classes=7)


skin_df_balanced_pre = pd.concat([x_train,y_train],axis=1)
df_0 = skin_df_balanced_pre[skin_df_balanced_pre['cell_type_idx'] == 0]
df_1 = skin_df_balanced_pre[skin_df_balanced_pre['cell_type_idx'] == 1]
df_2 = skin_df_balanced_pre[skin_df_balanced_pre['cell_type_idx'] == 2]
df_3 = skin_df_balanced_pre[skin_df_balanced_pre['cell_type_idx'] == 3]
df_4 = skin_df_balanced_pre[skin_df_balanced_pre['cell_type_idx'] == 4]
df_5 = skin_df_balanced_pre[skin_df_balanced_pre['cell_type_idx'] == 5]
df_6 = skin_df_balanced_pre[skin_df_balanced_pre['cell_type_idx'] == 6]


df_0_balanced = resample(df_0, replace=True, n_samples=500, random_state=42)
df_1_balanced = resample(df_1, replace=True, n_samples=500, random_state=42)
df_3_balanced = resample(df_3, replace=True, n_samples=500, random_state=42)
df_6_balanced = resample(df_6, replace=True, n_samples=500, random_state=42)

skin_df_balanced = pd.concat([df_2, df_4,
                              df_5, df_0_balanced,
                              df_1_balanced, df_3_balanced, df_6_balanced])

print(skin_df_balanced.cell_type_idx.value_counts())

print('sampling over')



x_train=np.asarray(skin_df_balanced['image_features'].tolist())
y_train=to_categorical(skin_df_balanced['cell_type_idx'],num_classes=7)

print('done')
print(x_train.shape)
print(x_test.shape)
print(x_validate.shape)
print(y_train.shape)
print(y_test.shape)
print(y_validate.shape)


def plot_loss_accuray(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['accuracy', 'val_accuracy'])
    plt.savefig('fig2.png')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('Epochs')
    plt.legend(['loss', 'val_loss'])
    plt.savefig('fig3.png')
    plt.show()


class_ids = {0:'Actinic keratoses',
              1:'Basal cell carcinoma',
              2:'Benign keratosis-like lesions ',
              3:'Dermatofibroma',
              4:'Melanocytic nevi',
              5:'Melanoma',
              6:'Vascular lesions'}

categories =['Actinic keratoses',
             'Basal cell carcinoma',
             'Benign keratosis-like lesions ',
             'Dermatofibroma',
             'Melanocytic nevi',
             'Melanoma',
             'Vascular lesions'

]

def cm_pred(y_test_o,predictions):
    y_test_o=y_test_o
    predictions = np.array(list(map(lambda x: np.argmax(x), predictions)))
    #print(predictions)
    print(classification_report(y_test_o, predictions, target_names=categories))
    CMatrix = pd.DataFrame(confusion_matrix(y_test_o, predictions), columns=categories, index =categories)
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(CMatrix, annot = True, fmt = 'g' ,vmin = 0, vmax = 10,cmap = 'YlGnBu')
    ax.set_xlabel('Predicted',fontsize = 14,weight = 'bold')
    ax.set_xticklabels(ax.get_xticklabels(),rotation =90);
    ax.set_ylabel('Actual',fontsize = 14,weight = 'bold')
    ax.set_title('Confusion Matrix - Test Set',fontsize = 16,weight = 'bold',pad=20);
    plt.savefig('fig4.png')
    plt.show()

    y_test_o = pd.DataFrame(y_test_o)
    y_test_o['predictions']=predictions
    fig = plt.figure(figsize=(20, 20))
    for i in range(1, 26):
        x = random.choice(list(x_test_o.index.values))
        fig.add_subplot(5, 5, i)
        plt.imshow(x_test_o['image_features'][x], cmap="gray")
        if y_test_o['predictions'][x] == skin_df['cell_type_idx'][x]:
            title_color = "green"
        else:
            title_color = "red"
        plt.title(class_ids[y_test_o['predictions'][x]], fontdict={'fontsize': 18}, color=title_color)
        plt.axis('off')
    plt.suptitle("Model Predictions")
    plt.savefig('fig5.png')
    plt.show()

x_train=x_train.reshape(x_train.shape[0],*(224,224,3))
x_test=x_test.reshape(x_test.shape[0],*(224,224,3))
x_validate = x_validate.reshape(x_validate.shape[0], *(224, 224, 3))
input_shape=(224,224,3)
num_clases=7

optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=6, verbose=1, factor=0.05,
                                            min_learning_rate=0.00001)
cb_early_stopper = EarlyStopping(monitor='val_accuracy',verbose=1,patience=40,mode='max')
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    vertical_flip=True)

datagen.fit(x_train)
lesion_ID_dict = {
    'nv': 0,
    'mel': 1,
    'bkl': 2,
    'bcc': 3,
    'akiec': 4,
    'vasc': 5,
    'df': 6
}
skin_df['lesion_ID'] = skin_df['dx'].map(lesion_ID_dict)

y_id = np.array(skin_df['lesion_ID'])
class_weights = np.around(compute_class_weight(class_weight='balanced',classes=np.unique(y_id),y=skin_df_balanced['cell_type_idx']),2)
class_weights = dict(zip(np.unique(y_id),class_weights))

print('The problem is unbalanced. We need to provide class_weights ')
print(class_weights)

batch_size = 32
vgg16_base=tf.keras.applications.VGG16(include_top=False,weights='imagenet',input_tensor=None,input_shape=(224,224,3))

#adding new layers
# newlayers_vgg16=vgg16_base.get_layer(index=-1).output
# newlayers_vgg16=GlobalAveragePooling2D()(newlayers_vgg16)
# newlayers_vgg16=BatchNormalization()(newlayers_vgg16)
# newlayers_vgg16=Dense(224,activation="relu")(newlayers_vgg16)
# newlayers_vgg16=Dropout(0.5)(newlayers_vgg16)
# newlayers_vgg16=BatchNormalization()(newlayers_vgg16)
# newlayers_vgg16=Dense(112,activation="softmax")(newlayers_vgg16)
# newlayers_vgg16=Dropout(0.5)(newlayers_vgg16)
# newlayers_vgg16=BatchNormalization()(newlayers_vgg16)
# newlayers_vgg16=Dense(7,activation="softmax")(newlayers_vgg16)

newlayers_vgg16=vgg16_base.get_layer(index=-1).output
newlayers_vgg16 = DropBlock2D(block_size=3, keep_prob=0.5)(newlayers_vgg16)
newlayers_vgg16 = BatchNormalization()(newlayers_vgg16)
newlayers_vgg16 = AveragePooling2D()(newlayers_vgg16)
newlayers_vgg16 = Flatten(name="flatten")(newlayers_vgg16)
newlayers_vgg16 = Dense(128, activation="relu")(newlayers_vgg16)
newlayers_vgg16 = Dropout(0.5)(newlayers_vgg16)
newlayers_vgg16 = Dense(64, activation="softmax")(newlayers_vgg16)
newlayers_vgg16 = Dense(7, activation="softmax")(newlayers_vgg16)

VGG16_model_dropout=Model(vgg16_base.input,newlayers_vgg16)
for layer in VGG16_model_dropout.layers[:-8]:
    layer.trainable=False
VGG16_model_dropout.summary()

#mcp=tf.keras.callbacks.ModelCheckpoint(monitor='val_accuracy',patience=6,verbose=1,filepath='./VGG16_model.h5',
#                                       save_best_only=True,mode='max')

VGG16_model_dropout.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
history_VGG16=VGG16_model_dropout.fit(x_train,y_train,batch_size=batch_size,epochs=3,
                              validation_data=(x_validate,y_validate),verbose=1,steps_per_epoch=x_train.shape[0]//batch_size,callbacks=[learning_rate_reduction])
VGG16_model_dropout.save('VGG16_model.h5')
loss,accuracy=VGG16_model_dropout.evaluate(x_test,y_test,verbose=0)
predictions=VGG16_model_dropout.predict(x_test)
loss_v,accuracy_v=VGG16_model_dropout.evaluate(x_validate,y_validate,verbose=0)
loss_t,accuracy_t=VGG16_model_dropout.evaluate(x_train,y_train,verbose=0)
print("Training: accuracy = %f" % (accuracy_t))
print("Validation: accuracy = %f" % (accuracy_v))
print("Test: accuracy = %f" % (accuracy))
plot_loss_accuray(history_VGG16)
cm_pred(y_test_o,predictions)













