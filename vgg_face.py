from scipy.io import loadmat
import tensorflow as tf
import numpy as np

path="D:/matroid_project/vgg_face_matconvnet/data/vgg_face.mat"
meta_data = loadmat(path)

averageImage=meta_data['net'][0][0][0][0][0][3]

data=meta_data['net'][0][0][1][0]
n_layers=len(data)

names=[i['name'][0][0][0] for i in data]


names=names[:-2]


names.append("custom1")
names.append("softmax")

# build the graph
def vgg(input_maps):
    model={}
    #input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])
    layer=input_maps

    for k,i in enumerate(data):
        name=names[k]
        if('conv'in name):
           # print(i['stride'])
            stride=i['stride'][0][0][0][0]
           # print(stride)
            padding='SAME'
            kernel,bias=i['weights'][0][0][0]
            kernel=np.array(kernel)
            bias = np.squeeze(bias).reshape(-1)
            layer = tf.nn.conv2d(layer, tf.constant(kernel),strides=(1, stride, stride, 1), padding=padding)
            #layer= tf.nn.bias_add(conv, bias)
            print (name, 'stride:', stride, 'kernel size:', np.shape(kernel))

        elif('fc' in name):
            stride=i['stride'][0][0][0][0]
            padding='VALID'
            kernel,bias=i['weights'][0][0][0]
            kernel=np.array(kernel)
            bias = np.squeeze(bias).reshape(-1)
            layer = tf.nn.conv2d(layer, tf.constant(kernel),strides=(1, stride, stride, 1), padding=padding)
            #layer= tf.nn.bias_add(conv, bias)
            print (name, 'stride:', stride, 'kernel size:', np.shape(kernel))


        elif('pool' in name):
            stride=i['stride'][0][0][0][0]
            pool=i['pool'][0][0][0]
            layer = tf.nn.max_pool(layer, ksize=(1, 2, 2, 1),strides=(1, 2, 2, 1), padding='VALID')
            print (name, 'stride:', stride)
        elif('relu' in name):
            layer = tf.nn.relu(layer)
            print(name)
        elif ('softmax'in name):
            layer = tf.nn.softmax(tf.reshape(layer, [-1,2]))
            print (name)
            
        elif('custom' in name):
            kernel=tf.Variable(np.random.rand(1,1,4096,2),dtype='float32',trainable=True)
            layer = tf.nn.conv2d(layer,kernel,strides=(1, 1, 1, 1), padding='VALID')
            print (name, 'stride:', stride, 'kernel size:', np.shape(kernel))
        elif('dropout'in name):
            continue

        model[name]=layer
    return model,layer


    import tensorflow as tf
#tf.enable_eager_execution()
import os
import numpy as np
#keras offical document example of loading image
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

import glob


os.chdir("D:/matroid_project/")
file_list=[]
for file in glob.glob("*.txt"):
    file_list.append(file)
print(file_list)

d=0
file="D:/matroid_project/fold_%d_data.txt"%(d)
f = open(file, 'r')
import pandas as pd
image_list=[]
label_list=[]
folder_name=[]
user_id_list=[]
for file in file_list:
    a=pd.read_csv(file,delimiter='\t')
    a=a.dropna(subset=['gender'])
    user_id_list+=list(a['user_id'])
    image_list+=list(a['original_image'])
    label_list+=list(a['gender'])
print("data_size:")

main_dir='D:/matroid_project/aligned/'
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

def preprocess(ipg_dir):
    img = image.load_img(ipg_dir, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x=x/127.5-1
    return x


n=100
data1=np.empty((n,224,224,3))
print(data.shape)

for i in range(n):
    dir1=main_dir+user_id_list[i]+'/'
    for file in os.listdir(dir1):
           if(file.endswith(image_list[i])):
                    jpg_dir=dir1+file
                    break
        
    data1[i]=np.array(preprocess(jpg_dir))
print('done')
x=data1
y=label_list[:n]
print(len(x),len(y))
#input,label are collected

y=np.array(y)
y[np.where(y=='f')]=0
y[np.where(y=='m')]=1
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
y=to_categorical(y,num_classes=2)
print("input:")
print(x.shape)
print("output:")
print(y.shape)



t1, t2 = tf.placeholder(tf.float32, shape=[None,224,224,3]), tf.placeholder(tf.float32, shape=[None,2])
dataset = tf.data.Dataset.from_tensor_slices((t1, t2)).repeat().batch(1)

iter = dataset.make_initializable_iterator()
features, labels = iter.get_next()
model,train_predictions=vgg(features)

loss=tf.reduce_mean(-tf.reduce_sum( labels* tf.log(train_predictions), [1]))
train_op = tf.train.AdamOptimizer().minimize(loss)

#I only rain 1 epoch for 1 batch, but my cpu is still overloaded and it takes 10s. So i am not able to  iteerator all the data points in the dataset
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # initialise iterator with train data
    sess.run(iter.initializer, feed_dict={ t1: x, t2: y})
    print('Training...')
    for i in range(1):
        tot_loss = 0
        for _ in range(1):
            _, loss_value = sess.run([train_op, loss])
            tot_loss += loss_value
        print("Iter: {}, Loss: {:.4f}".format(i, tot_loss / 1))
    # initialise iterator with test data