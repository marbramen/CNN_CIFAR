import numpy as np
import pickle
import os

data_path = "D:\\CesarBragagnini\\MCS\\SistemasInteligentes\\CNN\\data\\"
img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 10
num_files_train = 5
images_per_file = 10000
num_images_train = num_files_train * images_per_file


def getDataFile(filename):       
    file_path = os.path.join(data_path, "cifar-10-batches-py/", filename)
    print("Loading data: " + file_path)
    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    return data

def transformImages(raw):
    raw_float = np.array(raw, dtype=float) / 255.0   
    images = raw_float.reshape([-1, num_channels, img_size, img_size])
    images = images.transpose([0, 2, 3, 1])
    return images

def one_hot_encoded(class_numbers, num_classes=None):
    # Find the number of classes if None is provided.
    if num_classes is None:
        num_classes = np.max(class_numbers) - 1
    return np.eye(num_classes, dtype=float)[class_numbers]

def loadData(filename):
	data = getDataFile(filename)
	raw_images = data[b'data']
	cls = np.array(data[b'labels'])
	images = transformImages(raw_images)
	return images, cls

def loadClassNames():
    raw = getDataFile(filename="batches.meta")[b'label_names']
    names = [x.decode('utf-8') for x in raw]
    return names

def loadTrainingData():
    # se crea arrays  para images y para class numbers
    images = np.zeros(shape=[num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[num_images_train], dtype=int)
    begin  = 0
    for i in range(num_files_train):
        images_batch, cls_batch = loadData(filename="data_batch_" + str(i+1))
        num_images= len(images_batch)
        end = begin + num_images
        images[begin:end, :] = images_batch
        cls[begin:end] = cls_batch
        begin = end
    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)
    
def loadTestData():
    images, cls = loadData(filename="test_batch")
    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes)    


    
