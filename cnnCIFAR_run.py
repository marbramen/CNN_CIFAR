import cnnCIFAR_plots
import cnnCIFAR_utilsLoadData
import tensorflow as tf
import prettytensor as pt
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from datetime import timedelta
from sklearn.metrics import confusion_matrix

# variables globales
img_size_cropped = 24
train_batch_size = 64
batch_size = 256

num_classes_ = cnnCIFAR_utilsLoadData.num_classes
img_size_ = cnnCIFAR_utilsLoadData.img_size
num_channels_ = cnnCIFAR_utilsLoadData.num_channels

def getTestImage(i):
    return images_test[i, :, :, :], cls_test[i]

def preProcessImage(image, training):    
    if training:
        # para la fase de entrenamiento        
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, cnnCIFAR_utilsLoadData.num_channels])
        
        image = tf.image.random_flip_left_right(image)
                
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
        
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)
    return image

def preProcess(images, training):    
    images = tf.map_fn(lambda image: preProcessImage(image, training), images)

    return images

def mainNetwork(images, training): 
    x_pretty = pt.wrap(images)

    if training:
        phase = pt.Phase.train
    else:
        phase = pt.Phase.infer

    with pt.defaults_scope(activation_fn=tf.nn.relu, phase=phase):
        y_pred, loss = x_pretty.\
            conv2d(kernel=5, depth=64, name='layer_conv1', batch_normalize=True).\
            max_pool(kernel=2, stride=2).\
            conv2d(kernel=5, depth=64, name='layer_conv2').\
            max_pool(kernel=2, stride=2).\
            flatten().\
            fully_connected(size=256, name='layer_fc1').\
            fully_connected(size=128, name='layer_fc2').\
            softmax_classifier(num_classes=num_classes_, labels=y_true)

    return y_pred, loss 

def createNetwork(training):
    with tf.variable_scope('network', reuse=not training):
        images = x
        images = preProcess(images=images, training=training)
        y_pred, loss = mainNetwork(images=images, training=training)
        
    return y_pred, loss    

def randomBatch():    
    num_images = len(images_train)    
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)   
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch	

def optimize(num_iterations):
   
    start_time = time.time()

    for i in range(num_iterations):        
        # x_batch => batch de imagenes
        # y_true_batch => batch de los labels de las imagenes
        x_batch, y_true_batch = randomBatch()

        # coloca los batchs en los placeholders
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # We also want to retrieve the global_step counter.
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        # imprime el estado cada 100 iterations (and last).
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # calcula la exactitud por el batch procesado
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)           
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

        # guarda el checkpoint cada 1000 iteraciones (y la ultima).
        if (i_global % 1000 == 0) or (i == num_iterations - 1):
            # Guarda todas las variables del TensorFlow graph al
            # checkpoint. 
            saver.save(session,
                       save_path=save_path,
                       global_step=global_step)

            print("Saved checkpoint.")
    
    end_time = time.time()   
    time_dif = end_time - start_time   
    print("Tiempo empleado: " + str(timedelta(seconds=int(round(time_dif)))))

def predict_cls(images, labels, cls_true):
    # Number of images.
    num_images = len(images)  
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Prediciendo la clase por batch.    
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)
        
        feed_dict = {x: images[i:j, :],
                     y_true: labels[i:j, :]}

        # Se predice la clase usando TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)       
        i = j

    # Array donde la imagen es correctamente predecida
    correct = (cls_true == cls_pred)
    return correct, cls_pred

def classification_accuracy(correct):    
    # Retorna classification-accuracy
    # y el numero de clasificados correctamente.
    return correct.mean(), correct.sum()

def print_test_accuracy():
    # Para todas las imagenes en el test-set,
    # calcula la clase predecida y cual de ellas es correcta.
    correct, cls_pred = predict_cls(images = images_test, labels = labels_test, cls_true = cls_test)
        
    acc, num_correct = classification_accuracy(correct)
    num_images = len(correct)

    # Print the accuracy.
    msg = "Exactitud en el Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))
    
    print("Ejemplo de errores:")    
    incorrect = (correct == False)
    images = images_test[incorrect]    
    # Obteniendo las clases predecidas para estas imagenes.
    cls_pred_errors = cls_pred[incorrect]
    # Obteniendo las clases verdaderas para estas imagenes.
    cls_true_errors = cls_test[incorrect]

    # Plot some examples of mis-classifications, if desired.
    cnnCIFAR_plots.plot_images('Ejemplos de errores de prediccion', class_names,   images=images[0:9], cls_true=cls_true_errors[0:9], cls_pred=cls_pred_errors[0:9])

    # Plot matriz de confunsion
    cm = confusion_matrix(y_true=cls_test,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # Imprime la matriz de confunsion como texto.
    for i in range(num_classes_):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    # Imprimiendo las clases.
    class_numbers = [" ({0})".format(i) for i in range(num_classes_)]
    print("".join(class_numbers))    

def testCNN(array):
    for elem in array:        
        img, cls = getTestImage(elem)
        label_pred, cls_pred = session.run([y_pred, y_pred_cls], feed_dict={x: [img]})
        np.set_printoptions(precision=3, suppress=True)      
        cnnCIFAR_plots.plot_image(img, class_names[cls], class_names[cls_pred[0]])
        
if __name__ == "__main__":
    
	# cnnCIFAR_utilsLoadData	
	print("\t \t === CARGANDO DATA ===")
	class_names = cnnCIFAR_utilsLoadData.loadClassNames()
	images_train, cls_train, labels_train = cnnCIFAR_utilsLoadData.loadTrainingData()
	images_test, cls_test, labels_test = cnnCIFAR_utilsLoadData.loadTestData()
	print("Size of:")
	print("- Training-set:\t\t{}".format(len(images_train)))
	print("- Test-set:\t\t{}".format(len(images_test)))
		
	# Plot the images and labels using our helper-function above.
	#cnnCIFAR_plots.plot_images('imagenes al azar', class_names, images=images_test[0:9], cls_true=cls_test[0:9], smooth=True)

	print("\t \t === LEVANTANDO RED NEURONAL ===")	
	# PLACEHOLDERS 
	print("\t \t === CREANDO PLACEHOLDERS === ")	
	x = tf.placeholder(tf.float32, shape=[None, img_size_, img_size_, num_channels_], name='x')
	y_true = tf.placeholder(tf.float32, shape=[None, num_classes_], name='y_true')
	y_true_cls = tf.argmax(y_true, dimension=1)
	print("\t \t === PLACEHOLDERS CREADOS === \n")

	# CREATE TRAINING NETWORK
	global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
	# red neuronal para training
	_, loss = createNetwork(training=True)
	# optimiza la los-function
	optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)
	print("\t \t === RED DE TRAINING CREADA === \n")

	# CREATE TESTING NETWORK 
	y_pred,_ = createNetwork(training=False)
	y_pred_cls = tf.argmax(y_pred, dimension=1)
	correct_prediction = tf.equal(y_pred_cls, y_true_cls)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("\t \t === RED DE TESTING CREADA === \n\n")

	print("\t \t === EJECUTANDO RED NEURONAL === ")
	saver = tf.train.Saver()

	session = tf.Session()
	save_dir = 'checkpoints/'
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	save_path = os.path.join(save_dir, 'cifar10_cnn')
	try:
	    print("Intentando restaurar el ultimo punto")	    
	    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)

	    # cargando la data desde el ultimo punto
	    saver.restore(session, save_path=last_chk_path)
	    
	    print("Checkpoint restaurado desde ", last_chk_path)
	except:
	    # If the above failed for some reason, simply
	    print("Fallo la restauracion del checkpoint. Inicializando las variables")
	    session.run(tf.global_variables_initializer())

	if False:
		optimize(num_iterations=20000)

 	#mostrando la precision
	print_test_accuracy()

        # realizando un test a la red neuronal
	arrayTest = [16, 69, 100, 239, 341]
	testCNN(arrayTest)


