import matplotlib.pyplot as plt
from cnnCIFAR_utilsLoadData import img_size, num_channels, num_classes

img_size_cropped = 24

def plot_images(suptitle, classNames, images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true) == 9
    fig, axes = plt.subplots(3, 3)
    
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):        
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        ax.imshow(images[i, :, :, :],
                  interpolation=interpolation)
                
        cls_true_name = classNames[cls_true[i]]

        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            cls_pred_name = classNames[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

        ax.set_xlabel(xlabel)        
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle(suptitle)
    plt.show()    

    
def plot_image(image, cls_name, prd_name):
    imgplot = plt.imshow(image, interpolation = 'spline16')
    plt.xlabel('CLASE VERDADERA: ' + cls_name + '\n' + 'CLASE PREDECIDA: ' + prd_name)
    plt.show()
