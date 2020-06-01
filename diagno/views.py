from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from tensorflow import Graph, Session
from keras.applications.mobilenet import preprocess_input
import numpy as np

img_height, img_width =224,224

model_graph = Graph()
with model_graph.as_default():
    tf_session = Session()
    with tf_session.as_default():
        model = load_model('./models/medModel.h5')



def index(request):
    context = {'a':1}
    return render(request,'main.html',context)

def upload_img(request):
    return render(request,'a.html')

def gallery(request):
    return render(request, 'gallery.html')

def information(request):
    return render(request, 'information.html')

def predictImage(request):
    fileObj = request.FILES['filePath']   
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName) 
    test_image = '.'+filePathName
    img = image.load_img(test_image, target_size = (img_height, img_width))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    with model_graph.as_default():
        with tf_session.as_default():
            predi = model.predict(x)
   
    p_good,p_ill = np.around(predi, decimals=2)[0]
    predictedLabel = [p_good,p_ill]
    p_good = p_good * 100
    p_ill = p_ill *100

    if p_ill > p_good:
        predictedLabel = ' Chances of pneumonia is high with '+ str(p_ill) + ' %'
    else :
        predictedLabel = ' Chances of pneumonia is low with '+ str(p_ill) + ' %'
    context = {'filePathName': filePathName, 'predictedLabel': predictedLabel}
    return render(request,'index.html',context)

def viewDataBase(request):
    import os
    listOfImages = os.listdir('./media/')
    listOfImagesPath = ['./media/' + i for i in listOfImages]
    context = {'listOfImagespath' : listOfImagesPath}
    return render(request, 'viewdb.html', context)
