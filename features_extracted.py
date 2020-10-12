import plaidml.keras
plaidml.keras.install_backend()
import keras
from keras import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
import numpy as np
import pickle

model = InceptionV3(weights = "imagenet")
model_new = Model(model.input, model.layers[-2].output)



'''file = open("Dataset/Flickr_TextData/Flickr_8k.trainImages.txt",'r')
doc = file.read()
images = [image for image in doc.split("\n")]'''

file = open("Dataset/Flickr_TextData/Flickr_8k.testImages.txt",'r')
doc = file.read()
images = [image for image in doc.split("\n")]


extracted_features = dict()
test_features = dict()

def convert_img_to_features(image_name):
    img = image.load_img("Dataset/Images/"+image_name, target_size = (299, 299))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis =0)      
    x = preprocess_input(x)
    x = model_new.predict(x)   
    x = np.reshape(x, x.shape[1])
    return x

for image_ in images:
    test_features[image_] = convert_img_to_features(image_)
    
convert_img_to_features(images[0])

with open("train_features.pickle", "wb") as handle:
    pickle.dump(extracted_features, handle)


with open("test_features.pickle", "wb") as handle:
    pickle.dump(test_features, handle)
