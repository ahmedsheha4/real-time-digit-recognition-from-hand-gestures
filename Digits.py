import numpy as np
from tensorflow.keras.applications.vgg16  import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model , Sequential
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
import os
import cv2
import time
from glob import glob

def train_model():

    imageSize=(224,224,3)
    model = VGG16(input_shape=imageSize, weights="imagenet" , include_top='False')
    epochs = 3
    batch_size = 16
    train_path = './Dataset'
    image_files = glob(train_path + '/*/*.png')
    plt.imshow(image.img_to_array(image.load_img(np.random.choice(image_files))).astype('uint8'))
    plt.show()

    vggModel = Sequential()

    for i in range (0,len(model.layers)-4) :
        vggModel.add(model.layers[i])

    for layer in vggModel.layers :
        layer.trainable=False
    vggModel.add(Flatten())

    vggModel.add(Dense(512, activation='relu'))

    vggModel.add(Dense(256, activation='relu'))

    vggModel.add(Dense(6, activation='softmax'))

    vggModel.summary()

    vggModel.compile(
      loss='categorical_crossentropy',
      optimizer=Adam(0.0001),
      metrics=['accuracy']
    )

    gen = ImageDataGenerator(
      rotation_range=20,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=0.2,
      horizontal_flip=True,
      vertical_flip=True,
      preprocessing_function=preprocess_input
    )

    train_generator = gen.flow_from_directory(
      train_path,
      target_size=(224,224),
      shuffle=True,
      batch_size=batch_size,
    )

    vggModel.fit_generator(
      train_generator,
      epochs=epochs,
      steps_per_epoch=len(image_files) // batch_size,
    )

    model_json = vggModel.to_json()
    with open("Trained Model/model.json", "w") as json_file:
        json_file.write(model_json)

    vggModel.save_weights("Trained Model/model.h5")
    print("Saved model to disk")


#train_model()



def loadModel():
    global model

    json_file = open('Trained Model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("Trained Model/model.h5")
    print("Loaded model from disk")
    model = loaded_model




def SignLanguage():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (256, 256)
    fontScale =3
    fontColor = (255,255,255)
    lineType = 2

    while (True):
      ret, frame = cap.read()
      count = count + 1

      if ret == True :
        #cv2.imshow('frame', frame)
        clone1 = frame[0:300, 0:300]
        cv2.namedWindow("image")
        cv2.imshow("image", clone1)
        img = cv2.resize(clone1, (224, 224))
        img = img.reshape((1,224,224,3))
        if count % 10 == 0 :
          prediction = model.predict_classes(img)
          prediction =  str(prediction[0])
          print(prediction)
          img2 = np.zeros((512, 512, 3), np.uint8)
          frame = cv2.putText(img2, prediction,
                      bottomLeftCornerOfText,
                      font,
                      fontScale,
                      fontColor,
                      lineType)
          cv2.imshow('frame2', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
          break

      else : print('False')

    cap.release()
    cv2.destroyAllWindows()



loadModel()
SignLanguage()
