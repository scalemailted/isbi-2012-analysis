"""
Importing Modules
The necessary modules are : os, opencv, numpy, tqdm, matplotlib, keras and sklearn
"""

from operator import mod
import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from keras.models import Model, model_from_json
from tensorflow.keras.optimizers import Adam
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model
from keras import backend as K 
from sklearn.metrics import classification_report



def main():
    train_path = 'data/membrane/train/'
    test_path = 'data/membrane/test' 
    X_train, Y_train = load_data(train_path)                                                  #1. Load Training data
    X_test, Y_test = load_data(test_path)                                                     #2. Load Test data
    X_train, Y_train, X_test, Y_test = preprocess_data(X_train, Y_train, X_test, Y_test)      #3. Normalize Dataset
    train_model(X_train, Y_train, X_test, Y_test)                                             #4. Train model




"""
Constructing Training Datasets
Loading the Training Images
  - We first load all the training images and the corresponding segmentation masks.
  - They are stored in two lists X_train, Y_train and respectively
  - Moreover, the images are resized to 256x192
"""
def load_data(path):
    img_files = next(os.walk(f'{path}/input'))[2]
    mask_files = next(os.walk(f'{path}/target'))[2]
    
    img_files.sort()
    mask_files.sort()
    
    print(len(img_files))
    print(len(mask_files))
    
    X = []
    Y = []
    
    for img_file in tqdm(img_files):    
        if img_file.endswith('.png'):
            img = cv2.imread( f'{path}/input/{img_file}', cv2.IMREAD_COLOR)
            resized_img = cv2.resize(img,(256, 192), interpolation = cv2.INTER_CUBIC)
            X.append(resized_img)
            msk = cv2.imread(f'{path}/target/{img_file}', cv2.IMREAD_GRAYSCALE)
            resized_msk = cv2.resize(msk,(256, 192), interpolation = cv2.INTER_CUBIC)
            Y.append(resized_msk)
        
    return (X,Y)





"""
Pre-Process Data
 - The X, Y lists are converted to numpy arrays for convenience. 
 - Furthermore, the images are divided by 255 to bring down the pixel values to [0...1] range. 
 - On the other hand the segmentations masks are converted to binary (0 or 1) values.

"""
def preprocess_data(X_train, Y_train, X_test, Y_test):

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],3))
    X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],3))
    Y_train = Y_train.reshape((Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],1))
    Y_test = Y_test.reshape((Y_test.shape[0],Y_test.shape[1],Y_test.shape[2],1))
    
    X_train = X_train / 255
    X_test = X_test / 255
    Y_train = Y_train / 255
    Y_test = Y_test / 255

    Y_train = np.round(Y_train,0)	
    Y_test = np.round(Y_test,0)	

    return (X_train, Y_train, X_test, Y_test)



"""
Auxiliary Functions
Custom Metrics
  - Since Keras does not have build-in support for computing Dice Coefficient or Jaccard Index 
  - the following functions are declared
"""
def dice_coef(y_true, y_pred):
    smooth = 0.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jacard(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum ( y_true_f * y_pred_f)
    union = K.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)
    return intersection/union



"""
Saving Model
  - Function to save the model
"""
def saveModel(model):
    model_json = model.to_json()
    try:
        os.makedirs('models')
    except:
        pass
    fp = open('models/modelP.json','w')
    fp.write(model_json)
    model.save_weights('models/modelW.h5')


"""
Evaluate the Model
We evaluate the model on test data (X_test, Y_test).

We compute the values of Jaccard Index and Dice Coeficient, and save the predicted segmentation of first 10 images. The best model is also saved

(This could have been done using keras call-backs as well)
"""
def evaluateModel(model, X_test, Y_test, batchSize):
    print('evaluateModel')
    try:
        os.makedirs('results')
    except:
        pass 
    

    yp = model.predict(x=X_test, batch_size=batchSize, verbose=1)

    yp = np.round(yp,0)

    for i in range(30):

        plt.figure(figsize=(20,10))
        plt.subplot(1,3,1)
        plt.imshow(X_test[i])
        plt.title('Input')
        plt.subplot(1,3,2)
        plt.imshow(Y_test[i].reshape(Y_test[i].shape[0],Y_test[i].shape[1]))
        plt.title('Ground Truth')
        plt.subplot(1,3,3)
        plt.imshow(yp[i].reshape(yp[i].shape[0],yp[i].shape[1]))
        plt.title('Prediction')

        intersection = yp[i].ravel() * Y_test[i].ravel()
        union = yp[i].ravel() + Y_test[i].ravel() - intersection

        jacard = (np.sum(intersection)/np.sum(union))  
        plt.suptitle('Jacard Index'+ str(np.sum(intersection)) +'/'+ str(np.sum(union)) +'='+str(jacard))

        plt.savefig('results/'+str(i)+'.png',format='png')
        plt.close()

    jacard = 0
    dice = 0
    
    
    for i in range(len(Y_test)):
        yp_2 = yp[i].ravel()
        y2 = Y_test[i].ravel()
        
        intersection = yp_2 * y2
        union = yp_2 + y2 - intersection

        jacard += (np.sum(intersection)/np.sum(union))  

        dice += (2. * np.sum(intersection) ) / (np.sum(yp_2) + np.sum(y2))

    
    jacard /= len(Y_test)
    dice /= len(Y_test)
    


    print('Jacard Index : '+str(jacard))
    print('Dice Coefficient : '+str(dice))
    

    fp = open('models/log.txt','a')
    fp.write(str(jacard)+'\n')
    fp.close()

    fp = open('models/best.txt','r')
    best = fp.read()
    fp.close()

    if(jacard>float(best)):
        print('***********************************************')
        print('Jacard Index improved from '+str(best)+' to '+str(jacard))
        print('***********************************************')
        fp = open('models/best.txt','w')
        fp.write(str(jacard))
        fp.close()

        saveModel(model)


"""
Training the Model
The model is trained and evaluated after each epochs
"""
def trainStep(model, X_train, Y_train, X_test, Y_test, epochs, batchSize):
    print('trainStep')
    for epoch in range(epochs):
        print('Epoch : {}'.format(epoch+1))
        model.fit(x=X_train, y=Y_train, batch_size=batchSize, epochs=1, verbose=1)    
        evaluateModel(model,X_test, Y_test,batchSize)
    return model 


"""
Define Model, Train and Evaluate
"""
from MultiResUNet import MultiResUnet
def train_model(X_train, Y_train, X_test, Y_test):
    model = MultiResUnet(height=192, width=256, n_channels=3)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coef, jacard, 'accuracy'])
    saveModel(model)
    fp = open('models/log.txt','w')
    fp.close()
    fp = open('models/best.txt','w')
    fp.write('-1.0')
    fp.close()
    trainStep(model, X_train, Y_train, X_test, Y_test, epochs=10, batchSize=10)






if __name__ == "__main__":
    main()