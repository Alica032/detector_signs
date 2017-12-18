from keras.layers import Input, Dense, Dropout, Flatten, Lambda, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers import concatenate as concat_layers
from keras.models import Model

import cv2

def pre_process_image(image):
    image[:,:,0] = cv2.equalizeHist(image[:,:,0])
    image[:,:,1] = cv2.equalizeHist(image[:,:,1])
    image[:,:,2] = cv2.equalizeHist(image[:,:,2])
    image = image/255.-.5
    
    return image

def model_classify():
    
    pool_size = 2 # we will use 2x2 pooling throughout
    drop_prob_1 = 0.5 # dropout after pooling with probability 0.25
    N_classes = 43

    ## Convlayer 0
    filter_size0 = 1
    num_filters0 = 3

    ## Convlayer 1
    filter_size1 = 5
    num_filters1 = 32

    ## Convlayer 2
    filter_size2 = 5
    num_filters2 = 64

    ## Convlayer 3
    filter_size3 = 5
    num_filters3 = 128

    ## FC_size
    fc_size1 = 1024
    ## FC_size
    fc_size2 = 1024

    ## Dropout
    drop_prob = 0.5
    input_shape = (48,48,3)

    inp = Input(shape=input_shape)
    
    conv_0 = Conv2D(num_filters0, (filter_size0, filter_size0), padding='same', activation='relu')(inp)

    conv_1 = Conv2D(num_filters1, (filter_size1, filter_size1), padding='same', activation='relu')(conv_0)
    conv_1 = Conv2D(num_filters1, (filter_size1, filter_size1), padding='same', activation='relu')(conv_1)

    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
    drop_1 = Dropout(drop_prob_1)(pool_1)


    conv_2 = Conv2D(num_filters2, (filter_size2, filter_size2), padding='same', activation='relu')(drop_1)
    conv_2 = Conv2D(num_filters2, (filter_size2, filter_size2), padding='same', activation='relu')(conv_2)

    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    drop_2 = Dropout(drop_prob_1)(pool_2)


    conv_3 = Conv2D(num_filters3, (filter_size3, filter_size3), padding='same', activation='relu')(drop_2)
    conv_3 = Conv2D(num_filters3, (filter_size3, filter_size3), padding='same', activation='relu')(conv_3)

    pool_3 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_3)
    drop_3 = Dropout(drop_prob_1)(pool_3)

    l1 = Flatten()(drop_1)
    l2 = Flatten()(drop_2)
    l3 = Flatten()(drop_3)

    flat = concat_layers([l1, l2, l3])

    hidden1 = Dense(fc_size1, activation='relu')(flat)
    drop_4 = Dropout(drop_prob_1)(hidden1)

    hidden2 = Dense(fc_size1, activation='relu')(drop_4)
    drop_5 = Dropout(drop_prob_1)(hidden2)

    out = Dense(N_classes, activation='softmax')(drop_5)
    return Model(inputs=inp, outputs=out)
