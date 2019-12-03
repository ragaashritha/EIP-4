1. Validation accuracy for base network : 82.09

2. 
#Depthwise seperable convolution
from keras.layers import SeparableConv2D
from keras.layers.normalization import BatchNormalization
model1 = Sequential()

#output shape: 32x32 , receptive field: 3
model1.add(SeparableConv2D(48, 3,3,border_mode='same',input_shape=(32, 32, 3)))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(0.20))

#output shape: 30x30 , receptive field: 5
model1.add(SeparableConv2D(96, 3,3))#30
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(0.20))

#output shape: 30x30 , receptive field: 7
model1.add(SeparableConv2D(48, 3,3,border_mode='same'))
model1.add(Activation('relu'))

#output shape: 15x15 , receptive field: 8
model1.add(MaxPooling2D(pool_size=(2, 2)))

#output shape: 15x15 , receptive field: 12
model1.add(SeparableConv2D(48, 3,3,border_mode='same'))#15
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(0.15))

#output shape: 13x13 , receptive field: 16
model1.add(SeparableConv2D(96, 3,3))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(0.20))

#output shape: 13x13 , receptive field: 20
model1.add(SeparableConv2D(48, 3,3,border_mode='same'))
model1.add(Activation('relu'))

#output shape: 13x13 , receptive field: 24
model1.add(SeparableConv2D(96, 3,3,border_mode='same'))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(0.15))

#output shape: 11x11 , receptive field: 28
model1.add(SeparableConv2D(192, 3,3))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(0.15))

#output shape: 11x11 , receptive field: 32 
model1.add(SeparableConv2D(48,3,3,border_mode='same'))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(0.15))

#output shape: 9x9 , receptive field: 36
model1.add(SeparableConv2D(48,3,3))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(0.15))

#output shape: 7x7 , receptive field: 40
model1.add(SeparableConv2D(96,3,3))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(0.10))

#output shape: 5x5 , receptive field: 44
model1.add(SeparableConv2D(32,3,3))
model1.add(BatchNormalization())

model1.add(SeparableConv2D(10,5,5))

model1.add(Flatten())
model1.add(Activation('softmax'))
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 



3.
Epoch 1/50
390/390 [==============================] - 26s 66ms/step - loss: 1.7096 - acc: 0.3620 - val_loss: 1.8922 - val_acc: 0.3782
Epoch 2/50
390/390 [==============================] - 22s 57ms/step - loss: 1.2254 - acc: 0.5583 - val_loss: 1.6370 - val_acc: 0.4820
Epoch 3/50
390/390 [==============================] - 22s 57ms/step - loss: 1.0566 - acc: 0.6210 - val_loss: 1.0905 - val_acc: 0.6246
Epoch 4/50
390/390 [==============================] - 22s 56ms/step - loss: 0.9537 - acc: 0.6614 - val_loss: 1.0465 - val_acc: 0.6389
Epoch 5/50
390/390 [==============================] - 22s 56ms/step - loss: 0.8714 - acc: 0.6919 - val_loss: 1.0607 - val_acc: 0.6579
Epoch 6/50
390/390 [==============================] - 22s 57ms/step - loss: 0.8136 - acc: 0.7108 - val_loss: 0.9427 - val_acc: 0.6796
Epoch 7/50
390/390 [==============================] - 22s 57ms/step - loss: 0.7606 - acc: 0.7296 - val_loss: 0.8426 - val_acc: 0.7168
Epoch 8/50
390/390 [==============================] - 22s 56ms/step - loss: 0.7244 - acc: 0.7456 - val_loss: 0.8803 - val_acc: 0.6968
Epoch 9/50
390/390 [==============================] - 22s 57ms/step - loss: 0.6848 - acc: 0.7597 - val_loss: 0.9853 - val_acc: 0.6901
Epoch 10/50
390/390 [==============================] - 22s 57ms/step - loss: 0.6614 - acc: 0.7681 - val_loss: 1.1413 - val_acc: 0.6275
Epoch 11/50
390/390 [==============================] - 22s 57ms/step - loss: 0.6421 - acc: 0.7733 - val_loss: 0.6974 - val_acc: 0.7623
Epoch 12/50
390/390 [==============================] - 22s 57ms/step - loss: 0.6202 - acc: 0.7814 - val_loss: 0.8953 - val_acc: 0.7035
Epoch 13/50
390/390 [==============================] - 22s 56ms/step - loss: 0.5995 - acc: 0.7896 - val_loss: 0.8584 - val_acc: 0.7135
Epoch 14/50
390/390 [==============================] - 22s 56ms/step - loss: 0.5780 - acc: 0.7974 - val_loss: 0.7181 - val_acc: 0.7597
Epoch 15/50
390/390 [==============================] - 22s 56ms/step - loss: 0.5650 - acc: 0.8003 - val_loss: 0.7387 - val_acc: 0.7547
Epoch 16/50
390/390 [==============================] - 22s 57ms/step - loss: 0.5523 - acc: 0.8047 - val_loss: 0.7932 - val_acc: 0.7408
Epoch 17/50
390/390 [==============================] - 22s 56ms/step - loss: 0.5333 - acc: 0.8131 - val_loss: 0.7371 - val_acc: 0.7513
Epoch 18/50
390/390 [==============================] - 22s 57ms/step - loss: 0.5276 - acc: 0.8146 - val_loss: 0.5908 - val_acc: 0.7987
Epoch 19/50
390/390 [==============================] - 22s 56ms/step - loss: 0.5155 - acc: 0.8166 - val_loss: 0.7167 - val_acc: 0.7647
Epoch 20/50
390/390 [==============================] - 22s 57ms/step - loss: 0.5036 - acc: 0.8213 - val_loss: 0.9234 - val_acc: 0.7263
Epoch 21/50
390/390 [==============================] - 22s 56ms/step - loss: 0.4883 - acc: 0.8263 - val_loss: 0.6920 - val_acc: 0.7759
Epoch 22/50
390/390 [==============================] - 22s 56ms/step - loss: 0.4823 - acc: 0.8297 - val_loss: 0.8573 - val_acc: 0.7327
Epoch 23/50
390/390 [==============================] - 22s 57ms/step - loss: 0.4778 - acc: 0.8311 - val_loss: 0.9024 - val_acc: 0.7248
Epoch 24/50
390/390 [==============================] - 22s 57ms/step - loss: 0.4692 - acc: 0.8358 - val_loss: 0.7356 - val_acc: 0.7615
Epoch 25/50
390/390 [==============================] - 22s 58ms/step - loss: 0.4618 - acc: 0.8378 - val_loss: 0.6256 - val_acc: 0.7970
Epoch 26/50
390/390 [==============================] - 23s 58ms/step - loss: 0.4511 - acc: 0.8406 - val_loss: 0.7233 - val_acc: 0.7626
Epoch 27/50
390/390 [==============================] - 22s 57ms/step - loss: 0.4483 - acc: 0.8416 - val_loss: 0.6721 - val_acc: 0.7843
Epoch 28/50
390/390 [==============================] - 22s 57ms/step - loss: 0.4373 - acc: 0.8454 - val_loss: 0.7383 - val_acc: 0.7708
Epoch 29/50
390/390 [==============================] - 23s 58ms/step - loss: 0.4284 - acc: 0.8481 - val_loss: 0.7193 - val_acc: 0.7765
Epoch 30/50
390/390 [==============================] - 22s 56ms/step - loss: 0.4280 - acc: 0.8489 - val_loss: 0.6525 - val_acc: 0.7891
Epoch 31/50
390/390 [==============================] - 22s 56ms/step - loss: 0.4181 - acc: 0.8507 - val_loss: 0.5834 - val_acc: 0.8081
Epoch 32/50
390/390 [==============================] - 22s 57ms/step - loss: 0.4164 - acc: 0.8532 - val_loss: 0.8563 - val_acc: 0.7398
Epoch 33/50
390/390 [==============================] - 22s 56ms/step - loss: 0.4127 - acc: 0.8531 - val_loss: 0.6066 - val_acc: 0.8026
Epoch 34/50
390/390 [==============================] - 22s 56ms/step - loss: 0.4038 - acc: 0.8565 - val_loss: 0.6354 - val_acc: 0.7988
Epoch 35/50
390/390 [==============================] - 22s 56ms/step - loss: 0.3975 - acc: 0.8594 - val_loss: 0.5970 - val_acc: 0.8064
Epoch 36/50
390/390 [==============================] - 22s 56ms/step - loss: 0.3964 - acc: 0.8589 - val_loss: 0.6564 - val_acc: 0.7898
Epoch 37/50
390/390 [==============================] - 22s 56ms/step - loss: 0.3898 - acc: 0.8618 - val_loss: 0.6534 - val_acc: 0.8010
Epoch 38/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3844 - acc: 0.8643 - val_loss: 0.6787 - val_acc: 0.7837
Epoch 39/50
390/390 [==============================] - 22s 56ms/step - loss: 0.3822 - acc: 0.8634 - val_loss: 0.6092 - val_acc: 0.8070
Epoch 40/50
390/390 [==============================] - 22s 56ms/step - loss: 0.3741 - acc: 0.8667 - val_loss: 0.6393 - val_acc: 0.7957
Epoch 41/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3724 - acc: 0.8665 - val_loss: 0.7603 - val_acc: 0.7735
Epoch 42/50
390/390 [==============================] - 22s 56ms/step - loss: 0.3671 - acc: 0.8700 - val_loss: 0.7446 - val_acc: 0.7632
Epoch 43/50
390/390 [==============================] - 22s 56ms/step - loss: 0.3671 - acc: 0.8704 - val_loss: 0.5628 - val_acc: 0.8192
Epoch 44/50
390/390 [==============================] - 22s 56ms/step - loss: 0.3619 - acc: 0.8707 - val_loss: 0.6078 - val_acc: 0.8142
Epoch 45/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3560 - acc: 0.8726 - val_loss: 0.6027 - val_acc: 0.8142
Epoch 46/50
390/390 [==============================] - 22s 56ms/step - loss: 0.3535 - acc: 0.8737 - val_loss: 0.6003 - val_acc: 0.8132
Epoch 47/50
390/390 [==============================] - 22s 56ms/step - loss: 0.3537 - acc: 0.8741 - val_loss: 0.7595 - val_acc: 0.7717
Epoch 48/50
390/390 [==============================] - 22s 56ms/step - loss: 0.3480 - acc: 0.8753 - val_loss: 0.6011 - val_acc: 0.8173
Epoch 49/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3424 - acc: 0.8800 - val_loss: 0.6375 - val_acc: 0.8065
Epoch 50/50
390/390 [==============================] - 22s 57ms/step - loss: 0.3448 - acc: 0.8779 - val_loss: 0.5409 - val_acc: 0.8300
Model took 1108.42 seconds to train

Accuracy on test data is: 83.00
