## Setting ##

import tensorflow as tf

def Identity_block(x, filters):
    
    F1, F2 = filters

    shortcut = x
    
    x = tf.keras.layers.Conv2D(F1, (3,3), strides = 1, padding = 'same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x= tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(F2, (3,3), strides = 1, padding = 'same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.add([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)

    return x
    
def Convolution_block(x, filters):
    
    F1, F2 = filters
    
    shortcut = tf.keras.layers.Conv2D(F2, (3,3), strides = 1, padding = 'same', kernel_initializer='he_normal')(x)
        
    x = tf.keras.layers.Conv2D(F1, (3,3), strides = 1, padding = 'same',kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Conv2D(F2, (3,3), strides = 1, padding = 'same',kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)    
    
    x = tf.keras.layers.add([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
        
    return x
    
class Mycallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        
        if logs.get('val_accuracy') > 0.80 or logs.get('accuracy') > 0.9999 :
            print('done!')
            self.model.stop_training = True
            
## Model ##

def Model():
    model_input =  tf.keras.layers.Input(shape = (224,224,3))
    conv0 = tf.keras.layers.Conv2D(64, (3,3), padding = 'same', strides=1, kernel_initializer='he_normal')(model_input)
    bn0 = tf.keras.layers.BatchNormalization()(conv0)
    act1 = tf.keras.layers.Activation('relu')(bn0)
    
    block_1_1 = Identity_block(act1, (64, 64))
    block_1_2 = Identity_block(block_1_1, (64, 64))
    max_pool_1 = tf.keras.layers.MaxPooling2D((2,2))(block_1_2)
    
    block_2_1 = Convolution_block(max_pool_1, (128,128))
    block_2_2 = Identity_block(block_2_1, (128,128))
    max_pool_2 = tf.keras.layers.MaxPooling2D((2,2))(block_2_2)    
    
    block_3_1 = Convolution_block(max_pool_2, (256,256))
    block_3_2 = Identity_block(block_3_1,(256,256))
    max_pool_3 = tf.keras.layers.MaxPooling2D((2,2))(block_3_2)    
    
    block_4_1 = Convolution_block(max_pool_3, (512,512))
    block_4_2 = Identity_block(block_4_1, (512,512))
    
    argpool = tf.keras.layers.GlobalAveragePooling2D()(block_4_2)
    output = tf.keras.layers.Dense(1000, activation=tf.nn.softmax, kernel_initializer='he_normal')(argpool)
    
    model = tf.keras.Model(model_input, output)
    
    return model

## Train ##

model = Model()
callbacks = Mycallbacks()
model.compile(optimizer= 'Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_images,train_labels, epochs = 5, batch_size = 100, validation_data=(test_images, test_labels),
        validation_batch_size= 100, validation_steps= 10, callbacks= [callbacks])
