## Setting ##

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def mbConv_block(
    input_data,repeat_num, kernel_size,input_filter,output_filter,expand_ratio,se_ratio,strides, drop_ratio):
    
    
    # expansion phase (1,1)
    expanded_filter =  input_filter * expand_ratio
    x = tf.keras.layers.Conv2D(expanded_filter, 1,  padding='same',  use_bias=False)(input_data)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)

    # Depthwise convolution phase (k,k)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size, strides,  padding='same',  use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    
    # Squeeze and excitation phase (1,1)
    squeezed_filter = max(1, int(input_filter * se_ratio))
    se = tf.keras.layers.GlobalAveragePooling2D()(x)
    se = tf.keras.layers.Reshape((1, 1, expanded_filter))(se)
    se = tf.keras.layers.Conv2D(squeezed_filter,1)(se)
    se = tf.keras.layers.Activation('swish')(se)
    se = tf.keras.layers.Conv2D(expanded_filter,1, activation='sigmoid')(se)
    x = tf.keras.layers.multiply([x, se])
        
    # Output phase (1,1)
    x = tf.keras.layers.Conv2D(output_filter, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
        
    if repeat_num == 1:
        pass
    else:
        x = tf.keras.layers.Dropout(drop_ratio)(x)

    return x
    
class Mycallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        
        if logs.get('val_accuracy') > 0.80 or logs.get('accuracy') > 0.9999 :
            print('done!')
            self.model.stop_training = True
            
## Model ##

def Model():
    
    model_input =  tf.keras.layers.Input(shape = (224,224,3))
    
    # stem
    x = tf.keras.layers.Conv2D(32, (3,3), padding='same',strides = (2,2))(model_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    
    # mbConv_blocks
    MBConv1_1 = mbConv_block(x,3,1,32,16,1,0.25,1,0.2)
    
    MBConv6_2_1 = mbConv_block(MBConv1_1,3,1,16,24,6,0.25,2,0.2)
    MBConv6_2_2 = mbConv_block(MBConv1_1,3,2,16,24,6,0.25,2,0.2)
    x = tf.keras.layers.add([MBConv6_2_1, MBConv6_2_2])
    
    MBConv6_3_1 = mbConv_block(x,5,1,24,40,6,0.25,2,0.2)
    MBConv6_3_2 = mbConv_block(x,5,2,24,40,6,0.25,2,0.2)
    x = tf.keras.layers.add([MBConv6_3_1, MBConv6_3_2])
    
    MBConv6_4_1 = mbConv_block(x,3,1,40,80,6,0.25,2,0.2)
    MBConv6_4_2 =  mbConv_block(x,3,2,40,80,6,0.25,2,0.2)
    MBConv6_4_3 =  mbConv_block(x,3,3,40,80,6,0.25,2,0.2)
    x = tf.keras.layers.add([MBConv6_4_1, MBConv6_4_2, MBConv6_4_3])
    
    MBConv6_5_1 = mbConv_block(x,5,1,80,112,6,0.25,1,0.2)
    MBConv6_5_2 =  mbConv_block(x,5,2,80,112,6,0.25,1,0.2)
    MBConv6_5_3 =  mbConv_block(x,5,3,80,112,6,0.25,1,0.2)
    x = tf.keras.layers.add([MBConv6_5_1, MBConv6_5_2, MBConv6_5_3])
    
    
    MBConv6_6_1 = mbConv_block(x,5,1,112,192,6,0.25,2,0.2)
    MBConv6_6_2 =  mbConv_block(x,5,2,112,192,6,0.25,2,0.2)
    MBConv6_6_3 =  mbConv_block(x,5,3,112,192,6,0.25,2,0.2)
    MBConv6_6_4 =  mbConv_block(x,5,4,112,192,6,0.25,2,0.2)
    x = tf.keras.layers.add([MBConv6_6_1, MBConv6_6_2, MBConv6_6_3, MBConv6_6_4])
    
    MBConv6_7 = mbConv_block(x,3,1,192,320,6,0.25,1,0.2)
    
    # output
    pool = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(100, activation='softmax')(pool)
    
    
    model = tf.keras.Model(model_input, output)

    return model

## Train ##

model = Model()
callbacks = Mycallbacks()
model.compile(optimizer='Adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_images,train_labels, epochs = 100, batch_size = 100, validation_data=(test_images, test_labels),
          validation_batch_size= 100, validation_steps= 10, callbacks= [callbacks])
