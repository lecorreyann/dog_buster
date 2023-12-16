import tensorflow as tf
mopdel_1 = True
def model_2_init():
    input_shape=(255,255,3)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32,kernel_size=(5,5),input_shape=input_shape,padding='valid',activation='relu'))
    model.add(tf.keras.layers.Conv2D(64,kernel_size=(5,5),padding='valid',activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3)))
    model.add(tf.keras.layers.Conv2D(128,kernel_size=(4,4),padding='valid',activation='relu'))
    model.add(tf.keras.layers.Conv2D(160,kernel_size=(4,4),padding='valid',activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3)))
    model.add(tf.keras.layers.Conv2D(256,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(64,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(10,activation='softmax'))

    model.compile(optimizer='rmspror',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
