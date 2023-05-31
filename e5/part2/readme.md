## TensorFlow训练石头剪刀布数据集
1.  下载数据集：
    ```
    https://storage.googleapis.com/learning-datasets/rps.zip
    https://storage.googleapis.com/learning-datasets/rps-test-set.zip
    ```
2. 解压下载的数据集
    ```
    import os
    rock_dir = os.path.join('C:/Users/17764/Desktop/e5/rps/rps/rock')
    paper_dir = os.path.join('C:/Users/17764/Desktop/e5/rps/rps/paper')
    scissors_dir = os.path.join('C:/Users/17764/Desktop/e5/rps/rps/scissors')

    print('total training rock images:', len(os.listdir(rock_dir)))
    print('total training paper images:', len(os.listdir(paper_dir)))
    print('total training scissors images:', len(os.listdir(scissors_dir)))

    rock_files = os.listdir(rock_dir)
    print(rock_files[:10])

    paper_files = os.listdir(paper_dir)
    print(paper_files[:10])

    scissors_files = os.listdir(scissors_dir)
    print(scissors_files[:10])
    ```
3. 检测数据集的解压结果，打印相关信息
   ```
    total training rock images: 840
    total training paper images: 840
    total training scissors images: 840
    ['rock01-000.png', 'rock01-001.png', 'rock01-002.png', 'rock01-003.png', 'rock01-004.png', 'rock01-005.png', 'rock01-006.png', 'rock01-007.png', 'rock01-008.png', 'rock01-009.png']
    ['paper01-000.png', 'paper01-001.png', 'paper01-002.png', 'paper01-003.png', 'paper01-004.png', 'paper01-005.png', 'paper01-006.png', 'paper01-007.png', 'paper01-008.png', 'paper01-009.png']
    ['scissors01-000.png', 'scissors01-001.png', 'scissors01-002.png', 'scissors01-003.png', 'scissors01-004.png', 'scissors01-005.png', 'scissors01-006.png', 'scissors01-007.png', 'scissors01-008.png', 'scissors01-009.png']
    ```
4. 各打印两张石头剪刀布训练集图片
    ```
    %matplotlib inline

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    pic_index = 2

    next_rock = [os.path.join(rock_dir, fname) 
                    for fname in rock_files[pic_index-2:pic_index]]
    next_paper = [os.path.join(paper_dir, fname) 
                    for fname in paper_files[pic_index-2:pic_index]]
    next_scissors = [os.path.join(scissors_dir, fname) 
                    for fname in scissors_files[pic_index-2:pic_index]]

    for i, img_path in enumerate(next_rock+next_paper+next_scissors):
    #print(img_path)
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()
    ```
    <img src="../../images/e5/s1.png">
    <img src="../../images/e5/s2.png">
5. 调用TensorFlow的keras进行数据模型的训练和评估。Keras是开源人工神经网络库，TensorFlow集成了keras的调用接口，可以方便的使用。
   ```
   import tensorflow as tf
    import keras_preprocessing
    from keras_preprocessing import image
    from keras_preprocessing.image import ImageDataGenerator

    TRAINING_DIR = "C:/Users/17764/Desktop/e5/rps/rps/"
    training_datagen = ImageDataGenerator(
        rescale = 1./255,
            rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    VALIDATION_DIR = "C:/Users/17764/Desktop/e5/rps-test-set/rps-test-set/"
    validation_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150,150),
        class_mode='categorical',
    batch_size=126
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(150,150),
        class_mode='categorical',
    batch_size=126
    )

    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The third convolution
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The fourth convolution
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])


    model.summary()

    model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

    model.save("rps.h5")
   ```
   ```
   Found 2520 images belonging to 3 classes.
    Found 372 images belonging to 3 classes.
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)             (None, 148, 148, 64)      1792      
                                                                    
    max_pooling2d (MaxPooling2D  (None, 74, 74, 64)       0         
    )                                                               
                                                                    
    conv2d_1 (Conv2D)           (None, 72, 72, 64)        36928     
                                                                    
    max_pooling2d_1 (MaxPooling  (None, 36, 36, 64)       0         
    2D)                                                             
                                                                    
    conv2d_2 (Conv2D)           (None, 34, 34, 128)       73856     
                                                                    
    max_pooling2d_2 (MaxPooling  (None, 17, 17, 128)      0         
    2D)                                                             
                                                                    
    conv2d_3 (Conv2D)           (None, 15, 15, 128)       147584    
                                                                    
    max_pooling2d_3 (MaxPooling  (None, 7, 7, 128)        0         
    2D)                                                             
    ...
    Epoch 24/25
    20/20 [==============================] - 73s 4s/step - loss: 0.1128 - accuracy: 0.9603 - val_loss: 0.0880 - val_accuracy: 0.9570
    Epoch 25/25
    20/20 [==============================] - 85s 4s/step - loss: 0.0728 - accuracy: 0.9754 - val_loss: 0.1103 - val_accuracy: 0.9489
   ```
6. 完成模型训练之后，我们绘制训练和验证结果的相关信息。
   ```
   import matplotlib.pyplot as plt
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.show()
   ```
    <img src = "../../images/e5/output.png">
