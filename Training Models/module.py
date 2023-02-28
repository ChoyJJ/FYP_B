import tensorflow as tf
import csv
import os
import numpy as np
import shutil

class Model_Training:
    def pretrained_network(name='inceptionv3',img_height=300,img_width=300):
        """
        Insert the name of the pretrained model in small letters without spacing, the function will return the preprocessing layer and the model layer pre-trained on imagenet.
        """
        if name == 'inceptionv3':
            preprocessing = tf.keras.applications.inception_v3.preprocess_input
            pre_trained = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height,img_width,3))
        elif name == 'resnet50':
            preprocessing = tf.keras.applications.resnet50.preprocess_input
            pre_trained = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(img_height,img_width,3))
        elif name == 'vgg16':
            preprocessing = tf.keras.applications.vgg16.preprocess_input
            pre_trained = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(img_height,img_width,3))
        elif name == 'efficientnetv2m':
            preprocessing = tf.keras.applications.efficientnet_v2.preprocess_input
            pre_trained = tf.keras.applications.efficientnet_v2.EfficientNetV2M(weights='imagenet', include_top=False, input_shape=(img_height,img_width,3))
        elif name == 'efficientnetb3':
            preprocessing = tf.keras.applications.efficientnet.preprocess_input
            pre_trained = tf.keras.applications.efficientnet.EfficientNetB3(weights='imagenet', include_top=False, input_shape=(img_height,img_width,3))
        elif name == 'convnext':
            preprocessing = tf.keras.applications.convnext.preprocess_input
            pre_trained = tf.keras.applications.convnext.ConvNeXtBase(weights='imagenet', include_top=False, input_shape=(img_height,img_width,3))
        else:
            print('Unknown model has been inserted, please insert the model name in small letters without spacing. The pre-trained model has been defaulted to InceptionV3 in this model.')   
            preprocessing = tf.keras.applications.inception_v3.preprocess_input
            pre_trained = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height,img_width,3))   
        return preprocessing, pre_trained
    
    def training(model,train_dataset,validation_dataset,total_epochs,save_weights,patience,Earlystop,train_log,callbacks,optimiser):

        best_weights = None
        best_val_loss = float('inf')
        best_val_acc=float(0)
        all_history = {}
        all_history['train_loss'] = []
        all_history['train_acc'] = []
        all_history['val_loss'] = []
        all_history['val_acc'] = []
        min_lr = 1e-6
        if Earlystop == False:
            Earlystop = total_epochs
        if patience == False:
            patience = total_epochs
        if train_log:
            new_dict = {'Epoch','train_loss','train_accuracy','val_loss','val_accuracy'}
            directory = '/'.join(train_log.split('/')[:-1])
            if not os.path.exists(directory):
                os.makedirs(directory)
            log = open(train_log, 'a')
            writer = csv.writer(log)
            writer.writerow(new_dict)
            log.close()
        earlystop_counter = 0
        with tf.device('/GPU:0'):
            for i in range(total_epochs):
                # print("Epoch: {}".format(i))
                history = model.fit(train_dataset
                                    ,epochs=i+1
                                    ,initial_epoch=i
                                    ,validation_data=validation_dataset
                                    ,callbacks = [callbacks]
                                    # ,verbose=0
                                    )
                val_loss = history.history['val_loss'][-1]
                val_categorical_accuracy = history.history['val_categorical_accuracy'][-1]
                loss = history.history['loss'][-1]
                categorical_accuracy = history.history['categorical_accuracy'][-1]
                all_history['train_loss'].append(loss)
                all_history['train_acc'].append(categorical_accuracy)
                all_history['val_loss'].append(val_loss)
                all_history['val_acc'].append(val_categorical_accuracy)
                if train_log:
                    log = open(train_log, 'a')
                    writer = csv.writer(log)
                    new_dict = [i+1,history.history['loss'][-1],history.history['categorical_accuracy'][-1],val_loss,history.history['val_categorical_accuracy'][-1]]
                    writer.writerow(new_dict)
                    log.close()
                # val_improve = val_loss<best_val_loss and
                if val_loss < best_val_loss:
                    best_weights = model.get_weights()
                    model.save_weights(save_weights)
                    best_val_loss = val_loss
                    patience_counter = 0
                    earlystop_counter = 0
                else:
                    patience_counter += 1
                    earlystop_counter +=1
                # Check if we have reached the patience limit
                if patience_counter == patience:
                    # Load the best weights back into the model_fine
                    model.set_weights(best_weights)
                    # Reduce the learning rate
                    if optimiser.lr > min_lr:
                        optimiser.lr = optimiser.lr * 0.1
                    # Reset the patience counter
                    patience_counter = 0
                #Earlystop if model is not improving
                if earlystop_counter == Earlystop:
                    print(f'Early Stop at Epoch: {i+1}')
                    break
        return model, all_history

    def build_model(pretrained_model,trainable_layers=False,augmentation=True,Flatten='flatten',regulariser=False,load_weights = False,img_height=300,img_width=300,
                    optimiser=tf.keras.optimizers.Adam(),
                    losses = tf.keras.losses.CategoricalCrossentropy(),
                    metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.Recall(class_id=0),tf.keras.metrics.Recall(class_id=1)]):
        data_augmentation = tf.keras.Sequential([
          tf.keras.layers.RandomFlip('horizontal_and_vertical'),
          tf.keras.layers.RandomRotation((-0.2,0.2),fill_mode='nearest'),
          tf.keras.layers.RandomZoom(height_factor=(-0.2,0.7),width_factor=(-0.2,0.7),fill_mode='nearest'),
          tf.keras.layers.RandomTranslation(height_factor=(-0.1,0.1),width_factor=(-0.1,0.1),fill_mode='nearest')
          ])
        preprocessing,pre_trained = Model_Training.pretrained_network(name=pretrained_model,img_height=img_height,img_width=img_width)
        if trainable_layers:
            pre_trained.trainable = True
            for layer in pre_trained.layers:
              if isinstance(layer, tf.keras.layers.BatchNormalization):
                  layer.trainable = False
            for layer in pre_trained.layers[:-trainable_layers]:
              layer.trainable = False
        else:
            pre_trained.trainable = False
        tfinput = tf.keras.layers.Input(shape=(img_height,img_width,3))
        if augmentation:
            data_augment = data_augmentation(tfinput)
            pre_process = preprocessing(data_augment)
        else:
            pre_process = preprocessing(tfinput)
        tl_model=pre_trained(pre_process,training=False)
        if Flatten == 'flatten':
            flatten = tf.keras.layers.Flatten()(tl_model)
            x = tf.keras.layers.Dense(128,activation = 'relu')(flatten)
        elif Flatten == 'global_average_pooling':
            flatten = tf.keras.layers.GlobalAveragePooling2D()(tl_model)
            x = tf.keras.layers.Dense(2048,activation = 'relu')(flatten)
        elif Flatten == 'global_max_pooling':
            flatten = tf.keras.layers.GlobalMaxPooling2D()(tl_model)
            x = tf.keras.layers.Dense(2048,activation = 'relu')(flatten)
        x = tf.keras.layers.Dense(1024,activation='relu')(x)
        x= tf.keras.layers.Dropout(0.5)(x)
        if regulariser:
            x = tf.keras.layers.Dense(8,activation='relu',kernel_regularizer=regulariser)(x)
        else:
            x = tf.keras.layers.Dense(8,activation='relu')(x)
        output = tf.keras.layers.Dense(2, activation="softmax")(x)
        model = tf.keras.models.Model(tfinput,output)
        model.summary()
        if load_weights:
            model.load_weights(load_weights)
        model.compile(
                        optimizer= optimiser,
                        loss=losses,
                        metrics=metrics
                    ) 
        return model
    
    def main(           pretrained_model,
                        train_dataset,
                        validation_dataset,
                        epochs,
                        patience=False,
                        Earlystop=False,
                        augmentation = True,
                        flatten = 'flatten',
                        trainable_layers = False,
                        regulariser=False,
                        train_log=False,
                        load_weights = False,
                        save_weights = '/home/jj/FYP/Checkpoint/Placeholder/bestmodel',
                        learning_rate=1e-3,
                        optimiser=tf.keras.optimizers.Adam(),
                        losses = tf.keras.losses.CategoricalCrossentropy(),
                        metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.Recall(class_id=0),tf.keras.metrics.Recall(class_id=1)],
                        callbacks = [],
                        img_height=300,img_width=300):
        """
        pretrained_model is the name of the CNN model used in this transfer learning method
        """
        model = Model_Training.build_model(pretrained_model,trainable_layers,augmentation,flatten,regulariser,load_weights,img_height,img_width,optimiser,losses,metrics)
        optimiser.lr = learning_rate
        
        
        
        
        
        model, history = Model_Training.training(model,train_dataset,validation_dataset,epochs,save_weights,patience,Earlystop,train_log,callbacks,optimiser)


        return model, history
    
    def misclassified_images(   model,test_dataset,load_weights=False,
                                optimiser=tf.keras.optimizers.Adam(),
                                losses = tf.keras.losses.CategoricalCrossentropy(),
                                metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.Recall(class_id=0),tf.keras.metrics.Recall(class_id=1)]
                                ):
        if load_weights:
            model.load_weights(load_weights)
        model.compile(
                        optimizer= optimiser,
                        loss=losses,
                        metrics=metrics
                        ) 
        labels = ['benign','malignant']
        model.evaluate(test_dataset)
        test_predictions = model.predict(test_dataset)
        test_predicted_labels = np.argmax(test_predictions, axis=1)
        test_predicted_labels = [labels[i] for i in test_predicted_labels]
        test_true_labels = np.concatenate([y for x, y in test_dataset], axis=0)
        misclassified_indices = []
        for i,y in enumerate(test_true_labels):
            if y[1] == 1:
                if test_predicted_labels[i] != labels[1]:
                    misclassified_indices.append(i)
            elif y[0] == 1:
                if test_predicted_labels[i] != labels[0]:
                    misclassified_indices.append(i)
        test_filenames = test_dataset.file_paths
        misclassified_images = [test_filenames[i] for i in misclassified_indices]

        return misclassified_images
    
    def store_misclassified(model,test_dataset,dst,load_weights=False,
                            optimiser=tf.keras.optimizers.Adam(),
                            losses = tf.keras.losses.CategoricalCrossentropy(),
                            metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.Recall(class_id=0),tf.keras.metrics.Recall(class_id=1)]
                            ):
        misclassified = Model_Training.misclassified_images( model,test_dataset,load_weights,optimiser,losses,metrics)
        dest_dir_path = dst
        os.makedirs(dest_dir_path, exist_ok=True)
        os.makedirs(dest_dir_path+'benign/', exist_ok=True)
        os.makedirs(dest_dir_path+'malignant/', exist_ok=True)
        for path in misclassified:
            # get the filename
            filename = os.path.basename(path)
            if path.split('/')[-2] == 'malignant':
            # copy the file to the destination directory
                dest_path = os.path.join(dest_dir_path+'malignant', filename)
                shutil.copyfile(path, dest_path)
            if path.split('/')[-2] == 'benign':
            # copy the file to the destination directory
                dest_path = os.path.join(dest_dir_path+'benign', filename)
                shutil.copyfile(path, dest_path)
        return misclassified
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



