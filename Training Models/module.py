import tensorflow as tf
import csv
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

class Model_Training:


    def pretrained_network(self,name='inceptionv3',img_height=300,img_width=300):
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
    
    def training(self,model,train_dataset,validation_dataset,total_epochs,save_weights,patience,Earlystop,train_log,callbacks,optimiser,initialise,learning_rate,
                 warmup_epochs,cosine_epochs=0,min_lr=0,class_weight=None):

        best_weights = None
        best_val_loss = float('inf')
        best_val_acc=float(0)
        all_history = {}
        all_history['train_loss'] = []
        all_history['train_acc'] = []
        all_history['val_loss'] = []
        all_history['val_acc'] = []
        if Earlystop == False:
            Earlystop = total_epochs
        # if patience == False:
            # patience = total_epochs
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
        if warmup_epochs>0:
            optimiser.lr = initialise
        else:
            optimiser.lr = learning_rate
            lr = learning_rate
        with tf.device('/GPU:0'):
            for i in range(total_epochs):
                # print("Epoch: {}".format(i))
                history = model.fit(train_dataset
                                    ,epochs=i+1
                                    ,initial_epoch=i
                                    ,validation_data=validation_dataset
                                    ,callbacks = [callbacks]
                                    ,class_weight=class_weight
                                    # ,verbose=0
                                    )
                
                
                loss = history.history['loss'][-1]
                categorical_accuracy = history.history['categorical_accuracy'][-1]
                all_history['train_loss'].append(loss)
                all_history['train_acc'].append(categorical_accuracy)
                if validation_dataset != None:
                    val_loss = history.history['val_loss'][-1]
                    val_categorical_accuracy = history.history['val_categorical_accuracy'][-1]
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
                        # best_weights = model.get_weights()
                        model.save_weights(save_weights)
                        best_val_loss = val_loss
                        patience_counter = 0
                        earlystop_counter = 0
                    else:
                        patience_counter += 1
                        earlystop_counter +=1
                else:
                    if train_log:
                        log = open(train_log, 'a')
                        writer = csv.writer(log)
                        new_dict = [i+1,history.history['loss'][-1],history.history['categorical_accuracy'][-1],0,0]
                        writer.writerow(new_dict)
                        log.close()
                    # val_improve = val_loss<best_val_loss and
                    if loss < best_val_loss:
                        # best_weights = model.get_weights()
                        model.save_weights(save_weights)
                        best_val_loss = loss
                        patience_counter = 0
                        earlystop_counter = 0
                    else:
                        patience_counter += 1
                        earlystop_counter +=1
                    all_history['val_loss'].append(0)
                    all_history['val_acc'].append(0)
                # Initialise weight with lower learning rate
                # warmup_epochs = 5
                # cosine_epochs = total_epochs - warmup_epochs
                if initialise and i<warmup_epochs:
                    lr = (learning_rate- min_lr) / warmup_epochs * (i+1) + min_lr
                else:
                    if patience == False and cosine_epochs>0:
                        progress = (i+1 - warmup_epochs) / (cosine_epochs)
                        lr = min_lr + 0.5 * (learning_rate - min_lr) * (1 + np.cos(np.pi * progress))
                        if i - warmup_epochs >= cosine_epochs:
                            lr = min_lr
                    elif patience_counter >= patience:
                        # Reduce the learning rate
                        # if optimiser.lr > min_lr:
                        lr = optimiser.lr * 0.1
                        # else:
                            # lr = min_lr
                        # Reset the patience counter
                        patience_counter = 0
                optimiser.lr = lr    
                    
                #Earlystop if model is not improving
                if earlystop_counter >= Earlystop:
                    print(f'Early Stop at Epoch: {i+1}')
                    break
        return model, all_history

    def build_model(self,pretrained_model,trainable_layers=False,augmentation=True,flatten='global_average_pooling',regulariser=None,load_weights = False,img_height=300,img_width=300,
                    optimiser=tf.keras.optimizers.Adam(),
                    losses = tf.keras.losses.CategoricalCrossentropy(),
                    metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.Recall(class_id=0),tf.keras.metrics.Recall(class_id=1)],
                    training = True
                    ):
        data_augmentation = tf.keras.Sequential([
          tf.keras.layers.RandomFlip('horizontal_and_vertical'),
          tf.keras.layers.RandomRotation((-0.1,0.1),fill_mode='reflect'),
          tf.keras.layers.RandomZoom(height_factor=(-0.5,0.5),width_factor=(-0.5,0.5),fill_mode='reflect'),
          tf.keras.layers.RandomTranslation(height_factor=(-0.2,0.2),width_factor=(-0.2,0.2),fill_mode='reflect')
          ])
        preprocessing,pre_trained = self.pretrained_network(name=pretrained_model,img_height=img_height,img_width=img_width)
        if trainable_layers:
            pre_trained.trainable = True
            for layer in pre_trained.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
                if isinstance(layer, tf.keras.layers.LayerNormalization):
                    layer.trainable = False

            for layer in pre_trained.layers[:-trainable_layers]:
              layer.trainable = False
        else:
            pre_trained.trainable = False
        tfinput = tf.keras.layers.Input(shape=(img_height,img_width,3))
        if augmentation:
            data_augment = data_augmentation(tfinput)
            # tl_model=pre_trained(data_augment,training=False)
            pre_process = preprocessing(data_augment)
        else:
            pre_process = preprocessing(tfinput)
        tl_model=pre_trained(pre_process,training=training)
        ReLU = tf.keras.layers.ReLU(negative_slope=0.1)
        if flatten == 'flatten':
            # pooling = tf.keras.layers.AveragePooling2D((4,4),strides=(2,2),padding='same')(tl_model)
            x = tf.keras.layers.Flatten()(tl_model)
            # x= tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(1024)(x)
            # x = ReLU(x)
        elif flatten == 'global_average_pooling':
            x = tf.keras.layers.GlobalAveragePooling2D()(tl_model)
            # x= tf.keras.layers.Dropout(0.6)(x)
            # x= tf.keras.layers.Dropout(0.5)(flatten)
            x = tf.keras.layers.Dense(1024,activation = 'relu')(x)
            # x = ReLU(x)
        elif flatten == 'global_max_pooling':
            x = tf.keras.layers.GlobalMaxPooling2D()(tl_model)
            x = tf.keras.layers.Dense(1024,activation = 'relu')(x)
        x = ReLU(x)
        
        x = tf.keras.layers.Dense(1024)(x)
        x = ReLU(x)
        if augmentation:
            x= tf.keras.layers.Dropout(0.5)(x,training=training)
        x = tf.keras.layers.Dense(1024,kernel_regularizer=regulariser)(x)
        x = ReLU(x)
        output = tf.keras.layers.Dense(2, activation="softmax",kernel_initializer="random_uniform")(x)
        model = tf.keras.models.Model(tfinput,output)
        # model.summary()
        if load_weights != False:
            model.load_weights(load_weights)
        model.compile(
                        optimizer= optimiser,
                        loss=losses,
                        metrics=metrics
                    ) 
        return model
    
    def main(           self,pretrained_model,
                        train_dataset,
                        validation_dataset,
                        epochs,
                        patience=False,
                        Earlystop=False,
                        augmentation = True,
                        flatten = 'global_average_pooling',
                        trainable_layers = False,
                        regulariser=None,
                        train_log=False,
                        load_weights = False,
                        save_weights = '/home/jj/FYP/Checkpoint/Placeholder/bestmodel',
                        learning_rate=1e-3,
                        init_lr=False,
                        min_lr = 0,
                        warmup=0,
                        cosine=0,
                        optimiser=tf.keras.optimizers.Adam(),
                        losses = tf.keras.losses.CategoricalCrossentropy(),class_weight=None,
                        metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.Recall(class_id=0),tf.keras.metrics.Recall(class_id=1)],
                        callbacks = [],
                        img_height=300,img_width=300,
                        plot=True):
        """
        pretrained_model is the name of the CNN model used in this transfer learning method
        """
        if  cosine > epochs - warmup :
            cosine = epochs - warmup
            print(f'Cosine learning rate epochs is changed to: {cosine}')
        if min_lr <= 0:
            if cosine > 0:
                min_lr = learning_rate *0.01
                
        model = self.build_model(pretrained_model,trainable_layers,augmentation,flatten,regulariser,load_weights,img_height,img_width,optimiser,losses,metrics)
        # optimiser.lr = learning_rate
        # print(patience)
        
        
        model, history = self.training(model,train_dataset,validation_dataset,epochs,save_weights,patience,Earlystop,train_log,
                                       callbacks,optimiser,init_lr,learning_rate,warmup,cosine,min_lr,class_weight)
        if plot:
            plt.figure(figsize=(30,10))
            plt.subplot(221)
            plt.plot(history['train_loss'])
            plt.subplot(222)
            plt.plot(history['val_loss'])
            plt.subplot(223)
            plt.plot(history['train_acc'])
            plt.subplot(224)
            plt.plot(history['val_acc'])


        return model, history
    
    def misclassified_images(   self,model,test_dataset,load_weights=False,
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
        # model.evaluate(test_dataset)
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
    
    def store_misclassified(self,model,test_dataset,dst,load_weights=False,
                            optimiser=tf.keras.optimizers.Adam(),
                            losses = tf.keras.losses.CategoricalCrossentropy(),
                            metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.Recall(class_id=0),tf.keras.metrics.Recall(class_id=1)]
                            ):
        misclassified = self.misclassified_images( model,test_dataset,load_weights,optimiser,losses,metrics)
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
    
    
    def datasets(self,data_loc,label_mode='categorical',img_height: int=300,img_width: int=300, batch_size: int=300):
        return tf.keras.utils.image_dataset_from_directory(data_loc,
                                                            label_mode=label_mode,
                                                            image_size=(img_height, img_width),
                                                            batch_size=batch_size)
    
    def confusion_matrix(self,test_data,predictions,plot=True):
        '''
        Directly import the test dataset used for model prediction, and insert the prediction from model.predict()
        Returns: Confusion Metrics as a 2x2 array
        '''
        labels = []
        for x,y in test_data.unbatch():
            labels.append(y.numpy().astype('uint8'))
        onehot = np.argmax(predictions,axis=1)
        pred = np.zeros((len(onehot),2),dtype=int)
        for i,x in enumerate(onehot):
            pred[i,x]=1
        confusion_matrix = np.zeros((2,2),dtype=int)
        for i,x in enumerate(labels):
            if x[0] == 1:
                if pred[i,0] == 1:
                    confusion_matrix[0,0]+=1
                else:
                    confusion_matrix[0,1]+=1
            elif x[1] ==1:
                if pred[i,1] == 1:
                    confusion_matrix[1,1]+=1
                else:
                    confusion_matrix[1,0]+=1
        if plot:
            classes = ['Benign', 'Malignant']
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(confusion_matrix, cmap='Blues')

            # Add axis labels and tick marks
            ax.set_xticks(np.arange(len(classes)))
            ax.set_yticks(np.arange(len(classes)))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)

            # Add annotations
            thresh = (confusion_matrix.max()-confusion_matrix.min()) / 2 + confusion_matrix.min()
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                            ha='center', va='center',
                            color='white' if confusion_matrix[i, j] > thresh else 'black')

            ax.set_title('Confusion Matrix')
            fig.colorbar(im)

            # Display the plot
            plt.show()
        return confusion_matrix
    
    def AUC(self,test,prediction):
        '''
        Directly import the test dataset used for model prediction, and insert the prediction from model.predict()
        Returns: AUC score 
        '''
        y_true = np.zeros((2,))
        for x,y in test.unbatch():
            y_true = np.vstack((y_true,y.numpy()))
        y_true = y_true[1:]
        y_true.astype('uint8')
        y_pred = prediction
        # y_true and y_pred are numpy arrays containing the true labels and predicted probabilities for each class, respectively
        # n_classes = y_true.shape[1]
        # fpr = dict()
        # tpr = dict()
        # roc_auc = dict()
        # onehot = np.argmax(prediction,axis=1)
        pred = prediction[:,1]
        truth = np.zeros((len(pred),))
        for i,x in enumerate(y_true):
            if x[1] == 1:
                truth[i]=1
        # for i in range(y_true.shape[1]):
            # fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            # roc_auc[i] = auc(fpr[i], tpr[i])
        fpr, tpr,_ = roc_curve(truth,pred)
        roc_auc = auc(fpr,tpr)

        # Compute micro-average ROC curve and AUC
        # fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
        # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot ROC curves for each class and micro-average
        plt.figure()
        lw = 2
        # plt.plot(fpr["micro"], tpr["micro"], color='darkorange',
                #  lw=lw, label='micro-average ROC curve (area = {0:0.2f})'
                #  ''.format(roc_auc["micro"]))
        # for i in range(y_true.shape[1]):
        plt.plot(fpr, tpr, lw=lw,
                    label='ROC curve (area = {1:0.4f})'
                    ''.format(1, roc_auc))

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()
        # auc_score = roc_auc_score(y_true[:,], prediction[:,1])

        return roc_auc

    def cross_validation(self,pretrained_model,
                        dataset,
                        epochs,
                        k_fold,
                        patience=False,
                        Earlystop=False,
                        augmentation = True,
                        flatten = 'global_average_pooling',
                        trainable_layers = False,
                        regulariser=None,
                        train_log=False,
                        load_weights = False,
                        save_weights = '/home/jj/FYP/Checkpoint/Placeholder/bestmodel',
                        learning_rate=1e-3,
                        init_lr=False,
                        min_lr = 0,
                        warmup=0,
                        cosine=0,
                        optimiser=tf.keras.optimizers.Adam(),
                        losses = tf.keras.losses.CategoricalCrossentropy(),class_weight=None,
                        metrics = [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.Recall(class_id=0),tf.keras.metrics.Recall(class_id=1)],
                        callbacks = [],
                        img_height=300,img_width=300,
                        plot=True):
            '''
            Insert dataset as a list, ensure dataset has same length as the number of folds [data1,data2....]
            Trainable layer is only set for second
            '''
            if k_fold != len(dataset):
                print(f'Number of datasets: {len(dataset)} not equal to {k_fold}')
                return 
            


            model, history = self.main(pretrained_model,)
            # metrics={}


            # return metrics


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



