import tensorflow as tf
import numpy as np
import pandas as pd
import math
import time

CLASS_NUM = 17
BATCH_SIZE = 200
EARLY_STOP_THRE = 900


def __cnn_layer(layer, filters, is_training, is_pool, cnn_name, bn_name):
    # regularizer = l1_l2_regularizer(scale_l1=0.1, scale_l2=0.1)
    new_layer = tf.layers.conv2d(
                inputs=layer,  # N * 32 * 32 * Channels; (60000, 28, 28)
                filters=filters,
                # kernel_regularizer=regularizer, 
                kernel_size=[3, 3],
                padding="same",
                activation=None, 
                name = cnn_name)
    new_layer = tf.layers.batch_normalization(new_layer, training=is_training, name=bn_name)
    new_layer = tf.nn.relu(new_layer)
    if is_pool:
        new_layer = tf.layers.max_pooling2d(inputs=new_layer, pool_size=[2, 2], strides=2)
    return new_layer


def build_model(model_name):
    '''
    Define the names of tensor one by one, and get the parameters by name later. 
    '''

    print('Model Build Start!')
    
    tf.reset_default_graph()
    
    with tf.name_scope(model_name):
        # inputs
        y_place = tf.placeholder('float',[None,17], name='y_palce')
        X_place = tf.placeholder('float',[None,32,32,10], name='x_place')
        is_training = tf.placeholder('bool', name='is_trainning_place')
        # X_proba = tf.placeholder('float',[None,17], name='x_proba_palce')
        
        #OPs
        # input BN
        input_img = tf.layers.batch_normalization(X_place, training=is_training, name='input_BN')
        
        # CNN layers : 2C * MP * 2C * MP * 2C * MP
        layer_1_0 = __cnn_layer(input_img, 32, is_training, False, cnn_name='CNN1', bn_name='CNN1_BN')
        layer_1_1 = __cnn_layer(layer_1_0, 32, is_training, True, cnn_name='CNN2', bn_name='CNN2_BN')  # 16 * 16
        layer_1 = tf.layers.dropout(layer_1_1, rate=0.25, training=is_training)
        
        layer_2_0 = __cnn_layer(layer_1, 64, is_training, False, cnn_name='CNN3', bn_name='CNN3_BN')
        layer_2_1 = __cnn_layer(layer_2_0, 64, is_training, True, cnn_name='CNN4', bn_name='CNN4_BN')  # 8 * 8
        layer_2 = tf.layers.dropout(layer_2_1, rate=0.25, training=is_training)
        
        layer_3_0 = __cnn_layer(layer_2, 128, is_training, False, cnn_name='CNN5', bn_name='CNN5_BN')
        layer_3_1 = __cnn_layer(layer_3_0, 128, is_training, True, cnn_name='CNN6', bn_name='CNN6_BN')  # 4 * 4
        layer_3 = tf.layers.dropout(layer_3_1, rate=0.25, training=is_training)
        
        # Dense layers : 1D(128) * 1D(17)
        conv_reshape = tf.reshape(layer_3, [-1,4*4*128])
        
        dense = tf.layers.dense(inputs=conv_reshape, units=128, activation=None)
        dense = tf.layers.batch_normalization(dense, training=is_training)
        dense = tf.nn.sigmoid(dense)  # sigmoid is better but slow...
        dense = tf.layers.dropout(dense, rate=0.25, training=is_training)
        # dense_concat = tf.concat([dense, X_proba], 1)  # add proba in Train Set. 
        
        logits = tf.layers.dense(inputs=dense, units=CLASS_NUM)
        
        # Output
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_place, logits=logits), name='loss')
        predict_proba = tf.nn.softmax(logits, name='proba')
        y_predict_tf = tf.argmax(predict_proba, 1, name='pre_label')
        ac = tf.reduce_mean(tf.to_float(tf.equal(y_predict_tf, tf.argmax(y_place, 1))), name='ac')
        
    print('Model Build Success!')


def training(train_set, valid_set, model_name="Model", epochs=10, learning_rate=0.001, restore_name=None):
    # Get data. 
    X_train, y_train = train_set
    is_training = True

    # Get Tensors
    graph = tf.get_default_graph()

    y_place = graph.get_tensor_by_name("{}/y_palce:0".format(model_name))
    X_place = graph.get_tensor_by_name("{}/x_place:0".format(model_name))
    is_training_place = graph.get_tensor_by_name("{}/is_trainning_place:0".format(model_name))

    loss = graph.get_tensor_by_name("{}/loss:0".format(model_name))
    predict_proba = graph.get_tensor_by_name("{}/proba:0".format(model_name))
    y_predict_tf = graph.get_tensor_by_name("{}/pre_label:0".format(model_name))
    ac = graph.get_tensor_by_name("{}/ac:0".format(model_name))

    # Set Variables
    max_ac = -1
    min_loss = np.inf
    early_stop = 0
    train_num = y_train.shape[0]
    max_iteration = int(epochs * train_num / BATCH_SIZE)

    # Trainning Op Set
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)  # var_list=training_variables

    # Sess Start! 
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # is Restore ?
        saver = tf.train.Saver(max_to_keep=1) # saver = tf.train.Saver(saver_dict, max_to_keep=1)
        if restore_name is not None:
            saver.restore(sess, "ckpt\{}.ckpt".format(RESTORE_MODEL_NAME))

        start_time = time.time()
        print('start training!')

        for i_iteration in range(max_iteration):
            batch_samples_loc = np.sort(np.random.choice(train_num, BATCH_SIZE, replace=False))
            X_train_batch = X_train[batch_samples_loc, :, :, :]
            y_train_batch = y_train[batch_samples_loc, :]

            sess.run(train_step, feed_dict={X_place: X_train_batch, 
                                            y_place: y_train_batch, 
                                            is_training_place: True})

            if i_iteration % 50 == 0: # check at 50 it
                # early_stop_code
                if early_stop > EARLY_STOP_THRE:
                    break
                _ac, _loss = batch_vali(valid_set, sess, model_name)
                early_stop += 50
                if _loss < min_loss:
                    early_stop = 0
                    min_loss = _loss
                    print("Loss change at: ", i_iteration, "With AC: ", _ac, "With Loss: ", _loss)

        model_saver_path = '../ckpt/{}.ckpt'.format(model_name)
        saver = tf.train.Saver() # redefine a saver. 
        saver.save(sess, model_saver_path)
        print('Time use: ', time.time() - start_time)


def batch_vali(data, sess, model_name):
    # Get Tensors
    graph = tf.get_default_graph()

    y_place = graph.get_tensor_by_name("{}/y_palce:0".format(model_name))
    X_place = graph.get_tensor_by_name("{}/x_place:0".format(model_name))
    is_training_place = graph.get_tensor_by_name("{}/is_trainning_place:0".format(model_name))

    loss = graph.get_tensor_by_name("{}/loss:0".format(model_name))
    ac = graph.get_tensor_by_name("{}/ac:0".format(model_name))
    
    # Get data
    X_valid, y_valid = data

    # define parameters
    total_ac = 0
    total_loss = 0
    
    samples = y_valid.shape[0]
    batch_iteration = math.ceil(samples / BATCH_SIZE)
    for i in range(batch_iteration):
        x_range = i*BATCH_SIZE
        y_range = np.min(((i+1)*BATCH_SIZE, samples))
        batch_ac, batch_loss = sess.run([ac, loss], 
                                        feed_dict={X_place: X_valid[x_range:y_range, :, :, :], 
                                                   y_place: y_valid[x_range:y_range, :],
                                                   is_training_place: False})
        total_ac += batch_ac
        total_loss += batch_loss
    return total_ac/batch_iteration, total_loss/samples


def batch_predict(data, model_name):
    # Get Tensors
    graph = tf.get_default_graph()

    X_place = graph.get_tensor_by_name("{}/x_place:0".format(model_name))
    is_training_place = graph.get_tensor_by_name("{}/is_trainning_place:0".format(model_name))

    predict_proba = graph.get_tensor_by_name("{}/proba:0".format(model_name))
    y_predict_tf = graph.get_tensor_by_name("{}/pre_label:0".format(model_name))
    
    # Get data
    X_test = data

    # define parameters
    total_ac = 0
    total_loss = 0
    
    number_test = X_test.shape[0]
    batch_iteration = math.ceil(number_test / BATCH_SIZE)
    result = np.zeros((number_test, CLASS_NUM))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, "../ckpt/{}.ckpt".format(model_name))
        for i in range(batch_iteration):
            x_range = i*BATCH_SIZE
            y_range = np.min(((i+1)*BATCH_SIZE, number_test))
            _y_ = sess.run(predict_proba, feed_dict={X_place: X_test[x_range:y_range, :, :, :], is_training_place: False})
            result[x_range:y_range, :] = _y_
    return result
