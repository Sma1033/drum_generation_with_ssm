import tensorflow as tf


# define function to apply cqt ratio
def tf_broadcast(tensor, shape):    
    return (tensor + tf.zeros(shape, dtype=tensor.dtype))



# use CQT ration to give bar data different weight
def apply_cqt_ratio(cqt_tensor_in, ratio_tensor_in):
    
    ratio_tensor_in_4d = tf.reshape(ratio_tensor_in, tf.shape(cqt_tensor_in[:,0:1,0:1,:]))
    
    ratio_tensor_in_expanded = tf_broadcast(ratio_tensor_in_4d, tf.shape(cqt_tensor_in[:,:,:,:]))
    
    return (tf.multiply(cqt_tensor_in, ratio_tensor_in_expanded))



#define function to calculate diff y layer
# note: input tensor must be 4-D data
def get_matx_2_layer_tf(matx_data_in):
    
    matx_layer_0 = matx_data_in
    
    matx_data_pady = tf.pad(matx_data_in,
                            paddings=[[0,0], [0,0], [1,0], [0,0]],
                            mode='CONSTANT',
                            name='tf_diff2_pady',
                            constant_values=0
                            )[:, :, :-1, :]
                            #)[:, :matx_data_in.get_shape()[1], :, :]
        
    matx_layer_1 = matx_data_in - matx_data_pady
    
    matx_layer_1_concat = tf.concat([tf.zeros_like(matx_layer_1)[:,:,0:1,:], 
                                     matx_layer_1[:,:,1:,:]],
                                    axis=2)
    
    
    matx_layer_all = tf.concat([matx_layer_0, matx_layer_1_concat], 
                               axis=-1,
                               name='tf_diff2_concat')
    
    return matx_layer_all    
    
#print ('Diff-2 function define done.')



# define function to calculate 3 layer (add diif x/y layer)
# note: input tensor must be 4-D data
def get_matx_3_layer_tf(matx_data_in):
    
    matx_layer_0 = matx_data_in
    
    matx_data_pady = tf.pad(matx_data_in,
                            paddings=[[0,0], [0,0], [1,0], [0,0]],
                            mode='CONSTANT',
                            name='tf_diff3_pady',
                            constant_values=0
                            )[:, :, :-1, :]
                            #)[:, :matx_data_in.get_shape()[1], :, :]
        
    matx_layer_1 = matx_data_in - matx_data_pady
    
    matx_layer_1_concat = tf.concat([tf.zeros_like(matx_layer_1)[:,:,0:1,:], 
                                     matx_layer_1[:,:,1:,:]],
                                    axis=2)
    
    matx_data_padx = tf.pad(matx_data_in,
                            paddings=[[0,0], [1,0], [0,0], [0,0]],
                            mode='CONSTANT',
                            name='tf_diff3_padx',
                            constant_values=0
                            )[:, :-1, :, :]
                            #)[:, :, :matx_data_in.get_shape()[2], :]    
    
    matx_layer_2 = matx_data_in - matx_data_padx
    
    matx_layer_2_concat = tf.concat([tf.zeros_like(matx_layer_1)[:,0:1,:,:], 
                                     matx_layer_1[:,1:,:,:]],
                                    axis=1)
    
    matx_layer_all = tf.concat([matx_layer_0, matx_layer_1_concat, matx_layer_2_concat], 
                               axis=-1,
                               name='tf_diff3_concat')
    
    return matx_layer_all    
    
#print ('Diff-3 function define done.')



