import numpy as np

from neural_net import *

def feed_forward_nn(idx_user, normalized_featX_train, dataY_train, single_set_aux, layers_dims, reg, sub_data_train):

    # if idx_user == 0:
    #     clf = {}
    # #end

    # clf[idx_user]         = {}
    # clf[idx_user]['Name'] = normalized_featX_train[idx_user]['Name']
    # clf[idx_user]['Net']  = []
    # clf[idx_user]['Fit']  = []

    sub_data_train['clf'][idx_user]         = {}
    sub_data_train['clf'][idx_user]['Name'] = normalized_featX_train[idx_user]['Name']
    sub_data_train['clf'][idx_user]['Net']  = []
    sub_data_train['clf'][idx_user]['Fit']  = []

    X_train = np.transpose(normalized_featX_train[idx_user]['Feature'].values)
    y_train = dataY_train[idx_user]['Actual Label'].values.ravel().reshape((1,np.shape(single_set_aux)[0]))
    y_aux = np.zeros(X_train.shape)
    for i in range(np.shape(y_train)[1]):
        y_aux[int(y_train[0,i])-1,i] = 1
    #end
    y_train = y_aux

    # NN with tanh activation fn
    parameters_tanh = L_layer_model(
        X_train, y_train, layers_dims, learning_rate=0.03, num_iterations=50000,
        hidden_layers_activation_fn="tanh", lambda_reg=reg)
        
    sub_data_train['clf'][idx_user]['Fit'] = parameters_tanh

    # return clf
    return sub_data_train['clf']