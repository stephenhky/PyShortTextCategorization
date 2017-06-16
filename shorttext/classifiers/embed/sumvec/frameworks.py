from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2

from shorttext.utils.classification_exceptions import UnequalArrayLengthsException


def DenseWordEmbed(nb_labels,
                   dense_nb_nodes=[],
                   dense_actfcn=[],
                   vecsize=300,
                   reg_coef=0.1,
                   final_activiation='softmax',
                   optimizer='adam'):
    """ Return layers of dense neural network.

    Return layers of dense neural network. This assumes the input to be a rank-1 vector.

    :param nb_labels: number of class labels
    :param dense_nb_nodes: number of nodes in each later (Default: [])
    :param dense_actfcn: activation functions for each layer (Default: [])
    :param vecsize: length of the embedded vectors in the model (Default: 300)
    :param reg_coef: regularization coefficient (Default: 0.1)
    :param final_activiation: activation function of the final layer (Default: softmax)
    :param optimizer: optimizer for gradient descent. Options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam. (Default: adam)
    :return: keras sequential model for dense neural network
    :type nb_labels: int
    :type dense_nb_nodes: list
    :type dense_actfcn: list
    :type vecsize: int
    :type reg_coef: float
    :type final_activiation: str
    :type optimizer: str
    :rtype: keras.models.Sequential
    """
    if len(dense_nb_nodes)!=len(dense_actfcn):
        raise UnequalArrayLengthsException(dense_nb_nodes, dense_actfcn)
    nb_layers = len(dense_nb_nodes)

    model = Sequential()
    if nb_layers==0:
        model.add(Dense(nb_labels,
                        input_shape=(vecsize,),
                        activation=final_activiation,
                        kernel_regularizer=l2(reg_coef)))
    else:
        model.add(Dense(dense_nb_nodes[0],
                        input_shape=(vecsize,),
                        activation=dense_actfcn[0],
                        kernel_regularizer=l2(reg_coef))
                  )
        for nb_nodes, activation in zip(dense_nb_nodes[1:], dense_actfcn[1:]):
            model.add(Dense(nb_nodes,
                            activation=activation,
                            kernel_regularizer=l2(reg_coef))
                      )
        model.add(Dense(nb_labels,
                        activation=final_activiation,
                        kernel_regularizer=l2(reg_coef))
                  )
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model