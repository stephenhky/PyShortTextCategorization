
from typing import Optional, Literal

from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2

from ....utils.classification_exceptions import UnequalArrayLengthsException


def DenseWordEmbed(
        nb_labels: int,
        dense_nb_nodes: Optional[list[int]] = None,
        dense_actfcn: Optional[Literal["softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"]] = None,
        vecsize: int = 300,
        reg_coef: float = 0.1,
        final_activiation: Literal["softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"] = "softmax",
        optimizer: Literal["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"] = "adam"
) -> Model:
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
    :rtype: keras.models.Model
    """
    if dense_nb_nodes is None:
        dense_nb_nodes = []
    if dense_actfcn is None:
        dense_actfcn = []

    if len(dense_nb_nodes)!=len(dense_actfcn):
        raise UnequalArrayLengthsException(dense_nb_nodes, dense_actfcn)
    nb_layers = len(dense_nb_nodes)

    model = Sequential()
    if nb_layers==0:
        model.add(Dense(nb_labels, input_shape=(vecsize,), kernel_regularizer=l2(reg_coef)))
    else:
        model.add(Dense(dense_nb_nodes[0],
                        input_shape=(vecsize,),
                        activation=dense_actfcn[0],
                        kernel_regularizer=l2(reg_coef))
                  )
        for nb_nodes, activation in zip(dense_nb_nodes[1:], dense_actfcn[1:]):
            model.add(Dense(nb_nodes, activation=activation, kernel_regularizer=l2(reg_coef)))
        model.add(Dense(nb_labels, kernel_regularizer=l2(reg_coef)))

    # final activation layer
    model.add(Activation(final_activiation))
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)

    return model