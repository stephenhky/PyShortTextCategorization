
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
    """Create a dense neural network for embedding-based classification.

    Args:
        nb_labels: Number of class labels.
        dense_nb_nodes: Nodes per layer. Default: [].
        dense_actfcn: Activation functions per layer. Default: [].
        vecsize: Embedding vector size. Default: 300.
        reg_coef: L2 regularization coefficient. Default: 0.1.
        final_activiation: Final layer activation. Default: softmax.
        optimizer: Optimizer. Default: adam.

    Returns:
        Keras Sequential model.

    Raises:
        UnequalArrayLengthsException: If dense_nb_nodes and dense_actfcn have different lengths.
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

    model.add(Activation(final_activiation))
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)

    return model
