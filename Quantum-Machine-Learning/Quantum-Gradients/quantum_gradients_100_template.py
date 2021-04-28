#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


def parameter_shift(weights):
    """Compute the gradient of the variational circuit given by the
    ansatz function using the parameter-shift rule.

    Write your code below between the # QHACK # markers—create a device with
    the correct number of qubits, create a QNode that applies the above ansatz,
    and compute the gradient of the provided ansatz using the parameter-shift rule.

    Args:
        weights (array): An array of floating-point numbers with size (2, 3).

    Returns:
        array: The gradient of the variational circuit. The shape should match
        the input weights array.
    """
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def circuit(weights):
        for i in range(len(weights)):
            qml.RX(weights[i, 0], wires=0)
            qml.RY(weights[i, 1], wires=1)
            qml.RZ(weights[i, 2], wires=2)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 0])


        return qml.expval(qml.PauliY(0) @ qml.PauliZ(2))

    gradient = np.zeros(np.shape(weights))

    # QHACK #

    def parameter_shift_term(qnode, params,i,j):
        dx = 0.01
        
        shifted_forw = params.copy()
        shifted_forw[i][j] += dx
        forward = qnode(shifted_forw)

        shifted_back = params.copy()
        shifted_back[i][j] -= dx
        backward = qnode(shifted_back)

        return 0.5 * (forward - backward) /np.sin(dx)

    
    def parameter_shift(qnode, params, gradient):
        (row,column) = np.shape(weights)

        for i in range(row):
            for j in range(column):
                gradient[i][j] = parameter_shift_term(qnode, weights, i, j)

        return gradient
    
    gradient = parameter_shift(circuit, weights, gradient)

    

    
    # QHACK #
    


    return gradient


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = np.array([row.split(",") for row in weights.split("S") if row], dtype=np.float64)

    gradient = np.round(parameter_shift(weights), 10)

    output_array = gradient.flatten()
    print(",".join([str(val) for val in output_array]))
