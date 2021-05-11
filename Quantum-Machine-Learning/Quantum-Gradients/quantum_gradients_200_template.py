#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #


    def hessian_part_equal(hessian,gradient,params, i, j, dx,cl):
        if cl==0:
            cl=cl+1
            shifted = weights.copy()
            classic = circuit(shifted)
        shifted = weights.copy()
        shifted[i] += dx 
        Shifted_Forw = circuit(shifted)
                
        shifted[i] -= 2*dx
        Shifted_Back = circuit(shifted)
                
        hessian[i][j] =  (Shifted_Forw - 2*classic + Shifted_Back)/(2)
        gradient[i] = 0.5 * (Shifted_Forw - Shifted_Back)/(np.sin(dx))

        return gradient, hessian

    def hessian_part_different(hessian, gradient, params, i, j, dx):
        shifted = weights.copy()
        shifted[i] += dx
        shifted[j] += dx
        circuit_1 = circuit(shifted)

        shifted[i] -= 2*dx
        circuit_2 = circuit(shifted)

        shifted[i] += 2*dx
        circuit_3 = circuit(shifted)

        shifted[j] -= 2*dx
        circuit_4 = circuit(shifted)
                
        hessian[i][j] =  (circuit_1 - circuit_2 - circuit_3 + circuit_4) / ((2*np.sin(dx))**2)
        hessian[j][i] = hessian[i][j]
                
        return hessian

    def parameter_shift(params, gradient, hessian,dx):
        n = len(params)
        cl=0
        for i in range(n):
            for j in range(i,n):
                if j<i:
                    hessian = hessian_part_different(hessian, gradient ,weights, i, j, dx)
                
                elif j==i:
                    gradient, hessian = hessian_part_equal(hessian, gradient, weights,i,j, dx, cl)

                

        return gradient, hessian
    
    gradient, hessian = parameter_shift(weights, gradient, hessian , np.pi/2)




    


    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
