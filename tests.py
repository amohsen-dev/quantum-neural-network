import data_processing
from cirq import Simulator
from cirq.contrib.svg import SVGCircuit
import cirq
import numpy as np
from config import N_QUBITS
from gates import controlled_x


def testing_image_to_circuit():
    circuit, qubits = data_processing.circuit_from_image(np.zeros(2**N_QUBITS) + 1)
    print(circuit)
    simulator = Simulator()
    result=simulator.simulate(circuit, qubit_order=qubits, initial_state=0)
    out_state = result.final_state_vector
    out_state_rep = np.abs(np.around(out_state.astype(np.float) * np.power(2,N_QUBITS/2),3))
    print(out_state_rep)
    for i in range(0, len(out_state_rep), 2):
        print('{} x ({},{})'.format(format(i//2, '0{}b'.format(N_QUBITS)), int(out_state_rep[i]), int(out_state_rep[i + 1])))
    SVGCircuit(circuit)


def testing_controlledX(n_q):
    qubits = cirq.GridQubit.rect(n_q-1,1)
    qubits.append(cirq.GridQubit(n_q,1))

    circuit = cirq.Circuit()
    circuit.append([cirq.X(q) for q in qubits[:-1]])
    circuit.append(cirq.X.controlled(n_q-1)(*qubits))
    print(circuit)
    simulator = Simulator()
    result=simulator.simulate(circuit, qubit_order=qubits, initial_state=0)
    out_state = result.final_state_vector
    out_state_rep = np.abs(np.around(out_state.astype(np.float) * np.power(2,(n_q)/2),3))
    print(out_state_rep)
    for i in range(0, len(out_state_rep), 2):
        print('{} x ({},{})'.format(format(i//2, '0{}b'.format(n_q-1)), int(out_state_rep[i]), int(out_state_rep[i + 1])))

    circuit2 = cirq.Circuit()
    circuit2.append([cirq.X(q) for q in qubits[:-1]])
    circuit2.append(controlled_x(qubits))
    print(circuit2)
    simulator = Simulator()
    result=simulator.simulate(circuit2, qubit_order=qubits, initial_state=0)
    out_state = result.final_state_vector
    out_state_rep = np.abs(np.around(out_state.astype(np.float) * np.power(2,(n_q)/2),3))
    print(out_state_rep)
    for i in range(0, len(out_state_rep), 2):
        print('{} x ({},{})'.format(format(i//2, '0{}b'.format(n_q-1)), int(out_state_rep[i]), int(out_state_rep[i + 1])))


if __name__=='__main__':
    testing_controlledX(5)