import numpy as np
import cirq
import itertools
import sympy

class MultiPauliPowerGate(cirq.Gate):
    def __init__(self, paulis, exponent = 1.):
        super(MultiPauliPowerGate, self)
        self._n = len(paulis)
        self._paulis = paulis
        self._matrix = cirq.unitary(self._paulis[0])
        self._name = self._paulis[0]._name
        self._exponent = exponent
        for pauli in paulis[1:]:
            self._matrix = np.kron(self._matrix, cirq.unitary(pauli))
            self._name += pauli._name
        if self._exponent != 1.:
            self._matrix = cirq.linalg.map_eigenvalues(self._matrix, lambda b: b**self._exponent)

    def _num_qubits_(self):
        return self._n

    def _unitary_(self):
        return self._matrix

    def _circuit_diagram_info_(self, args):
        return cirq.protocols.CircuitDiagramInfo(wire_symbols=[c for c in self._name],
                                                 exponent=self._exponent,
                                                 exponent_qubit_index=len(self._name)-2)


class CircuitLayerBuilder2:
    def __init__(self, pixel_qubits, color_qubit, output_qubit):
        self._pixel_qubits = pixel_qubits
        self._color_qubit = color_qubit
        self._output_qubit = output_qubit

    def add_3q_layer(self, circuit, gate_name):
        for n, qubit in enumerate(self._pixel_qubits):
            symbol = sympy.Symbol(gate_name.lower() + '_' + str(n))
            gate = MultiPauliPowerGate(paulis=[getattr(cirq, p) for p in gate_name], exponent=symbol)
            circuit.append(gate(qubit, self._color_qubit, self._output_qubit))


class CircuitLayerBuilder:
    def __init__(self, pixel_qubits, color_qubit, output_qubit):
        self.pixel_qubits = pixel_qubits
        self.color_qubit = color_qubit
        self.output_qubit = output_qubit

    def add_layer(self, circuit, gate, prefix):
        symbols = sympy.symbols(prefix+'0:{}'.format(len(self.pixel_qubits)))
        for n, qubit in enumerate(self.pixel_qubits):
            circuit.append(gate(qubit, self.output_qubit) ** symbols[n])
            circuit.append(gate(self.color_qubit, self.output_qubit) ** symbols[n])

def controlled_x(qubits, exponent=1.):
    if len(qubits) == 2:
        yield (cirq.CNOT(*qubits) ** exponent)
        return
    yield (cirq.CNOT(qubits[-2], qubits[-1]) ** (exponent/2.))
    yield from controlled_x(qubits[:-1], 1.)
    yield (cirq.CNOT(qubits[-2], qubits[-1]) ** (-exponent/2.))
    yield from controlled_x(qubits[:-1], 1.)
    yield from (controlled_x(qubits[:-2] + [qubits[-1]], exponent/2.))
