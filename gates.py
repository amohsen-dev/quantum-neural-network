import cirq
import sympy
import tensorflow as tf
import num2words.lang_EN


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
        yield cirq.CNOT(*qubits) ** exponent
        return
    yield cirq.CNOT(qubits[-2], qubits[-1]) ** (exponent/2.)
    yield from controlled_x(qubits[:-1], 1.)
    yield cirq.CNOT(qubits[-2], qubits[-1]) ** (-exponent/2.)
    yield from controlled_x(qubits[:-1], 1.)
    yield from controlled_x(qubits[:-2] + [qubits[-1]], exponent/2.)


def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)


def create_model(n_qubits, n_layers):
    pixel_qubits = cirq.GridQubit.rect(n_qubits, 1)
    color_qubit = cirq.GridQubit(-1, -1)
    output_qubit = cirq.GridQubit(-2, -2)
    circuit = cirq.Circuit()

    circuit.append(cirq.X(output_qubit))
    circuit.append(cirq.H(output_qubit))

    builder = CircuitLayerBuilder(pixel_qubits, color_qubit, output_qubit)
    for layer in range(n_layers // 2):
        builder.add_layer(circuit, cirq.XX, 'xx{}'.format(num2words.num2words(layer)))
        builder.add_layer(circuit, cirq.ZZ, 'zz{}'.format(num2words.num2words(layer)))

    circuit.append(cirq.H(output_qubit))

    return circuit, cirq.Z(output_qubit)
