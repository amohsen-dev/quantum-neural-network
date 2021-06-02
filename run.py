import num2words.lang_EN
import data_processing
from cirq import Simulator
import numpy as np
from cirq.contrib.svg import SVGCircuit
import tensorflow as tf
import tensorflow_quantum as tfq
from tqdm import tqdm
import cirq
import sympy
import numpy as np
import collections
from gates import CircuitLayerBuilder
from config import N_QUBITS, N_LAYERS
from data_processing import  get_quantum_tensors
import time
import sys
tf.random.set_seed(229)


def create_model():
    pixel_qubits = cirq.GridQubit.rect(N_QUBITS, 1)
    color_qubit = cirq.GridQubit(-1, -1)
    output_qubit = cirq.GridQubit(-2, -2)
    circuit = cirq.Circuit()

    # Prepare the readout qubit.
    circuit.append(cirq.X(output_qubit))
    circuit.append(cirq.H(output_qubit))

    builder = CircuitLayerBuilder(pixel_qubits, color_qubit, output_qubit)
    for layer in range(N_LAYERS//2):
        builder.add_layer(circuit, cirq.XX, 'xx{}'.format(num2words.num2words(layer)))
        builder.add_layer(circuit, cirq.ZZ, 'zz{}'.format(num2words.num2words(layer)))

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(output_qubit))

    return circuit, cirq.Z(output_qubit)


if __name__ == '__main__':
    step = int(sys.argv[1])
    model_circuit, model_readout = create_model()

    model = tf.keras.Sequential([
        # The input is the data-circuit, encoded as a tf.string
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        # The PQC layer returns the expected value of the readout gate, range [-1,1].
        tfq.layers.PQC(model_circuit, model_readout),
    ])

    x_train, y_train, x_test, y_test = get_quantum_tensors(subset=None, load_tensors=True, save_tensors=False, step=step)
    y_train_hinge = 2.0 * y_train - 1.0
    y_test_hinge = 2.0 * y_test - 1.0

    def hinge_accuracy(y_true, y_pred):
        y_true = tf.squeeze(y_true) > 0.0
        y_pred = tf.squeeze(y_pred) > 0.0
        result = tf.cast(y_true == y_pred, tf.float32)

        return tf.reduce_mean(result)

    model.compile(
        loss=tf.keras.losses.Hinge(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[hinge_accuracy])

    print(model.summary())

    EPOCHS = 10
    BATCH_SIZE = 32
    t1 = time.time()
    qnn_history = model.fit(
        x_train, y_train_hinge,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(x_test, y_test_hinge))
    print('time to fit model : {}'.format(time.time() - t1))
    qnn_results = model.evaluate(x_test, y_test)

