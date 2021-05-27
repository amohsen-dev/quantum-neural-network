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
from config import N_QUBITS
from gates import controlled_x
import multiprocessing as mp
import time
gate = cirq.X.controlled(N_QUBITS)


def f(x):
    return tfq.convert_to_tensor([x]).numpy()

def generate_circuit_from_image_recursive(qubits, image):
    if len(image) == 1:
        (yield controlled_x(qubits)) if image[0] == 1 else None
        return
    yield from generate_circuit_from_image_recursive(qubits, image[:len(image)//2])
    for n in range(int(np.log2(len(image)))):
        yield cirq.X(qubits[-2-n])
    yield from generate_circuit_from_image_recursive(qubits, image[len(image)//2:])

def circuit_from_image(image):
    qubits = cirq.GridQubit.rect(N_QUBITS, 1)
    qubits.append(cirq.GridQubit(-1, -1))
    circuit = cirq.Circuit()
    circuit.append((cirq.H(qubit) for qubit in qubits[:-1]))
    circuit.append(generate_circuit_from_image_recursive(qubits, image))
    return circuit#, qubits


def remove_contradicting(xs, ys):
    mapping = collections.defaultdict(set)
    orig_x = {}
    # Determine the set of labels for each unique image:
    for x, y in zip(xs, ys):
        orig_x[tuple(x.flatten())] = x
        mapping[tuple(x.flatten())].add(y)
    new_x = []
    new_y = []
    for flatten_x in mapping:
        x = orig_x[flatten_x]
        labels = mapping[flatten_x]
        if len(labels) == 1:
            new_x.append(x)
            new_y.append(next(iter(labels)))
        else:
            # Throw out images that match more than one label.
            pass
    return np.array(new_x), np.array(new_y)

def get_images(filter_digits=True, single_label=True, digits=(3,6), black_and_white=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    def keep_two(x, y):
        keep = (y == digits[0]) | (y == digits[1])
        x, y = x[keep], y[keep]
        y = y == digits[0]
        return x, y

    if filter_digits:
        x_train, y_train = keep_two(x_train, y_train)
        x_test, y_test = keep_two(x_test, y_test)

    if single_label:
        x_train, y_train = remove_contradicting(x_train, y_train)

    image_width = int(np.power(2, N_QUBITS//2))
    x_train = tf.image.resize(x_train, (image_width, image_width)).numpy()
    x_test = tf.image.resize(x_test, (image_width, image_width)).numpy()

    if black_and_white:
        x_train = np.array(x_train > .5, dtype=np.float32)
        x_test = np.array(x_test > .5, dtype=np.float32)

    return x_train, y_train, x_test, y_test


def get_quantum_data(subset=None, load_tensors=False, step=0):
    x_train_classical, y_train, x_test_classical, y_test = get_images(filter_digits=True,
                                                                      single_label=True,
                                                                      digits=(3,6),
                                                                      black_and_white=True)
    if subset is not None:
        startint_point = 3000 + step * 500
        x_train_classical = x_train_classical[startint_point:(startint_point+subset)]
        x_test_classical = x_test_classical[:10]
        y_train = y_train[startint_point:(startint_point+subset)]
        y_test = y_test[:10]

    if load_tensors:
        x_train_quantum, x_test_quantum = None, None
    else:
        print('----- generating training quantum circuits ------')
        # x_train_quantum = []
        # for image in tqdm(x_train_classical):
        #     circuit, qubits = circuit_from_image(image.flatten())
        #     x_train_quantum.append(circuit)
        x_train_quantum = []
        x_train_classical_batches = np.array_split(x_train_classical, 100)
        with mp.Pool(mp.cpu_count()) as pool:
            for batch in tqdm(x_train_classical_batches):
                quantum_batch = pool.map(circuit_from_image, [image.flatten() for image in batch])
                x_train_quantum.extend(quantum_batch)

        print('----- generating test quantum circuits ------')
        # x_test_quantum = []
        # for image in tqdm(x_test_classical):
        #     x_test_quantum.append(circuit_from_image(image.flatten())[0])
        x_test_quantum = []
        x_test_classical_batches = np.array_split(x_test_classical, 100)
        with mp.Pool(mp.cpu_count()) as pool:
            for batch in tqdm(x_test_classical_batches):
                quantm_batch = pool.map(circuit_from_image, [image.flatten() for image in batch])
                x_test_quantum.extend(quantm_batch)

    return x_train_quantum, y_train, x_test_quantum, y_test


def get_quantum_tensors(subset=None, load_tensors=False, save_tensors=False, step=0):
    if load_tensors:
        _, y_train, _, y_test = get_quantum_data(subset, load_tensors, step)
        # x_train_quantum_tensor = tf.io.parse_tensor(tf.io.read_file('data/serialized_train_{}QUBITS'.format(N_QUBITS)), tf.string)
        # x_test_quantum_tensor = tf.io.parse_tensor(tf.io.read_file('data/serialized_test_{}QUBITS'.format(N_QUBITS)), tf.string)
        x_train_quantum_tensor = tf.io.parse_tensor(tf.io.read_file('data/serialized_train_{}QUBITS_{}'.format(N_QUBITS, 1)), tf.string)
        for i in range(2,19):
            tensor = tf.io.parse_tensor(tf.io.read_file('data/serialized_train_{}QUBITS_{}'.format(N_QUBITS, i)), tf.string)
            x_train_quantum_tensor = tf.convert_to_tensor(np.concatenate([x_train_quantum_tensor.numpy(), tensor.numpy()]))
        x_test_quantum_tensor = tf.io.parse_tensor(tf.io.read_file('data/serialized_test_{}QUBITS'.format(N_QUBITS)), tf.string)
    else:
        t1 = time.time()
        x_train_quantum, y_train, x_test_quantum, y_test = get_quantum_data(subset, load_tensors, step)
        print('time to generate circuits : {}'.format(time.time()-t1))
        print('\n----- converting train circuits to tensors ------')
        t1 = time.time()
        #x_train_quantum_tensor = tfq.convert_to_tensor(x_train_quantum)
        #ll = [tfq.convert_to_tensor([x]).numpy() for x in x_train_quantum]
        ll = []
        x_train_quantum_batches = np.array_split(np.array(x_train_quantum, dtype=object), 100)
        with mp.Pool(mp.cpu_count()) as pool:
            for batch in tqdm(x_train_quantum_batches):
                ll.extend(pool.map(f, batch))
        x_train_quantum_tensor = tf.convert_to_tensor(np.concatenate(ll))
        if save_tensors:
            tf.io.write_file('serialized_train_{}QUBITS_{}'.format(N_QUBITS, step), tf.io.serialize_tensor(x_train_quantum_tensor))
        print('\n----- converting test circuits to tensors ------')
        #x_test_quantum_tensor = tfq.convert_to_tensor(x_test_quantum)
        #ll = [tfq.convert_to_tensor([x]).numpy() for x in x_test_quantum]
        ll = []
        x_test_quantum_batches = np.array_split(np.array(x_test_quantum, dtype=object), 100)
        with mp.Pool(mp.cpu_count()) as pool:
            for batch in tqdm(x_test_quantum_batches):
                ll.extend(pool.map(f, batch))
        x_test_quantum_tensor = tf.convert_to_tensor(np.concatenate(ll))
        if save_tensors:
            pass
            #tf.io.write_file('serialized_test_{}QUBITS'.format(N_QUBITS), tf.io.serialize_tensor(x_test_quantum_tensor))
        print('time to generate tensors: {}'.format(time.time()-t1))
    return x_train_quantum_tensor, y_train, x_test_quantum_tensor, y_test

