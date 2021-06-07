import tensorflow as tf
import tensorflow_quantum as tfq
from tqdm import tqdm
import cirq
import numpy as np
import collections
from config import EXTRA_COMPRESSION, COMPRESSION_FACTOR
from gates import controlled_x
import multiprocessing as mp
import time
from functools import partial


def _convert_to_tensor(x):
    return tfq.convert_to_tensor([x]).numpy()


POW2 = np.array([.5**i for i in range(2 ** COMPRESSION_FACTOR)])


def generate_circuit_from_image_recursive(qubits, image):
    if EXTRA_COMPRESSION and len(image) == 2 ** COMPRESSION_FACTOR:
        (yield controlled_x(qubits, exponent=(POW2 * image).sum())) if image.sum() > 0 else None
        return
    if len(image) == 1:
        (yield controlled_x(qubits)) if image[0] == 1 else None
        return
    yield from generate_circuit_from_image_recursive(qubits, image[:len(image)//2])
    n_xs = int(np.log2(len(image)))
    if EXTRA_COMPRESSION:
        n_xs -= COMPRESSION_FACTOR
    for n in range(n_xs):
        yield cirq.X(qubits[-2-n])
    yield from generate_circuit_from_image_recursive(qubits, image[len(image)//2:])


def circuit_from_image(image, nqubits):
    qubits = cirq.GridQubit.rect(nqubits, 1)
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


def get_images(nqubits, filter_digits=True, single_label=True, digits=(3,6), black_and_white=True):
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

    image_width = int(np.power(2, nqubits//2))
    if EXTRA_COMPRESSION:
        image_width *= 2 ** (COMPRESSION_FACTOR // 2)
    x_train = tf.image.resize(x_train, (image_width, image_width)).numpy()
    x_test = tf.image.resize(x_test, (image_width, image_width)).numpy()

    if black_and_white:
        x_train = np.array(x_train > .5, dtype=np.float32)
        x_test = np.array(x_test > .5, dtype=np.float32)

    return x_train, y_train, x_test, y_test


def get_quantum_data(nqubits, subset=None, load_tensors=False, step=0, single_label=True, parallel=True):

    x_train_classical, y_train, x_test_classical, y_test = get_images(nqubits=nqubits,
                                                                      filter_digits=True,
                                                                      single_label=single_label,
                                                                      digits=(3,6),
                                                                      black_and_white=True)
    print('image dimensions: {}'.format(x_test_classical[0].shape))
    if subset is not None:
        starting_point = 0 + step * subset
        x_train_classical = x_train_classical[starting_point:(starting_point + subset)]
        x_test_classical = x_test_classical[starting_point:(starting_point + subset)]
        y_train = y_train[starting_point:(starting_point + subset)]
        y_test = y_test[starting_point:(starting_point + subset)]

    if load_tensors:
        x_train_quantum, x_test_quantum = None, None
    else:
        print('----- generating training quantum circuits ------')
        circuit_from_image_c = partial(circuit_from_image, nqubits=nqubits)
        x_train_quantum = []
        if parallel:
            x_train_classical_batches = np.array_split(x_train_classical, 100)
            with mp.Pool(mp.cpu_count()) as pool:
                for batch in tqdm(x_train_classical_batches):
                    quantum_batch = pool.map(circuit_from_image_c, [image.flatten() for image in batch])
                    x_train_quantum.extend(quantum_batch)
        else:
            for image in tqdm(x_train_classical):
                circuit = circuit_from_image(image.flatten(), nqubits=nqubits)
                x_train_quantum.append(circuit)

        print('----- generating test quantum circuits ------')
        x_test_quantum = []
        if parallel:
            x_test_classical_batches = np.array_split(x_test_classical, 100)
            with mp.Pool(mp.cpu_count()) as pool:
                for batch in tqdm(x_test_classical_batches):
                    quantm_batch = pool.map(circuit_from_image_c, [image.flatten() for image in batch])
                    x_test_quantum.extend(quantm_batch)
        else:
            for image in tqdm(x_test_classical):
                circuit = circuit_from_image(image.flatten(), nqubits=nqubits)
                x_test_quantum.append(circuit)

    return x_train_quantum, y_train, x_test_quantum, y_test


def load_quantum_tensors(nqubits, subset=None, load_tensors=False,
                         load_sequential_range=None,step=0, single_label=True):
    _, y_train, _, y_test = get_quantum_data(nqubits, subset, load_tensors, step, single_label)
    if load_sequential_range is None:
        x_train_quantum_tensor = tf.io.parse_tensor(tf.io.read_file(
            'data/serialized_train_{}QUBITS{}'.format(nqubits, '(compressed)' if EXTRA_COMPRESSION else '')),
                                                    tf.string)
        x_test_quantum_tensor = tf.io.parse_tensor(tf.io.read_file(
            'data/serialized_test_{}QUBITS{}'.format(nqubits, '(compressed)' if EXTRA_COMPRESSION else '')), tf.string)
    else:
        x_train_quantum_tensor = tf.io.parse_tensor(tf.io.read_file(
            'data/serialized_train_{}QUBITS{}_{}'.format(nqubits, '(compressed)' if EXTRA_COMPRESSION else '', 0)),
                                                    tf.string)
        for i in range(1, load_sequential_range):
            tensor = tf.io.parse_tensor(tf.io.read_file(
                'data/serialized_train_{}QUBITS{}_{}'.format(nqubits, '(compressed)' if EXTRA_COMPRESSION else '', i)),
                                        tf.string)
            x_train_quantum_tensor = tf.convert_to_tensor(
                np.concatenate([x_train_quantum_tensor.numpy(), tensor.numpy()]))
            print('loaded {}/{} training. Current length: {}'.format(i, load_sequential_range, len(x_train_quantum_tensor)))

        x_test_quantum_tensor = tf.io.parse_tensor(tf.io.read_file(
            'data/serialized_test_{}QUBITS{}_{}'.format(nqubits, '(compressed)' if EXTRA_COMPRESSION else '', 0)),
                                                   tf.string)
        for i in range(1, load_sequential_range):
            tensor = tf.io.parse_tensor(tf.io.read_file(
                'data/serialized_test_{}QUBITS{}_{}'.format(nqubits, '(compressed)' if EXTRA_COMPRESSION else '', i)),
                                        tf.string)
            x_test_quantum_tensor = tf.convert_to_tensor(
                np.concatenate([x_test_quantum_tensor.numpy(), tensor.numpy()]))
            print('loaded {}/{} testing. Current length: {}'.format(i, load_sequential_range, len(x_test_quantum_tensor)))

    return x_train_quantum_tensor, y_train, x_test_quantum_tensor, y_test


def circuit_to_tensor(x_circuits, parallel):
    ll = []
    if parallel:
        x_train_quantum_batches = np.array_split(np.array(x_circuits, dtype=object), 100)
        with mp.Pool(mp.cpu_count()) as pool:
            for batch in tqdm(x_train_quantum_batches):
                ll.extend(pool.map(_convert_to_tensor, batch))
    else:
        for x in tqdm(x_circuits):
            ll.extend([tfq.convert_to_tensor([x]).numpy()])
    x_tensor = tf.convert_to_tensor(np.concatenate(ll))
    return x_tensor


def get_quantum_tensors(nqubits, subset=None, load_tensors=False, load_sequential_range=None, save_tensors=False,
                        step=0, single_label=True, parallel=True, extra_compression_factor=0):
    if load_tensors:
        x_train_quantum_tensor, y_train, \
        x_test_quantum_tensor, y_test = load_quantum_tensors(nqubits, subset, load_tensors,
                                                             load_sequential_range, step, single_label)
    else:
        t1 = time.time()
        x_train_quantum, y_train, x_test_quantum, y_test = get_quantum_data(nqubits, subset, load_tensors,
                                                                            step, single_label, parallel)
        print('time to generate circuits : {}'.format(time.time()-t1))

        print('\n----- converting train circuits to tensors ------')
        t1 = time.time()
        x_train_quantum_tensor = circuit_to_tensor(x_train_quantum, parallel)
        if save_tensors:
            tf.io.write_file('data/serialized_train_{}QUBITS{}_{}'.format(
                nqubits, '(compressed)' if EXTRA_COMPRESSION else '', step),
                tf.io.serialize_tensor(x_train_quantum_tensor))

        print('\n----- converting test circuits to tensors ------')
        x_test_quantum_tensor = circuit_to_tensor(x_test_quantum, parallel)
        if save_tensors:
            tf.io.write_file('data/serialized_test_{}QUBITS{}_{}'.format(
                nqubits, '(compressed)' if EXTRA_COMPRESSION else '', step),
                tf.io.serialize_tensor(x_test_quantum_tensor))
        print('time to generate tensors: {}'.format(time.time()-t1))

    y_train_hinge = 2.0 * y_train - 1.0
    y_test_hinge = 2.0 * y_test - 1.0

    return x_train_quantum_tensor, y_train_hinge, x_test_quantum_tensor, y_test_hinge
