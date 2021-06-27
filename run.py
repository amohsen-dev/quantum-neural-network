import tensorflow as tf
import tensorflow_quantum as tfq
from config import N_QUBITS, N_LAYERS, SEED, BATCH_SIZE, EPOCHS, EXTRA_COMPRESSION_FACTOR
from data_processing import  get_quantum_tensors
import time
from gates import create_model, hinge_accuracy
from argparse import ArgumentParser


def run(nqubits, nlayers, subset, step, load_tensors, save_tensors, multi_label, load_sequential_range, parallel,
        extra_compression_factor, seed, batch_size, nepochs):

    tf.random.set_seed(seed)

    model_circuit, model_readout = create_model(n_qubits=nqubits, n_layers=nlayers)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        tfq.layers.PQC(model_circuit, model_readout),
    ])

    x_train, y_train, x_test, y_test = get_quantum_tensors(nqubits=nqubits,
                                                           subset=subset,
                                                           load_tensors=load_tensors,
                                                           load_sequential_range=load_sequential_range,
                                                           save_tensors=save_tensors,
                                                           step=step,
                                                           single_label=not multi_label,
                                                           parallel=parallel,
                                                           extra_compression_factor=extra_compression_factor)

    model.compile(
        loss=tf.keras.losses.Hinge(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[hinge_accuracy])

    print(model.summary())

    t1 = time.time()
    qnn_history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=nepochs,
        verbose=1,
        validation_data=(x_test, y_test))

    print('time to fit model : {}'.format(time.time() - t1))

    qnn_results = model.evaluate(x_test, y_test)

    return list(zip(model.metrics_names, qnn_results))


if __name__ == '__main__':
    parser = ArgumentParser(description='quantum neural network')

    parser.add_argument('--nlayers', '-l', action='store', type=int, required=False, default=N_LAYERS, help='')
    parser.add_argument('--nqubits', '-q', action='store', type=int, required=False, default=N_QUBITS, help='')
    parser.add_argument('--nepochs', '-e', action='store', type=int, required=False, default=EPOCHS, help='')
    parser.add_argument('--batch_size', '-b', action='store', type=int, required=False, default=BATCH_SIZE, help='')
    parser.add_argument('--seed', '-d', action='store', type=int, required=False, default=SEED, help='')
    parser.add_argument('--subset', '-s', action='store', type=int, required=False, default=None, help='')
    parser.add_argument('--step', '-i', action='store', type=int, required=False, default=None, help='')
    parser.add_argument('--load_tensors', '-r', action='store_true', required=False, default=False, help='')
    parser.add_argument('--save_tensors', '-w', action='store_true', required=False, default=False, help='')
    parser.add_argument('--parallel', '-p', action='store_true', required=False, default=False, help='')
    parser.add_argument('--multi_label', '-c', action='store_true', required=False, default=False, help='')
    parser.add_argument('--load_sequential_range', action='store', required=False, default=None, help='')
    parser.add_argument('--extra_compression_factor', action='store', type=int, required=False,
                        default=EXTRA_COMPRESSION_FACTOR, help='')

    args = parser.parse_args()

    run(nqubits=args.nqubits,
        nlayers=args.nlayers,
        subset=args.subset,
        step=args.step,
        load_tensors=args.load_tensors,
        save_tensors=args.save_tensors,
        multi_label=args.multi_label,
        load_sequential_range=args.load_sequential_range,
        parallel=args.parallel,
        extra_compression_factor=args.extra_compression_factor,
        seed=args.seed,
        batch_size=args.batch_size,
        nepochs=args.nepochs)

