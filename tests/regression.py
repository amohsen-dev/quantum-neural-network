import unittest
from run import run
from config import N_QUBITS, N_LAYERS, SEED, BATCH_SIZE, EPOCHS


class FullNetworkTrainingTest(unittest.TestCase):

    # Returns True or False.
    def test_vanilla(self):
        result = run(nqubits=N_QUBITS,
                     nlayers=N_LAYERS,
                     subset=500,
                     step=0,
                     load_tensors=False,
                     save_tensors=False,
                     multi_label=False,
                     load_sequential_range=None,
                     parallel=False,
                     extra_compression_factor=0,
                     seed=SEED,
                     batch_size=BATCH_SIZE,
                     nepochs=EPOCHS)

        reference = [('loss', 0.8487324118614197), ('hinge_accuracy', 0.8550781011581421)]

        self.assertAlmostEqual(first=result[0][1], second=reference[0][1], places=6,
                               msg='{} changed, fix or rebase'.format(reference[0][0]))

        self.assertAlmostEqual(first=result[1][1], second=reference[1][1], places=6,
                               msg='{} changed, fix or rebase'.format(reference[1][0]))

    def test_extra_compression(self):
        result = run(nqubits=N_QUBITS,
                     nlayers=N_LAYERS,
                     subset=200,
                     step=0,
                     load_tensors=False,
                     save_tensors=False,
                     multi_label=False,
                     load_sequential_range=None,
                     parallel=False,
                     extra_compression_factor=2,
                     seed=SEED,
                     batch_size=BATCH_SIZE,
                     nepochs=EPOCHS)

        reference = [('loss', 0.9448315501213074), ('hinge_accuracy', 0.718750)]

        self.assertAlmostEqual(first=result[0][1], second=reference[0][1], places=6,
                               msg='{} changed, fix or rebase'.format(reference[0][0]))

        self.assertAlmostEqual(first=result[1][1], second=reference[1][1], places=6,
                               msg='{} changed, fix or rebase'.format(reference[1][0]))


if __name__ == '__main__':
    unittest.main()