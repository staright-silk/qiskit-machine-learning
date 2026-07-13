# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2026.
# (C) Copyright UKRI-STFC (Hartree Centre) 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Test ProjectedQuantumKernel."""

from __future__ import annotations

import pickle
import unittest

from test import QiskitMachineLearningTestCase

import numpy as np
from sklearn.svm import SVC

from qiskit.circuit.library import z_feature_map, zz_feature_map
from qiskit.quantum_info import Statevector, partial_trace, Pauli

from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.kernels import ProjectedQuantumKernel


class TestProjectedQuantumKernel(QiskitMachineLearningTestCase):
    """Test ProjectedQuantumKernel."""

    def setUp(self):
        super().setUp()

        algorithm_globals.random_seed = 10598

        self.feature_map = z_feature_map(feature_dimension=2, reps=2)
        self.zz_feature_map = zz_feature_map(feature_dimension=2)

        self.sample_train = np.asarray(
            [
                [3.07876080, 1.75929189],
                [6.03185789, 5.27787566],
                [6.22035345, 2.70176968],
                [0.18849556, 2.82743339],
            ]
        )
        self.label_train = np.asarray([0, 0, 1, 1])

        self.sample_test = np.asarray([[2.199114860, 5.15221195], [0.50265482, 0.06283185]])
        self.label_test = np.asarray([0, 1])

    def test_input_validation(self):
        """Test invalid constructor arguments are rejected."""
        with self.subTest("non-positive gamma"):
            with self.assertRaises(ValueError):
                ProjectedQuantumKernel(feature_map=self.feature_map, gamma=0)
            with self.assertRaises(ValueError):
                ProjectedQuantumKernel(feature_map=self.feature_map, gamma=-1.0)

        with self.subTest("empty paulis"):
            with self.assertRaises(ValueError):
                ProjectedQuantumKernel(feature_map=self.feature_map, paulis=())

        with self.subTest("invalid pauli label"):
            with self.assertRaises(ValueError):
                ProjectedQuantumKernel(feature_map=self.feature_map, paulis=("X", "Q"))

    def test_diagonal_is_one(self):
        """K(x, x) must always be exactly 1 since the projected feature vectors coincide."""
        kernel = ProjectedQuantumKernel(feature_map=self.zz_feature_map)
        matrix = kernel.evaluate(self.sample_train)
        np.testing.assert_allclose(np.diag(matrix), np.ones(len(self.sample_train)))

    def test_symmetric(self):
        """The kernel matrix for a single input set must be symmetric."""
        kernel = ProjectedQuantumKernel(feature_map=self.zz_feature_map)
        matrix = kernel.evaluate(self.sample_train)
        np.testing.assert_allclose(matrix, matrix.T)

    def test_matches_manual_projection(self):
        """Cross-check evaluate() against an independent, brute-force computation."""
        kernel = ProjectedQuantumKernel(feature_map=self.feature_map, gamma=0.5)
        matrix = kernel.evaluate(self.sample_train)

        expected_features = np.array(
            [self._manual_projection(self.feature_map, x) for x in self.sample_train]
        )
        diffs = expected_features[:, np.newaxis, :] - expected_features[np.newaxis, :, :]
        expected_matrix = np.exp(-0.5 * np.sum(diffs**2, axis=-1))

        np.testing.assert_allclose(matrix, expected_matrix, rtol=1e-6, atol=1e-10)

    def test_asymmetric_matches_manual_projection(self):
        """Cross-check the asymmetric evaluate(x, y) path too."""
        kernel = ProjectedQuantumKernel(feature_map=self.feature_map, gamma=0.5)
        matrix = kernel.evaluate(self.sample_train, self.sample_test)

        x_features = np.array(
            [self._manual_projection(self.feature_map, x) for x in self.sample_train]
        )
        y_features = np.array(
            [self._manual_projection(self.feature_map, y) for y in self.sample_test]
        )
        diffs = x_features[:, np.newaxis, :] - y_features[np.newaxis, :, :]
        expected_matrix = np.exp(-0.5 * np.sum(diffs**2, axis=-1))

        np.testing.assert_allclose(matrix, expected_matrix, rtol=1e-6, atol=1e-10)

    @staticmethod
    def _manual_projection(feature_map, params):
        qc = feature_map.assign_parameters(params)
        sv = Statevector(qc)
        num_qubits = sv.num_qubits
        features = []
        for qubit in range(num_qubits):
            traced_out = [q for q in range(num_qubits) if q != qubit]
            reduced = partial_trace(sv, traced_out)
            for label in ("X", "Y", "Z"):
                features.append(np.real(reduced.expectation_value(Pauli(label))))
        return np.array(features)

    def test_custom_paulis(self):
        """Restricting to a subset of Paulis should shrink the feature dimension accordingly."""
        kernel = ProjectedQuantumKernel(feature_map=self.feature_map, paulis=("Z",))
        # pylint: disable=protected-access
        features = kernel._get_projected_features(tuple(self.sample_train[0]))
        self.assertEqual(features.shape[0], self.feature_map.num_qubits)

    def test_svc_callable(self):
        """Test callable kernel in sklearn."""
        kernel = ProjectedQuantumKernel(feature_map=self.feature_map, gamma=1.0)
        svc = SVC(kernel=kernel.evaluate)
        svc.fit(self.sample_train, self.label_train)
        score = svc.score(self.sample_test, self.label_test)

        self.assertGreaterEqual(score, 0.5)

    def test_svc_precomputed(self):
        """Test precomputed kernel in sklearn."""
        kernel = ProjectedQuantumKernel(feature_map=self.feature_map, gamma=1.0)
        kernel_train = kernel.evaluate(x_vec=self.sample_train)
        kernel_test = kernel.evaluate(x_vec=self.sample_test, y_vec=self.sample_train)

        svc = SVC(kernel="precomputed")
        svc.fit(kernel_train, self.label_train)
        score = svc.score(kernel_test, self.label_test)

        self.assertGreaterEqual(score, 0.5)

    def test_projected_cache(self):
        """Test filling and clearing the reduced-state cache."""
        kernel = ProjectedQuantumKernel(feature_map=self.zz_feature_map, auto_clear_cache=False)
        svc = SVC(kernel=kernel.evaluate)
        svc.fit(self.sample_train, self.label_train)
        with self.subTest("Check cache fills correctly."):
            # pylint: disable=no-member
            self.assertEqual(
                kernel._get_projected_features.cache_info().currsize, len(self.sample_train)
            )

        svc.fit(self.sample_test, self.label_test)
        with self.subTest("Check no auto_clear_cache."):
            # pylint: disable=no-member
            self.assertEqual(
                kernel._get_projected_features.cache_info().currsize,
                len(self.sample_train) + len(self.sample_test),
            )

        kernel = ProjectedQuantumKernel(
            feature_map=self.zz_feature_map, cache_size=3, auto_clear_cache=False
        )
        svc = SVC(kernel=kernel.evaluate)
        svc.fit(self.sample_train, self.label_train)
        with self.subTest("Check cache limit respected."):
            # pylint: disable=no-member
            self.assertEqual(kernel._get_projected_features.cache_info().currsize, 3)

        kernel.clear_cache()
        with self.subTest("Check cache clears correctly"):
            # pylint: disable=no-member
            self.assertEqual(kernel._get_projected_features.cache_info().currsize, 0)

    def test_enforce_psd(self):
        """A Gaussian kernel matrix is PSD by construction; check the flag doesn't break this."""
        kernel = ProjectedQuantumKernel(feature_map=self.zz_feature_map, enforce_psd=True)
        matrix = kernel.evaluate(self.sample_train)
        eigenvalues = np.linalg.eigvals(matrix)
        self.assertTrue(np.all(np.greater_equal(eigenvalues, -1e-10)))

    def test_no_feature_map_raises(self):
        """Constructing without a feature map must raise, matching BaseKernel's contract."""
        from qiskit_machine_learning.exceptions import QiskitMachineLearningError

        with self.assertRaises(QiskitMachineLearningError):
            ProjectedQuantumKernel(feature_map=None)

    def test_pickle(self):
        """Test that a kernel (and its cache machinery) survives a pickle round-trip."""
        kernel = ProjectedQuantumKernel(feature_map=self.feature_map)
        kernel.evaluate(self.sample_train)
        pickled = pickle.loads(pickle.dumps(kernel))
        matrix = pickled.evaluate(self.sample_train)
        expected = kernel.evaluate(self.sample_train)
        np.testing.assert_allclose(matrix, expected)


if __name__ == "__main__":
    unittest.main()
