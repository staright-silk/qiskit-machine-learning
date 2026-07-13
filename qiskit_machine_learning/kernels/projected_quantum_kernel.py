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
"""Projected Quantum Kernel"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from qiskit import QuantumCircuit
from qiskit.primitives.base import BaseEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.passmanager import BasePassManager

from ..primitives import QMLEstimator as Estimator
from .base_kernel import BaseKernel

# Default one-qubit Pauli observables used to build the projected feature vector.
_DEFAULT_PAULIS = ("X", "Y", "Z")


class ProjectedQuantumKernel(BaseKernel):
    r"""
    A quantum kernel whose feature space is built from reduced (marginal) single-qubit
    density matrices rather than from the overlap of full statevectors.

    Fidelity-type kernels such as :class:`~qiskit_machine_learning.kernels.FidelityQuantumKernel`
    and :class:`~qiskit_machine_learning.kernels.FidelityStatevectorKernel` compute

    .. math::

        K(x,y) = |\langle \phi(x) | \phi(y) \rangle|^2,

    which is known to concentrate exponentially towards a constant value as the number of
    qubits grows, making the kernel matrix numerically indistinguishable from the identity
    and destroying any classification/regression signal. This effect (and the fact that it
    prevents fidelity kernels from being trained on classically "easy" data without an
    explicit bias) is discussed in Huang *et al.* [1].

    A *projected* quantum kernel avoids full-state overlaps entirely. Each encoded state
    :math:`|\phi(x)\rangle` is first projected down to a vector of reduced physical
    quantities -- by default, the single-qubit Pauli expectation values

    .. math::

        \phi(x) = \Big( \operatorname{Tr}[\rho_k(x) P] \Big)_{k \in \{1, \dots, n\}, \, P \in
        \{X, Y, Z\}}

    where :math:`\rho_k(x) = \operatorname{Tr}_{\bar{k}}\big[ |\phi(x)\rangle\langle\phi(x)|
    \big]` is the reduced density matrix of qubit :math:`k`. A classical kernel (a Gaussian/RBF
    kernel by default) is then evaluated on these vectors:

    .. math::

        K(x, y) = \exp\Big( -\gamma \, \lVert \phi(x) - \phi(y) \rVert_2^2 \Big).

    Because :math:`\phi(x)` only contains local, few-body information, this construction sidesteps
    the exponential concentration of full-state fidelities and was shown by Huang *et al.* [1] to
    admit a *provable* prediction advantage over every classical model on a specifically
    engineered dataset, while remaining a well behaved, trainable kernel on generic data.

    Unlike a statevector-only implementation, the reduced expectation values
    :math:`\operatorname{Tr}[\rho_k(x) P]` are single-qubit *observable* expectation values, so
    this class is built directly on the :class:`~qiskit.primitives.BaseEstimatorV2` primitive
    interface -- the same abstraction used throughout the rest of the library (see
    :class:`~qiskit_machine_learning.neural_networks.EstimatorQNN`). This means
    ``ProjectedQuantumKernel`` runs unmodified on exact statevector simulation, shot-based
    simulation, or real quantum hardware, simply by supplying a different ``estimator``.
    All requests for a data point are packed into a single batched job (one
    :class:`~qiskit.primitives.containers.EstimatorPub` per unique input, containing all
    ``num_qubits * len(paulis)`` local observables at once), and evaluated projections are
    cached so that repeated points (e.g. across kernel training iterations) are never
    resubmitted to the estimator.

    **References:**
    [1] Huang, H.Y., Broughton, M., Mohseni, M. *et al.* Power of data in quantum machine
    learning. *Nat Commun* **12**, 2631 (2021).
    `arXiv:2011.01938 [quant-ph] <https://arxiv.org/abs/2011.01938>`_
    """

    def __init__(
        self,
        *,
        feature_map: QuantumCircuit | None = None,
        estimator: BaseEstimatorV2 | None = None,
        gamma: float = 1.0,
        paulis: Sequence[str] = _DEFAULT_PAULIS,
        precision: float = 0.0,
        pass_manager: BasePassManager | None = None,
        cache_size: int | None = None,
        auto_clear_cache: bool = True,
        enforce_psd: bool = True,
    ) -> None:
        """
        Args:
            feature_map: Parameterized circuit to be used as the feature map. This is required:
                if ``None`` is given, a :class:`~qiskit_machine_learning.exceptions.
                QiskitMachineLearningError` is raised. If there's a mismatch in the number of
                qubits of the feature map and the number of features in the dataset, then the
                kernel will try to adjust the feature map to reflect the number of features.
            estimator: The estimator primitive used to evaluate the single-qubit projections.
                If ``None``, a default instance of the reference estimator,
                :class:`~qiskit_machine_learning.primitives.QMLEstimator`, is used in exact
                (analytic) mode. Any :class:`~qiskit.primitives.BaseEstimatorV2` implementation
                is accepted, including hardware/noisy backends, which allows this kernel to
                run identically in simulation and on real devices.
            gamma: The bandwidth parameter :math:`\\gamma` of the classical Gaussian kernel
                applied on top of the projected feature vectors. Must be positive.
            paulis: Which single-qubit Pauli observables to include in the projection for each
                qubit. Defaults to all three of ``("X", "Y", "Z")``, matching the construction
                used by Huang *et al.* Any non-empty subset of ``{"X", "Y", "Z"}`` is accepted.
            precision: Target precision passed through to ``estimator.run``. Defaults to ``0.0``,
                which requests exact expectation values from estimators that support it (such as
                the default :class:`~qiskit_machine_learning.primitives.QMLEstimator`); for
                shot-based estimators this is interpreted as the target standard error per
                observable.
            pass_manager: The pass manager used to transpile the feature map before submission
                to the estimator, if necessary. Defaults to ``None``, as some primitives do not
                require transpiled circuits.
            cache_size: Maximum number of unique data points whose projected feature vectors are
                cached. When ``None`` this is unbounded. Eviction only ever removes points that
                were not requested in the current :meth:`evaluate` call, so a single call
                containing more unique points than ``cache_size`` will temporarily grow the
                cache beyond this limit rather than drop data it is about to return.
            auto_clear_cache: Determines whether the projected-feature cache is retained when
                :meth:`evaluate` is called. The cache is automatically cleared by default.
            enforce_psd: Project to the closest positive semidefinite matrix if ``x = y``.

        Raises:
            ValueError: If ``gamma`` is not positive, or ``paulis`` is empty or contains an
                invalid label.
        """
        super().__init__(feature_map=feature_map, enforce_psd=enforce_psd)

        if gamma <= 0:
            raise ValueError(f"gamma must be positive, received {gamma}.")
        if not paulis:
            raise ValueError("paulis must be a non-empty sequence.")
        for label in paulis:
            if label not in ("X", "Y", "Z"):
                raise ValueError(f"Unsupported Pauli label '{label}', expected one of X, Y, Z.")

        if estimator is None:
            estimator = Estimator()
        self._estimator = estimator

        self._gamma = gamma
        self._paulis = tuple(paulis)
        self._precision = precision
        self._pass_manager = pass_manager
        self._cache_size = cache_size
        self._auto_clear_cache = auto_clear_cache

        if pass_manager is not None:
            self._feature_map = pass_manager.run(self._feature_map)

        self._observables = self._build_observables()
        self._feature_cache: dict[tuple[float, ...], np.ndarray] = {}

    def _build_observables(self) -> list[SparsePauliOp]:
        num_qubits = self._feature_map.num_qubits
        observables = []
        for qubit in range(num_qubits):
            for label in self._paulis:
                observables.append(
                    SparsePauliOp.from_sparse_list([(label, [qubit], 1.0)], num_qubits=num_qubits)
                )
        return observables

    @property
    def estimator(self) -> BaseEstimatorV2:
        """Returns the estimator primitive used to evaluate the local observables."""
        return self._estimator

    @property
    def gamma(self) -> float:
        """Returns the bandwidth parameter of the classical Gaussian kernel."""
        return self._gamma

    @property
    def paulis(self) -> tuple[str, ...]:
        """Returns the single-qubit Pauli observables used in the projection."""
        return self._paulis

    def evaluate(
        self,
        x_vec: np.ndarray,
        y_vec: np.ndarray | None = None,
    ) -> np.ndarray:
        if self._auto_clear_cache:
            self.clear_cache()

        x_vec, y_vec = self._validate_input(x_vec, y_vec)

        is_symmetric = True
        if y_vec is None:
            y_vec = x_vec
        elif not np.array_equal(x_vec, y_vec):
            is_symmetric = False

        x_features = self._get_projected_features(x_vec)
        y_features = x_features if is_symmetric else self._get_projected_features(y_vec)

        # Pairwise squared Euclidean distances between projected feature vectors.
        diffs = x_features[:, np.newaxis, :] - y_features[np.newaxis, :, :]
        sq_dists = np.sum(diffs**2, axis=-1)
        kernel_matrix = np.exp(-self._gamma * sq_dists)

        if self._enforce_psd and is_symmetric:
            kernel_matrix = self._make_psd(kernel_matrix)

        return kernel_matrix

    def _get_projected_features(self, data: np.ndarray) -> np.ndarray:
        keys = [tuple(row) for row in data]
        unique_keys = dict.fromkeys(keys)
        missing = [key for key in unique_keys if key not in self._feature_cache]

        if missing:
            # Batch every missing data point into a single estimator job: one PUB per point,
            # each PUB evaluating all local observables at once.
            pubs = [(self._feature_map, self._observables, list(key)) for key in missing]
            job = self._estimator.run(pubs, precision=self._precision)
            results = job.result()
            for key, result in zip(missing, results):
                self._feature_cache[key] = np.asarray(result.data.evs, dtype=float)

            if self._cache_size is not None:
                # Evict oldest entries first, but never evict a key requested in this very
                # call -- eviction must not invalidate the batch we're about to return.
                evictable = [key for key in self._feature_cache if key not in unique_keys]
                num_to_evict = max(0, len(self._feature_cache) - self._cache_size)
                for key in evictable[:num_to_evict]:
                    del self._feature_cache[key]

        return np.asarray([self._feature_cache[key] for key in keys])

    def clear_cache(self):
        """Clear the projected-feature cache."""
        self._feature_cache.clear()
