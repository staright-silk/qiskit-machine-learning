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

from functools import lru_cache
from typing import Sequence, Type, TypeVar, Any

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace, Pauli

from .base_kernel import BaseKernel

SV = TypeVar("SV", bound=Statevector)

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

    This reference implementation is built directly on
    :class:`~qiskit.quantum_info.Statevector` (following the same design as
    :class:`~qiskit_machine_learning.kernels.FidelityStatevectorKernel`) and is therefore
    restricted to classically simulable feature maps/qubit counts. Reduced states are cached
    to avoid repeated circuit evaluation; the cache can be cleared with :meth:`clear_cache`.

    **References:**
    [1] Huang, H.Y., Broughton, M., Mohseni, M. *et al.* Power of data in quantum machine
    learning. *Nat Commun* **12**, 2631 (2021).
    `arXiv:2011.01938 [quant-ph] <https://arxiv.org/abs/2011.01938>`_
    """

    def __init__(
        self,
        *,
        feature_map: QuantumCircuit | None = None,
        statevector_type: Type[SV] = Statevector,
        gamma: float = 1.0,
        paulis: Sequence[str] = _DEFAULT_PAULIS,
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
            statevector_type: The type of Statevector that will be instantiated using the
                ``feature_map`` quantum circuit and used to compute the projected feature
                vectors. This type should inherit from (and defaults to)
                :class:`~qiskit.quantum_info.Statevector`.
            gamma: The bandwidth parameter :math:`\\gamma` of the classical Gaussian kernel
                applied on top of the projected feature vectors. Must be positive.
            paulis: Which single-qubit Pauli observables to include in the projection for each
                qubit. Defaults to all three of ``("X", "Y", "Z")``, matching the construction
                used by Huang *et al.* Any non-empty subset of ``{"X", "Y", "Z"}`` is accepted.
            cache_size: Maximum size of the reduced-state cache. When ``None`` this is unbounded.
            auto_clear_cache: Determines whether the reduced-state cache is retained when
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

        self._statevector_type = statevector_type
        self._gamma = gamma
        self._paulis = tuple(paulis)
        self._auto_clear_cache = auto_clear_cache
        self._cache_size = cache_size
        # Create the reduced-state cache at the instance level.
        self._get_projected_features = lru_cache(maxsize=cache_size)(self._get_projected_features_)

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

        return self._evaluate(x_vec, y_vec, is_symmetric)

    def _evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray, is_symmetric: bool) -> np.ndarray:
        x_features = np.asarray([self._get_projected_features(tuple(x)) for x in x_vec])
        if is_symmetric:
            y_features = x_features
        else:
            y_features = np.asarray([self._get_projected_features(tuple(y)) for y in y_vec])

        # Pairwise squared Euclidean distances between projected feature vectors.
        diffs = x_features[:, np.newaxis, :] - y_features[np.newaxis, :, :]
        sq_dists = np.sum(diffs**2, axis=-1)
        kernel_matrix = np.exp(-self._gamma * sq_dists)

        if self._enforce_psd and is_symmetric:
            kernel_matrix = self._make_psd(kernel_matrix)

        return kernel_matrix

    def _get_projected_features_(self, param_values: tuple[float]) -> np.ndarray:
        # lru_cache requires hashable function arguments.
        qc = self._feature_map.assign_parameters(param_values)
        statevector = self._statevector_type(qc)
        num_qubits = statevector.num_qubits

        features = np.empty(num_qubits * len(self._paulis))
        idx = 0
        for qubit in range(num_qubits):
            traced_out = [q for q in range(num_qubits) if q != qubit]
            reduced_state = partial_trace(statevector, traced_out)
            for label in self._paulis:
                features[idx] = np.real(reduced_state.expectation_value(Pauli(label)))
                idx += 1
        return features

    def clear_cache(self):
        """Clear the reduced-state cache."""
        # pylint: disable=no-member
        self._get_projected_features.cache_clear()

    def __getstate__(self) -> dict[str, Any]:
        kernel = dict(self.__dict__)
        kernel["_get_projected_features"] = None
        return kernel

    def __setstate__(self, kernel: dict[str, Any]):
        self.__dict__ = kernel
        self._get_projected_features = lru_cache(maxsize=self._cache_size)(
            self._get_projected_features_
        )
