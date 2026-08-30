"""Quantum-layer backend factories: PennyLane and Qiskit implementations behind
the same nn.Module interface (forward(x: (batch, n_qubits)) -> (batch, n_qubits)),
so a caller (e.g. QHBERTModel) can swap backends via a config string instead of
hand-editing circuit code per experiment.

Both circuits read one feature per qubit (angle/data encoding) and return one
expectation value per qubit (Pauli-Z readout) — same input/output contract,
different simulator underneath.
"""

import pennylane as qml
import torch.nn as nn


def build_pennylane_layer(n_qubits: int, n_layers: int) -> nn.Module:
    """AngleEmbedding -> [ring-CNOT, RY, RZ] x n_layers -> per-qubit PauliZ.

    Same circuit as kaggle/qhbert_end_to_end.ipynb's QuantumLayer, generalized
    over qubit/layer count for the ablation sweep.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    # backprop, not parameter-shift: this is a noiseless simulator, so there's nothing
    # parameter-shift buys over exact backprop through the statevector — and backprop is
    # the one that supports batched (broadcasted) input; parameter-shift raises
    # NotImplementedError for broadcasted tapes with trainable weights (PennyLane #4462).
    # Measured on 12 qubits/3 layers/batch 32: parameter-shift ~95s/batch, backprop ~1.3s/batch.
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            for qubit in range(n_qubits):
                qml.RY(weights[layer, qubit, 0], wires=qubit)
                qml.RZ(weights[layer, qubit, 1], wires=qubit)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 2)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


_FEATURE_MAPS = {}
_ANSATZE = {}


def _qiskit_builders():
    # Imported lazily so importing this module doesn't require qiskit unless
    # the qiskit backend is actually used.
    if not _FEATURE_MAPS:
        from qiskit.circuit.library import (
            efficient_su2,
            pauli_feature_map,
            real_amplitudes,
            zz_feature_map,
        )

        _FEATURE_MAPS.update({"zz": zz_feature_map, "pauli": pauli_feature_map})
        _ANSATZE.update({"real_amplitudes": real_amplitudes, "efficient_su2": efficient_su2})
    return _FEATURE_MAPS, _ANSATZE


def build_qiskit_layer(
    n_qubits: int,
    n_layers: int,
    feature_map: str = "zz",
    ansatz: str = "real_amplitudes",
) -> nn.Module:
    """ZZFeatureMap/PauliFeatureMap + RealAmplitudes/EfficientSU2 ansatz ->
    EstimatorQNN -> TorchConnector. Drop-in replacement for
    build_pennylane_layer's output above (per-qubit PauliZ expectation).

    `input_gradients=True` is required (not the EstimatorQNN default) because
    this layer sits behind a trainable classical bridge network in QHBERTModel
    — without it, gradients wouldn't flow back past the quantum layer during
    backprop.
    """
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_machine_learning.gradients import SPSAEstimatorGradient
    from qiskit_machine_learning.neural_networks import EstimatorQNN

    feature_maps, ansatze = _qiskit_builders()
    fmap = feature_maps[feature_map](feature_dimension=n_qubits, reps=1)
    ansatz_circuit = ansatze[ansatz](num_qubits=n_qubits, reps=n_layers)
    circuit = fmap.compose(ansatz_circuit)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (n_qubits - i - 1)) for i in range(n_qubits)
    ]

    # SPSA gradient, not the EstimatorQNN default (ParamShiftEstimatorGradient): parameter-shift
    # needs O(2 x num_params) circuit evaluations per sample and has no batched/backprop path here
    # (Qiskit's primitives don't support autodiff through the simulator the way PennyLane's
    # default.qubit does) — measured unusable at 12 qubits/3 layers (minutes per batch of 32,
    # never completed a full epoch). SPSA needs O(1) evaluations per step regardless of param
    # count: ~36s/batch at 12q/3L, ~1.4s/batch at 4q/2L on the same hardware. Estimates are noisier
    # than an exact gradient, which is the standard, accepted trade-off for SPSA in VQE/QML training.
    estimator = StatevectorEstimator()
    gradient = SPSAEstimatorGradient(estimator=estimator, epsilon=0.01)
    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observables,
        input_params=list(fmap.parameters),
        weight_params=list(ansatz_circuit.parameters),
        input_gradients=True,
        estimator=estimator,
        gradient=gradient,
    )
    return TorchConnector(qnn)


BACKENDS = {"pennylane": build_pennylane_layer, "qiskit": build_qiskit_layer}


def build_quantum_layer(backend: str, n_qubits: int, n_layers: int, **kwargs) -> nn.Module:
    """Single entry point used by QHBERTModel; kwargs (feature_map/ansatz) are
    ignored by the pennylane backend since it has no equivalent knobs."""
    if backend == "pennylane":
        return build_pennylane_layer(n_qubits, n_layers)
    if backend == "qiskit":
        return build_qiskit_layer(n_qubits, n_layers, **kwargs)
    raise ValueError(f"Unknown backend: {backend!r} (expected 'pennylane' or 'qiskit')")
