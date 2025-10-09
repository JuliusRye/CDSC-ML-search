from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    pauli_error,
    depolarizing_error,
    thermal_relaxation_error,
)
import stim
import numpy as np

def get_noise_model(name: str, *args) -> callable:
    """
    Returns a Qiskit noise model based on the specified name.

    Supported noise models:
    - ``spin`` — a noise model tailored for spin qubits, including depolarizing and thermal relaxation errors.
    - ``SI1000`` — a noise model based on the [SI1000](https://quantum-journal.org/papers/q-2021-12-20-605/pdf/)
      architecture with depolarizing and readout errors.

    Args:
        name (str): The name of the noise model to create.
        *args: Additional arguments required for specific noise models.

    Returns:
        callable: A function `fun` that applies the noise model to a Stim circuit `circ` with optional arguments depending on the selected noise model. Example usage: `fun(circ)`
    """
    def thermal_relaxation_error_to_pauli_probs(dwell_time: float, T1: float, T2: float) -> dict:
        """
        Converts thermal relaxation error parameters to equivalent Pauli error probabilities.

        Args:
            dwell_time (float): The time the qubit spends in the idle state (seconds).
            T1 (float): The T1 relaxation time (seconds).
            T2 (float): The T2 dephasing time (seconds).

        Returns:
            dict: A dictionary with keys 'X', 'Y', and 'Z' representing the probabilities of X, Y, and Z errors.
        """
        assert T1 >= 0, "T1 must be positive."
        assert T2 >= 0, "T2 must be positive."
        assert dwell_time >= 0, "Dwell time must be non-negative."
        assert T2 <= 2 * T1, "T2 must be less than or equal to 2*T1."

        # Calculate the probabilities of no error
        p_relax = np.exp(-dwell_time / T1)
        p_dephase = np.exp(-dwell_time / T2)

        # Calculate the probabilities of X, Y, and Z errors
        p_x = p_y = (1 - p_relax) / 4
        p_z = (1 - 2*p_dephase + p_relax) / 4

        return [p_x, p_y, p_z]

    match name:
        case "spin":
            def spin_noise_model(circ: stim.Circuit, noise_model: dict = None) -> stim.Circuit:
                """
                Adds noise to a Stim circuit based on the defined noise model.

                Args:
                    circ (stim.Circuit): The original Stim circuit.
                    noise_model (dict): A dictionary defining the noise parameters. If None, default parameters are used.

                Returns:
                    stim.Circuit: The Stim circuit with noise added.
                """
                def flatten(lst):
                    return [item.qubit_value for sublist in lst for item in sublist]
                
                if noise_model is None:
                    noise_model = {
                        "reset": 0.01,
                        "meas": 0.02,
                        "1q": [0.001, 0.001, 0.001],
                        "2q": [0.001]*15,
                        "T1": 100e-6,
                        "T2": 80e-6,
                        "reset_time": 50e-9,
                        "meas_time": 400e-9,
                        "1q_time": 50e-9,
                        "2q_time": 200e-9
                    }

                T1 = noise_model.get("T1", 100e-6)
                T2 = noise_model.get("T2", 100e-6)
                thermal_relaxation = {
                    "reset": thermal_relaxation_error_to_pauli_probs(noise_model.get("reset_time", 0.0), T1, T2),
                    "meas": thermal_relaxation_error_to_pauli_probs(noise_model.get("meas_time", 0.0), T1, T2),
                    "1q": thermal_relaxation_error_to_pauli_probs(noise_model.get("1q_time", 0.0), T1, T2),
                    "2q": thermal_relaxation_error_to_pauli_probs(noise_model.get("2q_time", 0.0), T1, T2),
                }

                noisy_circ = stim.Circuit()
                qubits = set(range(circ.num_qubits))
                idling = qubits.copy()
                
                tick_type = "reset"
                for op in circ:
                    if op.name == "REPEAT":
                        noisy_circ += spin_noise_model(op.body_copy(), noise_model) * op.repeat_count
                        continue
                    target = flatten(op.target_groups())
                    if op.name in ["R"]:
                        noisy_circ.append(op)
                        noisy_circ.append("X_ERROR", target, noise_model.get("reset", 0.0))
                        idling.difference_update(target)
                        continue
                    if op.name in ["M", "MX", "MY", "MZ"]:
                        noisy_circ.append("X_ERROR", target, noise_model.get("meas", 0.0))
                        noisy_circ.append(op)
                        idling.difference_update(target)
                        continue
                    if op.name in ["X", "Y", "Z", "C_XYZ", "C_ZYX", "H", "H_XY", "H_XZ", "H_YZ", "S", "SQRT_X", "SQRT_X_DAG", "SQRT_Y", "SQRT_Y_DAG", "SQRT_Z", "SQRT_Z_DAG", "S_DAG"]:
                        noisy_circ.append(op)
                        noisy_circ.append("PAULI_CHANNEL_1", target, noise_model.get("1q", [0.0]*3))
                        idling.difference_update(target)
                        continue
                    if op.name in ["CX", "CZ", "CY"]:
                        noisy_circ.append(op)
                        noisy_circ.append("PAULI_CHANNEL_2", target, noise_model.get("2q", [0.0]*15))
                        idling.difference_update(target)
                        continue
                    if op.name in ["TICK"]:
                        # Handle idling qubits
                        if idling != qubits:
                            noisy_circ.append("PAULI_CHANNEL_1", list(idling), thermal_relaxation[tick_type])       
                        # Handle tick
                        noisy_circ.append(op)
                        tick_type = op.tag
                        assert tick_type in ["reset", "meas", "1q", "2q"], f"Unexpected tick type: {tick_type}"
                        idling = qubits.copy()
                        continue
                    if op.name in ["QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"]:
                        noisy_circ.append(op)
                        continue
                    raise ValueError(f"Unexpected operation: {op.name}")
                # Handle idling qubits at the end of the circuit
                if idling != qubits:
                    noisy_circ.append("PAULI_CHANNEL_1", list(idling), thermal_relaxation[tick_type])

                return noisy_circ
            return spin_noise_model
        case "SI1000":
            def SI1000_noise_model(circ: stim.Circuit, prob: float = 0.001) -> stim.Circuit:
                """
                Adds noise to a Stim circuit based on the SI1000 noise model.

                Args:
                    circ (stim.Circuit): The original Stim circuit.
                    prob (float): The error probability.

                Returns:
                    stim.Circuit: The Stim circuit with noise added.
                """
                def flatten(lst):
                    return [item.qubit_value for sublist in lst for item in sublist]
                
                noisy_circ = stim.Circuit()
                qubits = set(range(circ.num_qubits))
                idling = qubits.copy()
                
                for op in circ:
                    if op.name == "REPEAT":
                        noisy_circ += SI1000_noise_model(op.body_copy(), prob) * op.repeat_count
                        continue
                    target = flatten(op.target_groups())
                    if op.name in ["CZ"]:
                        noisy_circ.append(op)
                        noisy_circ.append("PAULI_CHANNEL_2", target, [prob/15]*15)
                        idling.difference_update(target)
                        continue
                    if op.name in ["H", "X", "Y", "Z", "S", "T"]:
                        noisy_circ.append(op)
                        noisy_circ.append("PAULI_CHANNEL_1", target, [prob/10/3]*3)
                        idling.difference_update(target)
                        continue
                    if op.name in ["R"]:
                        noisy_circ.append(op)
                        noisy_circ.append("X_ERROR", target, 2*prob)
                        idling.difference_update(target)
                        continue
                    if op.name in ["M", "MZ"]:
                        noisy_circ.append("X_ERROR", target, 5*prob)
                        noisy_circ.append(op)
                        idling.difference_update(target)
                        continue
                    if op.name in ["TICK"]:
                        # Handle idling qubits
                        if idling != qubits:
                            if tick_type == "meas":
                                noisy_circ.append("DEPOLARIZE1", list(idling), 2*prob)
                            else:
                                noisy_circ.append("DEPOLARIZE1", list(idling), prob/10)
                        # Handle tick
                        noisy_circ.append(op)
                        tick_type = op.tag
                        assert tick_type in ["reset", "meas", "1q", "2q"], f"Unexpected tick type: {tick_type}"
                        idling = qubits.copy()
                        continue
                    if op.name in ["QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE"]:
                        noisy_circ.append(op)
                    raise ValueError(f"Unexpected operation: {op.name}")
                return noisy_circ
            return SI1000_noise_model
        case _:
            raise ValueError(f"Unknown noise model name: {name}")