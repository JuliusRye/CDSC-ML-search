from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    pauli_error,
    depolarizing_error,
    thermal_relaxation_error,
)

def get_noise_model(name: str, *args) -> NoiseModel:
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
        NoiseModel: The constructed Qiskit noise model.
    """
    match name:
        case "spin":
            noise_model_spin = NoiseModel()

            # Example realistic noise parameters for spin qubits
            if len(args) == 0:
                noise_model_settings = {
                    "op_time": {
                        "R": 50e-9,    # Reset time (seconds)
                        "M": 400e-9,   # Measurement time (seconds)
                        "1q": 50e-9,   # 1-qubit gate time (seconds)
                        "2q": 200e-9   # 2-qubit gate time (seconds)
                    },
                    "thermal_relaxation": {
                        "T1": 100e-6,  # T1 relaxation time (seconds)
                        "T2": 80e-6    # T2 dephasing time (seconds)
                    },
                    "p_depol": 0.001,  # depolarizing error probability for gates
                    "meas_error_prob": 0.02 # measurement error probability
                }
            else:
                noise_model_settings = args[0]

            # Depolarizing errors
            error_1q = depolarizing_error(noise_model_settings["p_depol"], 1)
            error_2q = depolarizing_error(noise_model_settings["p_depol"], 2)

            # Thermal relaxation errors
            error_reset_tr = thermal_relaxation_error(noise_model_settings["thermal_relaxation"]["T1"], noise_model_settings["thermal_relaxation"]["T2"], noise_model_settings["op_time"]["R"])
            error_meas_tr = thermal_relaxation_error(noise_model_settings["thermal_relaxation"]["T1"], noise_model_settings["thermal_relaxation"]["T2"], noise_model_settings["op_time"]["M"])
            error_1q_tr = thermal_relaxation_error(noise_model_settings["thermal_relaxation"]["T1"], noise_model_settings["thermal_relaxation"]["T2"], noise_model_settings["op_time"]["1q"])
            error_2q_tr = thermal_relaxation_error(noise_model_settings["thermal_relaxation"]["T1"], noise_model_settings["thermal_relaxation"]["T2"], noise_model_settings["op_time"]["2q"])

            # Measurement readout error
            meas_error = ReadoutError([[1-noise_model_settings["meas_error_prob"], noise_model_settings["meas_error_prob"]],
                                    [noise_model_settings["meas_error_prob"], 1-noise_model_settings["meas_error_prob"]]])

            # Add errors to the noise model
            noise_model_spin.add_all_qubit_quantum_error(error_1q.compose(error_1q_tr), ['h', 'reset', 'x', 'y', 'z', 'sx', 'sxdg', 't', 'tdg'])
            noise_model_spin.add_all_qubit_quantum_error(error_2q.compose(error_2q_tr), ['cx', 'cz', 'cy'])
            noise_model_spin.add_all_qubit_readout_error(meas_error, 'measure')
            noise_model_spin.add_all_qubit_quantum_error(error_reset_tr, ['Idle_R'])
            noise_model_spin.add_all_qubit_quantum_error(error_meas_tr, ['Idle_M'])
            noise_model_spin.add_all_qubit_quantum_error(error_1q_tr, ['Idle_1q'])
            noise_model_spin.add_all_qubit_quantum_error(error_2q_tr, ['Idle_2q'])

            return noise_model_spin
        case "SI1000":
            noise_model_sc = NoiseModel()

            if len(args) == 0:
                prob = 0.001  # Default error probability
            else:
                prob = args[0]

            # CZ(p)
            noise_model_sc.add_all_qubit_quantum_error(depolarizing_error(prob, 2), ['cz'])
            # AnyClifford1(p/10)
            noise_model_sc.add_all_qubit_quantum_error(depolarizing_error(prob / 10, 1), ['h', 'x', 'y', 'z', 's', 't'])
            # InitZ (2p)
            noise_model_sc.add_all_qubit_quantum_error(pauli_error([('I', 1 - 2 * prob), ('X', 2 * prob)]), ['reset'])
            # MZ (5p)
            noise_model_sc.add_all_qubit_readout_error(ReadoutError([[1 - 5 * prob, 5 * prob], [5 * prob, 1 - 5 * prob]]), 'measure')
            # Idle(p/10)
            noise_model_sc.add_all_qubit_quantum_error(depolarizing_error(prob / 10, 1), ['Idle'])
            # ResonatorIdle(2p)
            noise_model_sc.add_all_qubit_quantum_error(depolarizing_error(2 * prob, 1), ['ResonatorIdle'])

            return noise_model_sc
        case _:
            raise ValueError(f"Unknown noise model name: {name}")