from qecsim.models.rotatedplanar import RotatedPlanarCode
import jax.numpy as jnp
from qiskit.circuit import QuantumCircuit, AncillaRegister, QuantumRegister, ClassicalRegister
from qiskit_aer.library.save_instructions import SaveExpectationValue
from qiskit.quantum_info import Pauli
from qiskit.circuit.library import IGate
import numpy as np
import stim
from src.data_gen import transform_code_stabilizers

# Define idle gates to represent wait times (with idling errors) for 1-qubit and 2-qubit operations
idle_reset = IGate(label='Idle_R')
idle_meas = IGate(label='Idle_M')
idle_1q = IGate(label='Idle_1q')
idle_2q = IGate(label='Idle_2q')

def prepare_circuit_info(code: RotatedPlanarCode, deformation: jnp.ndarray):
    """
    Prepares information about the code needed to construct the quantum circuit

    Args:
        code (RotatedPlanarCode): The quantum error correcting code
        deformation (jnp.ndarray): The deformation applied to the code (An array of shape (num_data_qubits,) with entries in {0,1,2,3,4,5} representing the index of the Clifford deformation on each data qubit)

    Returns:
        tuple: A tuple containing:
            stabilizers (np.ndarray): The stabilizer matrix of the code
            syndrome_loc (list): The locations of the syndrome qubits
            data_loc (list): The locations of the data qubits
            connection_order (list): A list containing the order of operations where the elements in each sublist can be executed in parallel
    """
    # stabilizers = code.stabilizers
    # logicals = code.logicals
    # Transpose the coordinates to align with the plotting convention
    syndrome_loc = [(y,x) for x, y in code._plaquette_indices]
    data_loc = [(j-.5,i-.5) for j in range(code.size[0]) for i in range(code.size[1])]
    # Determine the order of connections based on relative positions (Standard surface code layout is assumed)
    # The order is independent of the deformation, as the connectivity remains the same
    connection_order = [[], [], [], []]
    for s, s_loc in enumerate(syndrome_loc):
        for d, d_loc in enumerate(data_loc):
            relative_offset = tuple(a-b for a,b in zip(d_loc, s_loc))
            even = (d_loc[0] + d_loc[1]) % 2 == 0
            if relative_offset == (0.5, 0.5): # SW if even else NE
                connection_order[0].append((s,d))
            elif relative_offset == (0.5, -0.5): # SE
                connection_order[1 if even else 2].append((s,d))
            elif relative_offset == (-0.5, 0.5): # NW
                connection_order[2 if even else 1].append((s,d))
            elif relative_offset == (-0.5, -0.5): # NE if even else SW
                connection_order[3].append((s,d))
            else:
                continue
    stabilizers, logicals = transform_code_stabilizers(code, deformation)
    return stabilizers, logicals, syndrome_loc, data_loc, connection_order

def build_qiskit_circuit(code: RotatedPlanarCode, deformation: jnp.ndarray, rounds: int, with_idling: bool = True, basis: str = '+Z', return_metadata: bool = False) -> QuantumCircuit:
    """
    Builds a Qiskit quantum circuit for the given code and number of rounds

    Args:
        code (RotatedPlanarCode): The quantum error correcting code
        deformation (jnp.ndarray): The deformation applied to the code (An array of shape (num_data_qubits,) with entries in {0,1,2,3,4,5} representing the index of the Clifford deformation on each data qubit)
        rounds (int): The number of error correction rounds
        with_idling (bool, optional): Whether to include idling gates to represent wait times. Defaults to True.
        basis (str, optional): The logical basis for initializing and measurement of the logical qubit. Can be '+Z' (|0>), '-Z' (|1>), '+X' (|+>), and '-X' (|->). Defaults to '+Z'.
        return_metadata (bool, optional): Whether to return additional metadata about the circuit in the form of a dictionary. Defaults to False.
    
    Returns:
        QuantumCircuit: The constructed quantum circuit or a tuple (QuantumCircuit, metadata) if return_metadata is True
    """
    stabilizers, logicals, syndrome_loc, data_loc, connection_order = prepare_circuit_info(code, deformation)

    stab = [AncillaRegister(1, f'synd @ ({x}, {y})') for x, y in syndrome_loc]
    data = [QuantumRegister(1, f"data @ ({x}, {y})") for x, y in data_loc]
    meas = [ClassicalRegister(len(syndrome_loc), f'$Round_{i}$') for i in range(rounds)]

    def create_qec_round():
        data = QuantumRegister(len(data_loc), "data")
        stab = AncillaRegister(len(syndrome_loc), "syndrome")
        meas = ClassicalRegister(len(syndrome_loc), "measure")
        qec_round = QuantumCircuit(data, stab, meas, name="QEC_Round")

        # Prepare syndrome qubits
        qec_round.reset(stab)
        if with_idling:
            qec_round.append(idle_reset, [data])

        qec_round.barrier()

        qec_round.h(stab)
        if with_idling:
            qec_round.append(idle_1q, [data])

        qec_round.barrier()

        # Apply stabilizers
        for direction in connection_order:
            idling_qubits = set([*data, *stab])
            for s, d in direction:
                x, z = stabilizers[s, d], stabilizers[s, d + len(data_loc)]
                match (x, z):
                    case (1, 0):  # X
                        qec_round.cx(stab[s], data[d])
                    case (0, 1):  # Z
                        qec_round.cz(stab[s], data[d])
                    case (1, 1):  # Y
                        qec_round.cy(stab[s], data[d])
                    case _:
                        raise ValueError(f"Unexpected stabilizer entry {(x, z)}")
                idling_qubits.remove(data[d])
                idling_qubits.remove(stab[s])
            if with_idling:
                qec_round.append(idle_2q, [list(idling_qubits)])
            qec_round.barrier()
        
        # Measure syndrome qubits
        qec_round.h(stab)
        if with_idling:
            qec_round.append(idle_1q, [data])

        qec_round.barrier()

        qec_round.measure(stab, meas)
        if with_idling:
            qec_round.append(idle_meas, [data])
        
        qec_round.barrier()

        return qec_round

    def create_init_state():
        data = QuantumRegister(len(data_loc), "data")
        init_circ = QuantumCircuit(data, name="Init_State")

        # Prepare initial state
        init_circ.reset(data)
        match basis.upper():
            case '+Z':  # |0>
                pass
            case '-Z':  # |1>
                init_circ.x(data)
            case '+X':  # |+>
                init_circ.h(data)
            case '-X':  # |->
                init_circ.x(data)
                init_circ.h(data)
            case _:
                raise ValueError(f"Unexpected basis {basis}, supported bases are '+Z', '-Z', '+X', and '-X'.")
        
        init_circ.barrier()
        for i, defo in enumerate(deformation):
            match defo:
                case 0:  # I
                    pass
                case 1:  # X-Y
                    init_circ.s(data[i])
                case 2:  # Y-Z
                    init_circ.sxdg(data[i])
                case 3:  # X-Z
                    init_circ.h(data[i])
                case 4:  # X-Z-Y
                    init_circ.sx(data[i])
                    init_circ.h(data[i])
                case 5:  # X-Y-Z
                    init_circ.h(data[i])
                    init_circ.sxdg(data[i])
                case _:
                    raise ValueError(f"Unexpected deformation index {defo}")

        return init_circ

    circ = QuantumCircuit(*data, *stab, *meas)

    # Simplify registers for easier gate application
    stab = [s[0] for s in stab]
    data = [d[0] for d in data]

    init_circ = create_init_state()
    qec_round = create_qec_round()

    circ.append(init_circ.to_instruction(), [*data])

    circ.barrier()

    for i in range(rounds):
        circ.append(qec_round.to_instruction(), [*data, *stab], [*meas[i]])
        # Logical expectation values
        prefactor = -1 if basis[0] == '-' else 1
        pauli_x_string = ''.join(["IXZY"[lx+2*lz] for lx, lz in zip(*logicals[0].reshape(2,-1))])
        pauli_z_string = ''.join(["IXZY"[lx+2*lz] for lx, lz in zip(*logicals[1].reshape(2,-1))])
        # Qiskit uses little-endian ordering for Pauli strings, so we need to reverse them
        logical_x = Pauli(pauli_x_string[::-1])
        logical_z = Pauli(pauli_z_string[::-1])
        if 'X' in basis.upper():
            observable = prefactor * logical_x
        elif 'Z' in basis.upper():
            observable = prefactor * logical_z
        else:
            raise ValueError(f"Unexpected basis {basis}")
        observable_exp = SaveExpectationValue(observable, label=f'observable_R{i}', pershot=True)
        circ.append(observable_exp, data)
        circ.barrier()

    final_circuit = circ.decompose(gates_to_decompose=["QEC_Round", "Init_State"])

    if not return_metadata:
        return final_circuit
    return final_circuit, {
        "stabilizers": stabilizers,
        "logicals": logicals,
        "syndrome_loc": syndrome_loc,
        "data_loc": data_loc,
        "connection_order": connection_order,
        "logical_X": logical_x,
        "logical_Z": logical_z,
        "logical_Y": 1j*logical_x & logical_z,
        "observable": observable
    }

def build_stim_circuit(code: RotatedPlanarCode, deformation: jnp.ndarray, rounds: int, basis: str = '+Z', return_metadata: bool = False) -> stim.Circuit | tuple[stim.Circuit, dict]:
    """
    Builds a Stim quantum circuit for the given code

    Args:
        code (RotatedPlanarCode): The quantum error correcting code
        deformation (jnp.ndarray): The deformation applied to the code (An array of shape (num_data_qubits,) with entries in {0,1,2,3,4,5} representing the index of the Clifford deformation on each data qubit)
        rounds (int): The number of QEC rounds to perform
        basis (str, optional): The basis in which to measure the logical qubit. Defaults to '+Z'.
        return_metadata (bool, optional): Whether to return metadata about the circuit. Defaults to False.
    
    Returns:
        stim.Circuit: The constructed Stim quantum circuit
    """
    stabilizers, logicals, syndrome_loc, data_loc, connection_order = prepare_circuit_info(code, deformation)

    all_qubits = np.arange(len(syndrome_loc) + len(data_loc))
    data = all_qubits[:len(data_loc)]
    stab = all_qubits[len(data_loc):]

    def create_qec_round():
        qec_round = stim.Circuit()
        stab_order = [
            stim.Circuit(),  # SW
            stim.Circuit(),  # SE
            stim.Circuit(),  # NW
            stim.Circuit(),  # NE
        ]

        for i, connections in enumerate(connection_order):
            for s, d in connections:
                x, z = stabilizers[s, d], stabilizers[s, d + len(data_loc)]
                match (x, z):
                    case (1, 0):  # X
                        stab_order[i].append_operation("CX", [stab[s], data[d]])
                    case (0, 1):  # Z
                        stab_order[i].append_operation("CZ", [stab[s], data[d]])
                    case (1, 1):  # Y
                        stab_order[i].append_operation("CY", [stab[s], data[d]])
                    case _:
                        raise ValueError(f"Unexpected stabilizer entry {(x, z)}")

        qec_round.append("R", stab)
        qec_round.append("TICK", tag="1q")
        qec_round.append("H", stab)
        qec_round.append("TICK", tag="2q")
        qec_round.append(stab_order[0])
        qec_round.append("TICK", tag="2q")
        qec_round.append(stab_order[1])
        qec_round.append("TICK", tag="2q")
        qec_round.append(stab_order[2])
        qec_round.append("TICK", tag="2q")
        qec_round.append(stab_order[3])
        qec_round.append("TICK", tag="1q")
        qec_round.append("H", stab)
        qec_round.append("TICK", tag="meas")
        qec_round.append("M", stab)
        qec_round.append("TICK", tag="reset")

        return qec_round

    def create_init_state():
        init_circ = stim.Circuit()

        # Prepare initial state
        init_circ.append("R", data)

        init_circ.append("TICK", tag="1q")
        match basis.upper():
            case '+Z':  # |0>
                pass
            case '-Z':  # |1>
                init_circ.append("X", data)
            case '+X':  # |+>
                init_circ.append("H", data)
            case '-X':  # |->
                init_circ.append("X", data)
                init_circ.append("H", data)
            case _:
                raise ValueError(f"Unexpected basis {basis}")

        init_circ.append("TICK", tag="1q")
        for i, defo in enumerate(deformation):
            match defo:
                case 0:  # I
                    pass
                case 1:  # X-Y
                    init_circ.append("H_XY", data[i])
                case 2:  # Y-Z
                    init_circ.append("H_YZ", data[i])
                case 3:  # X-Z
                    init_circ.append("H_XZ", data[i])
                case 4:  # X-Z-Y
                    init_circ.append("C_ZYX", data[i])
                case 5:  # X-Y-Z
                    init_circ.append("C_XYZ", data[i])
                case _:
                    raise ValueError(f"Unexpected deformation index {defo}")

        return init_circ

    def measure_observable():
        observable_circ = stim.Circuit()
        
        obs = logicals[0] if basis.upper()[1] == 'X' else logicals[1]
        actives = 0
        observable_circ.append("TICK", tag="meas")
        for i, elm in enumerate(obs.reshape(2,-1).T):
            match elm.tolist():
                case [1, 0]:  # X
                    actives += 1
                    observable_circ.append("MX", data[i])
                case [0, 1]:  # Z
                    actives += 1
                    observable_circ.append("MZ", data[i])
                case [1, 1]:  # Y
                    actives += 1
                    observable_circ.append("MY", data[i])
                case [0, 0]:  # I
                    pass
                case _:
                    raise ValueError(f"Unexpected logical entry {tuple(elm)}")
        observable_circ.append("OBSERVABLE_INCLUDE", [stim.target_rec(-(1+i)) for i in range(actives)], 0)

        return observable_circ

    layout = stim.Circuit()
    for i, cord in enumerate(syndrome_loc):
        layout.append_operation("QUBIT_COORDS", [stab[i]], cord)
    for i, cord in enumerate(data_loc):
        layout.append_operation("QUBIT_COORDS", [data[i]], cord)

    init_circ = create_init_state()
    qec_round = create_qec_round()
    obs_circ = measure_observable()

    round_1_detectors = stim.Circuit()
    for i in range(len(stab)):
        if (i < (len(stab) // 2) and basis.upper()[1] == 'X') or (i >= (len(stab) // 2) and basis.upper()[1] == 'Z'):
            round_1_detectors.append("DETECTOR", [stim.target_rec(-1 - i)])
    
    round_detectors = stim.Circuit()
    for i in range(len(stab)):
        round_detectors.append("DETECTOR", [stim.target_rec(-1 - i), stim.target_rec(-1 - i - len(stab))])

    final_circuit = layout + init_circ + (qec_round + round_1_detectors) + (qec_round + round_detectors) * (rounds-1) + obs_circ

    if not return_metadata:
        return final_circuit

    pauli_x_string = np.array(["IXZY"[lx+2*lz] for lx, lz in zip(*logicals[0].reshape(2,-1))]).reshape(code.size).T
    pauli_z_string = np.array(["IXZY"[lx+2*lz] for lx, lz in zip(*logicals[1].reshape(2,-1))]).reshape(code.size).T
    pauli_y_string = np.array(["IXZY"[lx+2*lz] for lx, lz in zip(*(logicals.sum(axis=0) % 2).reshape(2,-1))]).reshape(code.size).T
    return final_circuit, {
        "stabilizers": stabilizers,
        "logicals": logicals,
        "syndrome_loc": syndrome_loc,
        "data_loc": data_loc,
        "connection_order": connection_order,
        "logical_X": pauli_x_string,
        "logical_Z": pauli_z_string,
        "logical_Y": pauli_y_string,
        "observable": pauli_x_string if basis.upper()[1] == 'X' else pauli_z_string
    }
