"""
Task I: Quantum Computing Part
1) 5-qubit circuit with Cirq
2) SWAP test circuit with PennyLane
"""

# ============================================================
# Part 1: Cirq — 5-qubit circuit
# ============================================================
import cirq
import numpy as np
import matplotlib.pyplot as plt

# (a) 5 qubits
qubits = cirq.LineQubit.range(5)

circuit = cirq.Circuit()

# (b) Hadamard on every qubit
circuit.append(cirq.H.on_each(*qubits))

# (c) CNOT on (0,1), (1,2), (2,3), (3,4)
circuit.append([
    cirq.CNOT(qubits[0], qubits[1]),
    cirq.CNOT(qubits[1], qubits[2]),
    cirq.CNOT(qubits[2], qubits[3]),
    cirq.CNOT(qubits[3], qubits[4]),
])

# (d) SWAP(0, 4)
circuit.append(cirq.SWAP(qubits[0], qubits[4]))

# (e) Rotate X with pi/2 on qubit 2
circuit.append(cirq.rx(np.pi / 2)(qubits[2]))

# (f) Measure all
circuit.append(cirq.measure(*qubits, key='result'))

print("=" * 60)
print("Part 1: Cirq 5-qubit Circuit")
print("=" * 60)
print(circuit)

# Plot the circuit
fig, ax = plt.subplots(1, 1, figsize=(14, 4))
cirq_diagram = circuit.to_text_diagram(transpose=False)
ax.text(0.02, 0.5, cirq_diagram, transform=ax.transAxes,
        fontsize=10, verticalalignment='center', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax.set_axis_off()
ax.set_title("Part 1: 5-Qubit Circuit (Cirq)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("task1/part1_cirq_circuit.png", dpi=150, bbox_inches='tight')
print("Saved: task1/part1_cirq_circuit.png")

# Simulate
simulator = cirq.Simulator()
result = simulator.simulate(circuit[:-1])  # exclude measurement for state vector
print(f"\nFinal state vector (first 8 amplitudes):")
print(np.round(result.final_state_vector[:8], 4))

# ============================================================
# Part 2: PennyLane — SWAP Test Circuit
# ============================================================
import pennylane as qml

# 5 qubits: q0=ancilla, q1=first qubit, q2=second qubit,
#            q3=third qubit, q4=fourth qubit
# State 1: |q1 q2> prepared on wires 1,2
# State 2: |q3 q4> prepared on wires 3,4

dev = qml.device("default.qubit", wires=5)

@qml.qnode(dev)
def swap_test_circuit():
    # Prepare |q1>: Hadamard on first qubit (wire 1)
    qml.Hadamard(wires=1)

    # Prepare |q2>: Rotate second qubit by pi/3 around X (wire 2)
    qml.RX(np.pi / 3, wires=2)

    # Prepare |q3>: Hadamard on third qubit (wire 3)
    qml.Hadamard(wires=3)

    # Prepare |q4>: Hadamard on fourth qubit (wire 4)
    qml.Hadamard(wires=4)

    # --- SWAP Test ---
    # The SWAP test compares |q1 q2> with |q3 q4>
    # Ancilla qubit = wire 0

    # Step 1: Hadamard on ancilla
    qml.Hadamard(wires=0)

    # Step 2: Controlled-SWAP between corresponding qubits
    # CSWAP(ancilla, q1, q3) and CSWAP(ancilla, q2, q4)
    qml.CSWAP(wires=[0, 1, 3])
    qml.CSWAP(wires=[0, 2, 4])

    # Step 3: Hadamard on ancilla
    qml.Hadamard(wires=0)

    # Measure ancilla: P(|0>) = (1 + |<psi|phi>|^2) / 2
    return qml.probs(wires=0)

print("\n" + "=" * 60)
print("Part 2: PennyLane SWAP Test Circuit")
print("=" * 60)

# Draw the circuit
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 6))
drawer = qml.draw(swap_test_circuit)
circuit_str = drawer()
print(circuit_str)

ax2.text(0.02, 0.5, circuit_str, transform=ax2.transAxes,
         fontsize=9, verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
ax2.set_axis_off()
ax2.set_title("Part 2: SWAP Test Circuit (PennyLane)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("task1/part2_pennylane_swap_test.png", dpi=150, bbox_inches='tight')
print("Saved: task1/part2_pennylane_swap_test.png")

# Run the circuit
probs = swap_test_circuit()
print(f"\nAncilla measurement probabilities:")
print(f"  P(|0>) = {probs[0]:.6f}")
print(f"  P(|1>) = {probs[1]:.6f}")

# Fidelity interpretation
# P(|0>) = (1 + |<psi|phi>|^2) / 2
# => |<psi|phi>|^2 = 2 * P(|0>) - 1
overlap_sq = 2 * probs[0].item() - 1
print(f"\n|<ψ|φ>|² = {overlap_sq:.6f}")
print(f"|<ψ|φ>|  = {np.sqrt(max(overlap_sq, 0)):.6f}")

print("\nInterpretation:")
print("  |ψ> = H|0> ⊗ Rx(π/3)|0> = |+> ⊗ (cos(π/6)|0> + i·sin(π/6)|1>)")
print("  |φ> = H|0> ⊗ H|0>       = |+> ⊗ |+>")
print(f"  The overlap² = {overlap_sq:.6f} shows partial similarity between the two states.")
