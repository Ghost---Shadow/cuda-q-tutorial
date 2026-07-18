"""
A made-up display technology: each pixel is one electron, and its color is
whatever direction its spin is pointing on the Bloch sphere. This is the
same "qubit as a unit 3D vector" idea from the vector-calculus conversation,
just aimed at something concrete: pick a target RGB color for each of a 3x3
grid of pixels, and build the actual circuit that makes that pixel's
electron spin point the right way - verified exactly via get_state(), not
sampled.

The mapping: a Bloch vector's 3 components are ⟨X⟩, ⟨Y⟩, ⟨Z⟩, each ranging
over [-1, 1]. An RGB channel ranges over [0, 1]. So:
    (bx, by, bz) = (2R-1, 2G-1, 2B-1) / sqrt(3)
(the /sqrt(3) guarantees the result never leaves the unit ball, since the
RGB cube's corners are the worst case and (+-1,+-1,+-1)/sqrt(3) has length
exactly 1).

That last point is the fun part: the 8 corners of the RGB cube (red, green,
blue, yellow, cyan, magenta, black, white - every channel either 0 or 1)
land EXACTLY on the surface of the Bloch sphere - genuine PURE states, full
spin polarization, no missing information. Any color that isn't a corner
(anything with a channel that isn't fully on or off) needs a Bloch vector
with length < 1, which a pure state literally cannot represent - it needs a
MIXED state instead. The purest example: 50% gray (0.5, 0.5, 0.5) maps to
the exact center of the sphere, length 0 - the maximally mixed state. Gray
is, quite literally, the "I have no idea which way this spin points"
color - there's no spin information left to display.

Mixed states are built the same way schrodingers-style entanglement
demos build them elsewhere in this repo (see fun_stuff/entanglement_propagation.py):
entangle the pixel qubit with a throwaway ancilla, then never measure the
ancilla. Concretely, per pixel:
  1. An ancilla gets RY(theta_mix) so P(ancilla=1) = (1 - length) / 2.
  2. x.ctrl(ancilla, pixel) - conditionally flips the pixel qubit's *input*
     from |0> to |1> before the fixed prep rotation below.
  3. A fixed RY(theta_n) then RZ(phi_n) rotates the pixel qubit to point
     along the target direction n - applied to BOTH the |0> and |1>
     branches, which lands on ANTIPODAL points (since |0> and |1> are
     already antipodal, and a rigid rotation preserves that). So the
     ancilla being 0 or 1 controls "+n or -n", with probability set by
     step 1 - and once the ancilla is discarded (never measured), the
     pixel qubit's own marginal Bloch vector is a genuine probabilistic
     mixture of +n and -n, landing exactly at the target (possibly
     shorter-than-1) vector.

Verification is exact, not sampled: since the ancilla is never actually
measured mid-circuit, the full 18-qubit system stays a pure state the
whole time, and get_state() returns its exact statevector. Each pixel's
own reduced density matrix (via partial trace over every other qubit) is
computed classically in numpy, its <X>,<Y>,<Z> read off directly, and
converted back to RGB - checked against the original target color to
machine precision, then rendered to an actual PNG.
"""

import math

import numpy as np
import cudaq
from PIL import Image

GRID_WIDTH, GRID_HEIGHT = 3, 3
NUM_PIXELS = GRID_WIDTH * GRID_HEIGHT
SQRT3 = math.sqrt(3.0)

# Row-major, top-left to bottom-right. All 8 RGB-cube corners (pure states)
# plus one non-corner color (50% gray - the maximally mixed state).
PIXEL_COLORS = [
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),   # red, green, blue
    (1.0, 1.0, 0.0), (0.5, 0.5, 0.5), (0.0, 1.0, 1.0),   # yellow, gray, cyan
    (1.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),   # magenta, black, white
]

PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def rgb_to_bloch(r: float, g: float, b: float) -> tuple[float, float, float]:
    return (2 * r - 1) / SQRT3, (2 * g - 1) / SQRT3, (2 * b - 1) / SQRT3


def bloch_to_angles(bx: float, by: float, bz: float) -> tuple[float, float, float]:
    """Returns (theta_mix, theta_n, phi_n) - the 3 rotation angles the
    kernel needs to prepare this exact (possibly-mixed) Bloch vector."""
    length = min(math.sqrt(bx * bx + by * by + bz * bz), 1.0)
    if length < 1e-9:
        theta_n, phi_n = 0.0, 0.0  # direction is meaningless at the origin
    else:
        theta_n = math.acos(max(-1.0, min(1.0, bz / length)))
        phi_n = math.atan2(by, bx)
    theta_mix = 2 * math.asin(math.sqrt(max(0.0, (1 - length) / 2)))
    return theta_mix, theta_n, phi_n


@cudaq.kernel
def electron_monitor(theta_mix: list[float], theta_n: list[float], phi_n: list[float]):
    """One (ancilla, pixel) qubit pair per pixel. The ancilla is entangled
    with the pixel and then simply never measured - see module docstring
    for why that reproduces an arbitrary mixed-state Bloch vector."""
    q = cudaq.qvector(2 * NUM_PIXELS)
    for i in range(NUM_PIXELS):
        anc = q[2 * i]
        pix = q[2 * i + 1]
        ry(theta_mix[i], anc)
        x.ctrl(anc, pix)
        ry(theta_n[i], pix)
        rz(phi_n[i], pix)


def reduced_density_matrix(psi_tensor: np.ndarray, qubit_index: int, n_qubits: int) -> np.ndarray:
    """cudaq's statevector is little-endian (qubit 0 is the fastest-varying
    axis), so the tensor axis for a given qubit index is reversed relative
    to its allocation order."""
    axis = n_qubits - 1 - qubit_index
    flat = np.moveaxis(psi_tensor, axis, 0).reshape(2, -1)
    return flat @ flat.conj().T


def measure_pixel_rgb(rho: np.ndarray) -> tuple[float, float, float, float]:
    ex = float(np.real(np.trace(rho @ PAULI_X)))
    ey = float(np.real(np.trace(rho @ PAULI_Y)))
    ez = float(np.real(np.trace(rho @ PAULI_Z)))
    r, g, b = (ex * SQRT3 + 1) / 2, (ey * SQRT3 + 1) / 2, (ez * SQRT3 + 1) / 2
    purity_length = math.sqrt(ex ** 2 + ey ** 2 + ez ** 2)
    return r, g, b, purity_length


if __name__ == "__main__":
    cudaq.set_target("nvidia")

    theta_mix_list, theta_n_list, phi_n_list = [], [], []
    for (r, g, b) in PIXEL_COLORS:
        bloch = rgb_to_bloch(r, g, b)
        tm, tn, pn = bloch_to_angles(*bloch)
        theta_mix_list.append(tm)
        theta_n_list.append(tn)
        phi_n_list.append(pn)

    state = np.array(cudaq.get_state(electron_monitor, theta_mix_list, theta_n_list, phi_n_list))
    n_qubits = 2 * NUM_PIXELS
    psi = state.reshape([2] * n_qubits)

    print(f"{'pixel':>5} {'target RGB':>16} {'measured RGB':>22} {'|Bloch vector|':>15}  status")
    all_ok = True
    measured_colors = []
    for i, target in enumerate(PIXEL_COLORS):
        pixel_qubit_index = 2 * i + 1
        rho = reduced_density_matrix(psi, pixel_qubit_index, n_qubits)
        r, g, b, purity_length = measure_pixel_rgb(rho)
        measured_colors.append((r, g, b))

        ok = all(abs(a - t) < 1e-6 for a, t in zip((r, g, b), target))
        all_ok = all_ok and ok
        print(f"{i:>5} {str(target):>16} {f'({r:.4f},{g:.4f},{b:.4f})':>22} "
              f"{purity_length:>15.4f}  {'OK' if ok else 'MISMATCH'}")

    print()
    if all_ok:
        print("SUCCESS: every pixel's electron spin points exactly where its target")
        print("color demands - checked against the exact statevector, not shots.")
        print("Notice pixel 4 (50% gray) is the only one with |Bloch vector| = 0:")
        print("the one color that ISN'T an RGB-cube corner is the one that needed a")
        print("genuinely mixed state, not a pure spin direction, to represent it.")
    else:
        print("FAILURE: see mismatches above.")

    scale = 100
    img = Image.new("RGB", (GRID_WIDTH * scale, GRID_HEIGHT * scale))
    pixels = img.load()
    for i, (r, g, b) in enumerate(measured_colors):
        row, col = divmod(i, GRID_WIDTH)
        rgb_255 = (round(r * 255), round(g * 255), round(b * 255))
        for y in range(row * scale, (row + 1) * scale):
            for x_ in range(col * scale, (col + 1) * scale):
                pixels[x_, y] = rgb_255
    output_path = "fun_stuff/electron_spin_monitor_output.png"
    img.save(output_path)
    print(f"\nRendered {GRID_WIDTH}x{GRID_HEIGHT} grid to {output_path}")
