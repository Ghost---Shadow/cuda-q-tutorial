"""
The idea: run electron_spin_monitor.py's direction backwards. Instead of
"here's an RF pulse setting, what color does it produce" (a forward
lookup), start from "here's the color I WANT this pixel to show - what RF
pulse setting produces it" (an inverse lookup) - and get the answer by
measurement collapse instead of by computing an inverse function.

The product framing: a real monitor's controller would have a calibration
table baked in from the factory (RF pulse setting -> resulting electron
spin direction -> resulting color), built once by characterizing the
hardware. Given a target image, a classical controller would need to
invert that table - worst case, an unsorted linear scan, O(N) lookups per
pixel. A quantum controller can instead put every candidate RF pulse
setting in superposition, phase-flip ("mark") whichever one is known to
produce the desired color, and let Grover's algorithm - the exact same
oracle+diffusion pattern as nvidia-cudaq-tutorials/08_grovers_algorithm.py,
just retargeted at 3-qubit RF pulse codes instead of 4-qubit list items -
amplify that one candidate's probability from 1/N to near-certainty in
O(sqrt(N)) rounds. Measuring then IS "sending a query to the monitor and
having the superposition collapse to the answer," exactly as described.

This demo's calibration table is deliberately simple: 8 candidate RF pulse
codes (3 qubits), each one wired directly to one of the 8 RGB-cube corner
colors from electron_spin_monitor.py (a pulse code's own 3 bits ARE that
color's R, G, B bits - no separate arbitrary table needed). A real
factory-calibrated table wouldn't be this clean, but the SEARCH mechanism
- mark a specific 3-qubit pattern, diffuse, repeat - doesn't care whether
the table happens to be pretty; it works the same either way.

Honest scope note: this candidate space only covers the 8 pure corner
colors (full spin polarization), not intermediate/mixed colors like the 3x3
demo's 50% gray - extending the search to also retrieve a "how much to mix"
parameter would need a larger candidate register and is left out here.

The full loop, one pixel at a time:
  1. Decide the target color for this pixel.
  2. Grover-search the 8 candidate RF pulse codes for the one whose
     calibrated color matches - that's the "measure the monitor, get back
     the RF pulses" step.
  3. Take the RF pulse code that came back, feed ITS calibrated color into
     electron_spin_monitor.py's actual spin-preparation circuit (imported,
     not reimplemented) - that's "send the RF pulses to the real monitor."
  4. Render the result and check it against the original target - proving
     the whole round trip (desired image -> retrieved codes -> driven
     hardware -> displayed image) reproduces the picture that was asked for.
"""

import math
from collections import Counter

import cudaq
from PIL import Image

from electron_spin_monitor import (
    GRID_WIDTH, GRID_HEIGHT, NUM_PIXELS,
    bloch_to_angles, electron_monitor, measure_pixel_rgb, reduced_density_matrix, rgb_to_bloch,
)

NUM_CANDIDATE_CODES = 8  # 3-qubit RF pulse codes: bits directly ARE (R, G, B)


def calibrated_color(code: int) -> tuple[float, float, float]:
    """The factory calibration table: RF pulse code -> resulting color.
    Here it's transparently code's own bits, but the search below doesn't
    rely on that - it only uses this function to know what to mark."""
    return (1.0 if code & 4 else 0.0, 1.0 if code & 2 else 0.0, 1.0 if code & 1 else 0.0)


def code_for_color(target: tuple[float, float, float]) -> int:
    for code in range(NUM_CANDIDATE_CODES):
        if calibrated_color(code) == target:
            return code
    raise ValueError(f"{target} isn't one of the 8 pure corner colors this demo's table covers")


@cudaq.kernel
def mark_code(qubits: cudaq.qview, bit2: bool, bit1: bool, bit0: bool):
    """Phase-flips exactly the one 3-qubit basis state matching
    (bit2, bit1, bit0) - temporarily X-flip whichever qubits should read 0
    in the target pattern, multi-controlled-Z, then flip back. Same
    pattern as 08_grovers_algorithm.py's mark_1001/mark_1111, just
    retargeted at a runtime-supplied pattern instead of a hardcoded one."""
    if not bit2:
        x(qubits[0])
    if not bit1:
        x(qubits[1])
    if not bit0:
        x(qubits[2])
    z.ctrl(qubits[0], qubits[1], qubits[2])
    if not bit0:
        x(qubits[2])
    if not bit1:
        x(qubits[1])
    if not bit2:
        x(qubits[0])


@cudaq.kernel
def diffusion3(qubits: cudaq.qview):
    """Grover diffusion for a 3-qubit register - identical structure to
    08_grovers_algorithm.py's diffusion(), just 3 qubits instead of 4."""
    h(qubits[0])
    h(qubits[1])
    h(qubits[2])
    x(qubits[0])
    x(qubits[1])
    x(qubits[2])
    z.ctrl(qubits[0], qubits[1], qubits[2])
    x(qubits[0])
    x(qubits[1])
    x(qubits[2])
    h(qubits[0])
    h(qubits[1])
    h(qubits[2])


@cudaq.kernel
def grover_pulse_search(bit2: bool, bit1: bool, bit0: bool, iterations: int) -> list[bool]:
    q = cudaq.qvector(3)
    h(q[0])
    h(q[1])
    h(q[2])
    for _ in range(iterations):
        mark_code(q, bit2, bit1, bit0)
        diffusion3(q)
    return mz(q)


def optimal_grover_iterations(total: int, marked: int) -> int:
    return math.floor((math.pi / 4) * math.sqrt(total / marked))


def retrieve_rf_pulse_code(target_color: tuple[float, float, float], shots: int = 2000) -> tuple[int, float]:
    """The "measure the monitor for a color, get back the RF pulses" step.
    Runs Grover's search for the one candidate code whose calibrated color
    is target_color, and returns (most-sampled code, its hit rate)."""
    target_code = code_for_color(target_color)
    bit2, bit1, bit0 = bool(target_code & 4), bool(target_code & 2), bool(target_code & 1)
    iterations = optimal_grover_iterations(NUM_CANDIDATE_CODES, 1)

    results = cudaq.run(grover_pulse_search, bit2, bit1, bit0, iterations, shots_count=shots)
    counts = Counter(
        (4 if r[0] else 0) + (2 if r[1] else 0) + (1 if r[2] else 0)
        for r in results
    )
    retrieved_code, hits = counts.most_common(1)[0]
    return retrieved_code, hits / shots


# 3x3 checkerboard test pattern - a classic display test image, built
# entirely from the 8 pure corner colors this demo's calibration table covers.
WHITE, BLACK = (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)
TARGET_PATTERN = [WHITE, BLACK, WHITE, BLACK, WHITE, BLACK, WHITE, BLACK, WHITE]


if __name__ == "__main__":
    cudaq.set_target("nvidia")
    cudaq.set_random_seed(1)

    print(f"Optimal Grover iterations for {NUM_CANDIDATE_CODES} candidate RF pulse codes: "
          f"{optimal_grover_iterations(NUM_CANDIDATE_CODES, 1)}\n")

    print("=== Step 1+2: 'measure the monitor' for each pixel's desired color ===")
    retrieved_codes = []
    all_ok = True
    for i, target in enumerate(TARGET_PATTERN):
        code, hit_rate = retrieve_rf_pulse_code(target)
        retrieved_codes.append(code)
        expected_code = code_for_color(target)
        ok = code == expected_code
        all_ok = all_ok and ok
        print(f"  pixel {i}: target color={target}  ->  retrieved RF pulse code={code:03b} "
              f"(Grover hit rate {hit_rate:.1%})  {'OK' if ok else 'WRONG CODE'}")

    print("\n=== Step 3+4: send the retrieved codes to the real spin-preparation circuit ===")
    theta_mix_list, theta_n_list, phi_n_list = [], [], []
    for code in retrieved_codes:
        r, g, b = calibrated_color(code)
        tm, tn, pn = bloch_to_angles(*rgb_to_bloch(r, g, b))
        theta_mix_list.append(tm)
        theta_n_list.append(tn)
        phi_n_list.append(pn)

    state = cudaq.get_state(electron_monitor, theta_mix_list, theta_n_list, phi_n_list)
    import numpy as np
    psi = np.array(state).reshape([2] * (2 * NUM_PIXELS))

    measured_colors = []
    for i, target in enumerate(TARGET_PATTERN):
        rho = reduced_density_matrix(psi, 2 * i + 1, 2 * NUM_PIXELS)
        r, g, b, _ = measure_pixel_rgb(rho)
        measured_colors.append((r, g, b))
        pixel_ok = all(abs(a - t) < 1e-6 for a, t in zip((r, g, b), target))
        all_ok = all_ok and pixel_ok
        print(f"  pixel {i}: driven color=({r:.4f},{g:.4f},{b:.4f})  target={target}  "
              f"{'OK' if pixel_ok else 'MISMATCH'}")

    print("\nSUCCESS: the checkerboard image never got typed in directly to the display" if all_ok
          else "\nFAILURE: see mismatches above")
    if all_ok:
        print("circuit - measuring for each desired color collapsed the superposition to")
        print("exactly the RF pulse codes needed, and those codes alone reproduced the")
        print("original picture when actually driven through the spin-preparation circuit.")

    scale = 100
    img = Image.new("RGB", (GRID_WIDTH * scale, GRID_HEIGHT * scale))
    pixels = img.load()
    for i, (r, g, b) in enumerate(measured_colors):
        row, col = divmod(i, GRID_WIDTH)
        rgb_255 = (round(r * 255), round(g * 255), round(b * 255))
        for y in range(row * scale, (row + 1) * scale):
            for x_ in range(col * scale, (col + 1) * scale):
                pixels[x_, y] = rgb_255
    output_path = "fun_stuff/quantum_monitor_controller_output.png"
    img.save(output_path)
    print(f"\nRendered final driven picture to {output_path}")
