#!/usr/bin/env python3
"""End-to-end example: a 200 kHz channel filter for a USRP B210 at 10 MS/s.

Walks the whole pipeline without touching the GUI -- design, measure, inject
a pulse, quantize for an Artix-7, and export every code target. Run it to see
what the tool produces:

    python examples/design_channel_filter.py [output_directory]
"""

from __future__ import annotations

import os
import sys

# Make the package importable when running straight from a checkout.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from filter_engine import codegen
from filter_engine.core.analysis import measure
from filter_engine.core.design import design
from filter_engine.core.quantize import estimate_fpga, quantize
from filter_engine.core.signals import PulseKind, PulseSpec, apply_filter, generate
from filter_engine.core.spec import FilterSpec, FirMethod
from filter_engine.presets import get_fpga, get_sdr


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main(out_dir: str) -> int:
    radio = get_sdr("USRP B210")
    fpga = get_fpga("Artix-7 XC7A35T")

    # ---------------------------------------------------------------- design
    spec = FilterSpec(
        name="channel_filter",
        fir_method=FirMethod.REMEZ,   # equiripple: uniformly deep stopband
        sample_rate=10e6,             # well inside the B210's 61.44 MS/s
        f_low=200e3,                  # 200 kHz channel
        transition_width=50e3,
        passband_ripple_db=0.1,
        stopband_atten_db=80.0,
    )

    warning = radio.check_sample_rate(spec.sample_rate)
    if warning:
        print(f"Radio warning: {warning}")

    fd = design(spec)
    rule("Design")
    print(fd.summary())

    # --------------------------------------------------------------- measure
    rule("Measured from the realised coefficients")
    for label, value in measure(fd).as_rows():
        print(f"  {label:<24} {value}")

    # --------------------------------------------------------- pulse response
    rule("Pulse response")
    pulse = PulseSpec(
        kind=PulseKind.RECT,
        sample_rate=spec.sample_rate,
        duration_s=400e-6,
        width_s=20e-6,
        delay_s=100e-6,
        carrier_hz=150e3,   # inside the passband, so it should survive
        snr_db=30.0,
    )
    generated = generate(pulse)
    filtered = apply_filter(fd, generated.x)
    for note in generated.notes:
        print(f"  - {note}")
    print(
        f"  input peak {np.max(np.abs(generated.x)):.4f}  ->  "
        f"output peak {np.max(np.abs(filtered)):.4f}"
    )

    # ----------------------------------------------------------- fixed point
    rule(f"Fixed point, and what it costs on an {fpga.name}")
    print("  bits   stopband      noise floor   verdict")
    chosen = None
    for bits in (8, 12, 16, 18, 24):
        q = quantize(fd, total_bits=bits)
        if not q.usable:
            print(f"  {bits:<6} {'-':<13} {'-':<13} unusable")
            continue
        meets = q.quantized_stopband_db >= spec.stopband_atten_db
        print(
            f"  {bits:<6} {q.quantized_stopband_db:>6.1f} dB     "
            f"{q.error_floor_db:>7.1f} dB     "
            f"{'meets the spec' if meets else 'short of the spec'}"
        )
        if chosen is None and meets:
            chosen = q

    if chosen is None:
        chosen = quantize(fd, total_bits=18)
        print("\n  No offered word length met the spec; exporting at 18 bits.")

    rule(f"Resources at {chosen.fmt}")
    estimate = estimate_fpga(
        chosen, clock_hz=fpga.typical_clock_hz, sample_rate=spec.sample_rate
    )
    for label, value in estimate.as_rows():
        print(f"  {label:<30} {value}")
    # The notes carry the caveats that the numbers alone do not: a coefficient
    # too wide for the DSP48E1's port, an accumulator past 48 bits, and so on.
    for note in estimate.notes:
        print(f"  ! {note}")
    usage = fpga.check_usage(estimate.dsp_slices)
    print(f"  {usage or f'Fits {fpga.name} comfortably.'}")

    # --------------------------------------------------------------- export
    rule("Generated files")
    os.makedirs(out_dir, exist_ok=True)
    for target in codegen.TARGETS:
        if not target.supports(fd):
            print(f"  {target.label:<32} skipped (FIR only)")
            continue
        code = codegen.generate(target.key, fd, chosen)
        suffix = "_gr" if target.key == "gnuradio" else ""
        path = os.path.join(out_dir, f"{spec.name}{suffix}{target.extension}")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(code)
        print(f"  {target.label:<32} {os.path.basename(path)} "
              f"({len(code.splitlines()):,} lines)")

    print(f"\nWritten to {os.path.abspath(out_dir)}")
    print("Run the generated Python directly to verify its coefficients:")
    print(f"    python {os.path.join(out_dir, spec.name + '.py')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "out"))
