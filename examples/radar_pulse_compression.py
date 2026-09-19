#!/usr/bin/env python3
"""Pulse compression, weighting, and why Taylor is the radar default.

Designs an X-band matched filter for a chirped pulse, compares what each
weighting costs and buys, then puts two targets of very different strength in
front of it to show what the sidelobe level actually decides.

    python examples/radar_pulse_compression.py [output_directory]
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from filter_engine import codegen
from filter_engine.core import signals
from filter_engine.core.analysis import radar_metrics
from filter_engine.core.design import design
from filter_engine.core.quantize import estimate_fpga, quantize
from filter_engine.core.spec import FilterSpec, Response

# An X-band radar: 20 us pulse swept over 20 MHz, sampled at 2x the chirp.
SAMPLE_RATE = 40e6
PULSE_WIDTH = 20e-6
BANDWIDTH = 20e6
PRI = 1e-3
CARRIER = 10e9


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def base_spec(**changes) -> FilterSpec:
    return FilterSpec(
        name="pulse_compression",
        response=Response.MATCHED_LFM,
        sample_rate=SAMPLE_RATE,
        pulse_width_s=PULSE_WIDTH,
        chirp_bandwidth_hz=BANDWIDTH,
        pri_s=PRI,
        radar_carrier_hz=CARRIER,
        window="taylor",
        taylor_sll_db=35.0,
    ).copy(**changes)


def main(out_dir: str) -> int:
    spec = base_spec()

    # ------------------------------------------------------------- the radar
    rule("What this waveform gives you")
    print(f"  Pulse width          {PULSE_WIDTH * 1e6:,.6g} us")
    print(f"  Chirp bandwidth      {BANDWIDTH / 1e6:,.6g} MHz")
    print(f"  Range resolution     {spec.range_resolution_m:,.4g} m "
          "(bandwidth alone sets this)")
    print(f"  Compression ratio    {spec.time_bandwidth_product:,.0f}:1  ->  "
          f"{10 * np.log10(spec.time_bandwidth_product):.1f} dB processing gain")
    print(f"  PRI                  {PRI * 1e6:,.6g} us "
          f"({spec.prf_hz:,.6g} Hz PRF)")
    print(f"  Unambiguous range    {spec.unambiguous_range_m / 1e3:,.4g} km")
    print(f"  Duty cycle           {spec.duty_cycle * 100:.2f}%")
    print(f"  First blind speed    {spec.blind_speed_ms:,.4g} m/s "
          f"at {CARRIER / 1e9:.0f} GHz")

    # -------------------------------------------------------- the trade-off
    rule("What each weighting costs and buys")
    print("  weighting        PSLR      ISLR   resolution  broadening  SNR loss")
    for window, sll in (
        ("boxcar", None),
        ("hann", None),
        ("hamming", None),
        ("taylor", 25.0),
        ("taylor", 35.0),
        ("taylor", 45.0),
        ("chebwin", None),
    ):
        candidate = base_spec(window=window)
        if sll is not None:
            candidate = candidate.copy(taylor_sll_db=sll)
        metrics = radar_metrics(design(candidate))
        label = f"{window} {sll:.0f}dB" if sll else window
        print(
            f"  {label:<14} {metrics.pslr_db:7.1f} dB {metrics.islr_db:7.1f} dB"
            f" {metrics.mainlobe_3db_m:8.2f} m {metrics.broadening:9.2f}x"
            f" {metrics.weighting_loss_db:8.2f} dB"
        )
    print("\n  Lower sidelobes always cost mainlobe width and a little SNR.")
    print("  Taylor lets you name the level; the rest give you what they give.")

    # ------------------------------------------------- does it actually help
    rule("Two targets, 40 dB apart, 22 range cells apart")
    pulse = signals.PulseSpec(
        kind=signals.PulseKind.TWO_TARGETS,
        sample_rate=SAMPLE_RATE,
        duration_s=60e-6,
        width_s=PULSE_WIDTH,
        delay_s=15e-6,
        lfm_bandwidth_hz=BANDWIDTH,
        target_separation_s=1.1e-6,
        target2_relative_db=-40.0,
    )
    both = signals.generate(pulse)
    alone = signals.generate(
        signals.PulseSpec(
            **{**pulse.__dict__, "kind": signals.PulseKind.LFM_PULSE}
        )
    )
    offset = int(round(pulse.target_separation_s * SAMPLE_RATE))

    def level_at_target(x: np.ndarray, fd) -> float:
        y = np.abs(signals.apply_filter(fd, x))
        peak = int(np.argmax(y))
        return 20 * np.log10(max(y[peak + offset] / y[peak], 1e-12))

    print("  weighting     sidelobe there   reading with target   verdict")
    for window, sll in (("boxcar", None), ("hamming", None), ("taylor", 45.0)):
        candidate = base_spec(window=window)
        if sll is not None:
            candidate = candidate.copy(taylor_sll_db=sll)
        fd = design(candidate)
        floor = level_at_target(alone.x, fd)
        reading = level_at_target(both.x, fd)
        clear = floor < -46.0
        label = f"{window} {sll:.0f}dB" if sll else window
        print(
            f"  {label:<13} {floor:9.1f} dB {reading:16.1f} dB   "
            + ("target is clear" if clear else "buried in sidelobes")
        )
    print("\n  The true target is -40 dB. Unweighted, the strong target's own")
    print("  sidelobe sits above it and the reading is meaningless.")

    # ------------------------------------------------------- the MTI filter
    rule("MTI clutter canceller at the same PRF")
    for pulses in (2, 3):
        mti = design(
            FilterSpec(
                name=f"mti_{pulses}",
                response=Response.MTI_CANCELLER,
                sample_rate=1.0 / PRI,
                pri_s=PRI,
                mti_pulses=pulses,
                radar_carrier_hz=CARRIER,
            )
        )
        taps = ", ".join(f"{v:+.3f}" for v in mti.b)
        print(f"  {pulses}-pulse: [{taps}]  notch order {pulses - 1}")
    print(f"  Blind speeds every {spec.blind_speed_ms:,.4g} m/s -- a target at")
    print("  one of those is cancelled along with the clutter.")

    # ------------------------------------------------------------ hardware
    rule("Fixed point and hardware")
    fd = design(spec)
    q = quantize(fd, 16)
    print(f"  {q.fmt}: error floor {q.error_floor_db:.1f} dB, "
          f"complex taps: {q.is_complex}")
    estimate = estimate_fpga(q, clock_hz=200e6, sample_rate=SAMPLE_RATE)
    print(f"  {fd.num_taps:,} complex taps -> "
          f"{estimate.effective_multipliers:,} real multiplies")
    for note in estimate.notes:
        print(f"  ! {note}")

    # ------------------------------------------------------------- export
    rule("Generated files")
    os.makedirs(out_dir, exist_ok=True)
    for target in codegen.TARGETS:
        if not target.supports(fd):
            print(f"  {target.label:<32} skipped (FIR only)")
            continue
        code = codegen.generate(target.key, fd, q)
        suffix = "_gr" if target.key == "gnuradio" else ""
        path = os.path.join(out_dir, f"{spec.name}{suffix}{target.extension}")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(code)
        print(f"  {target.label:<32} {os.path.basename(path)} "
              f"({len(code.splitlines()):,} lines)")

    print(f"\nWritten to {os.path.abspath(out_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "out_radar"))
