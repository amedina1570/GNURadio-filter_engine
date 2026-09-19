# Digital Filter Engine

Design digital filters, see exactly what they do, and export them as working
code — for **radar**, for software-defined radio (USRP B2xx), and for FPGA
targets (Artix-7).

Built on NumPy and SciPy. The GUI is modelled on GNU Radio's filter design
tool, with four things added that it does not do: **radar pulse compression**
with Taylor weighting and range-sidelobe measurement, **fixed-point analysis**
against a real FPGA target, **synthetic pulse injection**, and **code
generation to several targets** from one design.

Every parameter carries a plain-language explanation, and the panel restates
your settings as the quantities you actually think in — metres of resolution,
kilometres of unambiguous range, decibels of processing gain.

---

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

Or install it as a package and use the entry point:

```bash
pip install -e .[gui]
filter-engine
```

Requires Python 3.10+. The GUI runs on PySide6 (default) or PyQt6; set
`FILTER_ENGINE_QT_API=pyqt6` to force the latter. Everything except the GUI
works with no Qt binding at all.

---

## What it does

### Radar

Radar filtering is judged differently from communications filtering. Nobody
designing a pulse compression filter cares about passband ripple; they care
about **range sidelobes** — how far a strong target smears across neighbouring
range cells and buries a weak one.

| | |
|---|---|
| **Pulse compression** | Matched filter for a linear-FM chirp. Complex (I/Q) taps, because a real-tap filter cannot tell an up-sweep from its mirror image. |
| **MTI cancellers** | 2- to 5-pulse binomial cancellers, running in slow time at one sample per PRI. |
| **Weightings** | Taylor, Dolph–Chebyshev, Kaiser, Hamming, Hann, Blackman, Blackman–Harris, and none. |
| **Measured** | PSLR, ISLR, compressed mainlobe width, mainlobe broadening, weighting SNR loss, processing gain. |
| **Plots** | Compressed pulse with the sidelobe level marked, the weighting and its transform, the range–Doppler ambiguity surface, and MTI velocity response with blind speeds. |

**Taylor** is the one you reach for. You name the sidelobe level you want and
it gets there with less mainlobe broadening — less lost resolution — than any
fixed window achieving the same level:

```
weighting        PSLR      ISLR   resolution  broadening  SNR loss
boxcar          -13.3 dB   -9.7 dB     6.62 m      1.00x    0.00 dB
hann            -31.5 dB  -28.7 dB    10.77 m      1.63x    1.77 dB
hamming         -42.6 dB  -29.0 dB     9.74 m      1.47x    1.35 dB
taylor 25 dB    -25.3 dB  -18.8 dB     7.89 m      1.19x    0.43 dB
taylor 35 dB    -34.7 dB  -25.4 dB     8.86 m      1.34x    0.92 dB
taylor 45 dB    -44.1 dB  -29.3 dB     9.77 m      1.48x    1.36 dB
```

Lower sidelobes always cost mainlobe width and a little SNR. These figures are
measured, not quoted, and the test suite pins them to published values.

**Taylor's two parameters are coupled.** The number of flat sidelobes `nbar`
has to grow with the level you ask for — at `nbar=4` a 50 dB design quietly
delivers 44.7 dB. The tool derives `nbar` from the level by default, and says
so when an override falls short.

The **two-target** excitation is the question that decides whether a weighting
was worth applying: put a target 40 dB below a nearby strong one and see
whether it survives. Unweighted, the strong target's own sidelobe sits *above*
the weak one and the reading is meaningless. With Taylor at 45 dB the sidelobe
is 57 dB down and the target reads its true strength.

See [examples/radar_pulse_compression.py](examples/radar_pulse_compression.py)
for all of that as a runnable script.

### Design

| | |
|---|---|
| **FIR methods** | windowed sinc (`firwin`), equiripple Parks–McClellan (`remez`), weighted least squares (`firls`), frequency sampling (`firwin2`) |
| **IIR prototypes** | Butterworth, Chebyshev I, Chebyshev II, elliptic, Bessel |
| **Responses** | low pass, high pass, band pass, band stop, Hilbert transformer, differentiator |
| **Pulse shaping** | root raised cosine, raised cosine, Gaussian (GMSK) |
| **Windows** | Hamming, Hann, Blackman, Blackman–Harris, Bartlett, rectangular, Nuttall, flat top, Kaiser |

Order and tap count are estimated from the tolerances you ask for, and the
estimate is explained in the design notes rather than appearing as a bare
number.

### Analyse

Magnitude, phase, group delay, impulse response, step response, pole–zero
plot, and the coefficients themselves — with the requested mask shaded behind
the magnitude curve so a miss is obvious.

Every design is also **measured**: the achieved passband ripple, stopband
attenuation, −3 dB and −6 dB cutoffs and group delay are re-derived from the
realised coefficients, not repeated back from the request. If a design misses
its mask, the status bar says so.

### Inject a pulse

Fourteen excitations — impulse, step, rectangular, Gaussian, sinc,
raised-cosine, tone burst, chirp, 13-chip Barker, PRBS BPSK, two-tone, AWGN,
and the two radar ones: an **LFM pulse** and **two targets** at adjustable
separation and strength difference.

Timing is set by **PRI**, not PRF, because that is how radar timing is
reasoned about: the PRI is the listening window, and unambiguous range is
just that window times c/2. The panel shows the derived PRF, duty cycle and
unambiguous range beside it.

A non-zero carrier offset produces complex baseband (I/Q), which is what an
SDR front end actually delivers; a radar echo is always I/Q.

The filtered output is drawn shifted back by the filter's group delay, so
you are looking at distortion rather than latency. For radar excitations the
output switches to a **dB envelope** automatically — a target 40 dB below its
neighbour is one hundredth of the height on a linear axis, which hides the
very thing the filter exists to reveal.

### Check the fixed-point reality

A filter that is flawless in double precision can lose 30 dB of stopband at
12 bits, and an IIR can go unstable outright. Pick a word length and the tool
reports the stopband you can *actually* have, the quantization noise floor,
the coefficient SNR, and — for an IIR — whether the poles are still inside the
unit circle.

"Compare word lengths" sweeps 8/12/16/18/24/32 bits and tabulates the result.

Alongside that is an Artix-7 resource estimate: DSP48E1 slices after
symmetry folding and time-multiplexing, the full-precision accumulator width,
and whether the design fits the part you chose.

### Generate code

| Target | Output |
|---|---|
| **Python** | a standalone module: the taps, *the SciPy call that produced them*, and a streaming filter class |
| **GNU Radio** | a `gr.hier_block2` subclass, using `firdes` where the design maps onto it |
| **Xilinx `.coe`** | integer coefficients for the Vivado FIR Compiler |
| **`.mif`** | one coefficient word per line, for a coefficient ROM |
| **Verilog** | a synthesisable transposed-form FIR with rounding and saturation |
| **VHDL** | the same filter as a package and entity |

A complex (pulse compression) design generates complex taps throughout: the
Python module keeps them complex, GNU Radio switches to `fir_filter_ccc`, and
the Vivado and HDL exports carry both coefficient sets with a note on the four
real multiplies a complex tap costs.

Every generated file leads with the specification that produced it —
response, band edges, tolerances, window, the lot. Coefficients on their own
are unmaintainable; six months later nobody remembers whether 0.1 dB of ripple
was a requirement or a guess.

The generated Python carries both the stored taps and a `design_taps()` that
recomputes them. Run the file directly and it checks one against the other.
**109 tests do exactly that**, across every design path, by executing the
generated module as a subprocess.

The HDL is not compiled by the test suite — no simulator is assumed — but the
arithmetic it encodes is. A bit-exact Python model of the emitted transposed
datapath (same rounding, same saturation) is held against the filter it
implements, and the `.coe`/`.mif` files are decoded back to integers and
compared. **Synthesise and simulate the RTL before trusting it in hardware.**

---

## Using it as a library

The GUI is a front end over a plain Python API; nothing needs Qt.

```python
from filter_engine.core.spec import FilterSpec, FirMethod
from filter_engine.core.design import design
from filter_engine.core.analysis import measure
from filter_engine.core.quantize import quantize, estimate_fpga
from filter_engine import codegen

spec = FilterSpec(
    name="channel_filter",
    fir_method=FirMethod.REMEZ,
    sample_rate=10e6,        # 10 MS/s from a B210
    f_low=200e3,             # 200 kHz channel
    transition_width=50e3,
    passband_ripple_db=0.1,
    stopband_atten_db=80.0,
)

fd = design(spec)
print(fd.summary())
print(measure(fd).as_rows())

q = quantize(fd, total_bits=16)
print(f"16-bit gives {q.quantized_stopband_db:.1f} dB of stopband")
print(estimate_fpga(q, clock_hz=100e6, sample_rate=10e6).as_rows())

open("channel_filter.py", "w").write(codegen.generate("python", fd))
open("channel_filter.coe", "w").write(codegen.generate("coe", fd, q))
```

See [examples/design_channel_filter.py](examples/design_channel_filter.py)
for a runnable version.

---

## Layout

```
filter_engine/
  core/
    spec.py        FilterSpec: what you want, with strict validation
    design.py      spec -> coefficients, via scipy
    firdes.py      RRC/RC/Gaussian taps and order estimation
    radar.py       LFM waveforms, Taylor/Chebyshev weighting, radar units
    analysis.py    responses, radar metrics, ambiguity surface
    quantize.py    fixed point, and Artix-7 resource estimation
    signals.py     synthetic excitations and the filter's response
    explain.py     plain-language help for every response and parameter
  codegen/         Python, GNU Radio, .coe, .mif, Verilog, VHDL
  gui/             PySide6 application
  plotting.py      matplotlib figures, no Qt
  presets.py       USRP / Artix-7 limits and starting-point designs
tests/             core DSP, codegen round-trip, headless GUI
```

---

## Notes on hardware targets

**USRP B2xx.** Presets cover the B200, B210, B200mini and B205mini. Note that
*B206 is not an Ettus part number* — if that is what you were after, the B200
(1×1) or B210 (2×2) is almost certainly the board you mean. The tool warns
when a sample rate exceeds what the USB 3.0 link sustains (61.44 MS/s
single-channel) or the AD9361/AD9364's 56 MHz analog bandwidth.

**Artix-7.** Resource estimates assume the DSP48E1 slice these parts carry: a
signed 25×18 multiplier with a 48-bit accumulator. A coefficient wider than
18 bits needs two slices per multiply, which the estimate accounts for. Part
sizes cover the XC7A35T through XC7A200T.

Estimates are a sanity check before you open Vivado, not a substitute for
synthesis. For pulse compression in particular the tool will tell you when a
filter is too long for a direct FIR and wants fast convolution instead — a
20 us pulse at 40 MS/s is 801 complex taps, which is 3,204 real multiplies per
sample and nobody's direct-form filter.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest                          # everything (~4 min; the round-trip tests
                                # spawn a subprocess per case)
pytest tests/test_core.py       # DSP properties only, ~7 s
pytest tests/test_radar.py      # radar figures against published values
pytest tests/test_hdl.py        # fixed-point datapath and coefficient files
pytest tests/test_gui_smoke.py  # headless GUI
```

The GUI tests run under Qt's `offscreen` platform and need no display.

---

## Licence

Not yet chosen. `scipy` is BSD-licensed; PySide6 is LGPL.
