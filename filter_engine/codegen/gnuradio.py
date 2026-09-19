"""Generate GNU Radio Python for a design.

The output is a :class:`gr.hier_block2` subclass -- a block you can drop into
a flowgraph or import from GRC, rather than a throwaway script.

Where the design maps onto a ``firdes`` call the generated code makes that
call, so the filter stays editable in GNU Radio's own vocabulary.  Where it
does not (``remez``, ``firls`` and frequency sampling have no firdes
equivalent) the taps are emitted as a literal list.  Both routes produce the
same filter; only one of them can be retuned by editing a cutoff.

Targets GNU Radio 3.10, and notes where 3.8 differs.
"""

from __future__ import annotations

from ..core.design import FilterDesign
from ..core.spec import FilterFamily, FirMethod, Response
from ._common import format_floats, spec_comment

__all__ = ["generate"]

#: firdes window constants, by our window name.  GNU Radio 3.9 moved these
#: from ``firdes.WIN_*`` to ``fft.window.WIN_*``; 3.10 keeps the latter.
_GR_WINDOWS = {
    "hamming": "window.WIN_HAMMING",
    "hann": "window.WIN_HANN",
    "blackman": "window.WIN_BLACKMAN",
    "blackmanharris": "window.WIN_BLACKMAN_HARRIS",
    "bartlett": "window.WIN_BARTLETT",
    "boxcar": "window.WIN_RECTANGULAR",
    "kaiser": "window.WIN_KAISER",
    "nuttall": "window.WIN_NUTTALL",
    "flattop": "window.WIN_FLATTOP",
}


def generate(
    fd: FilterDesign, class_name: str | None = None, complex_io: bool = True
) -> str:
    """Return GNU Radio Python implementing ``fd``.

    ``complex_io`` selects the block type: complex in / complex out with real
    taps (``fir_filter_ccf``), which is what an I/Q stream needs, versus the
    all-float ``fir_filter_fff``.
    """
    spec = fd.spec
    name = class_name or _class_name(spec.name)

    lines = [
        "#!/usr/bin/env python3",
        "# -*- coding: utf-8 -*-",
        '"""',
        spec_comment(fd, prefix=""),
        "",
        "Written for GNU Radio 3.10.",
        "On 3.8, replace `from gnuradio.fft import window` with",
        "`from gnuradio.filter import firdes` and use firdes.WIN_* constants.",
        '"""',
        "",
        "from gnuradio import gr",
        "from gnuradio import filter as gr_filter",
        "from gnuradio.filter import firdes",
        "from gnuradio.fft import window",
        "",
        "",
        f"SAMPLE_RATE = {spec.sample_rate!r}  # Hz",
        "",
    ]

    if fd.is_fir:
        lines.extend(_fir_taps_section(fd))
        lines.extend(_fir_block(fd, name, complex_io))
    else:
        lines.extend(_iir_section(fd, name))

    lines.extend(_demo_block(fd, name, complex_io))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
def _class_name(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_")
    return cleaned.lower() or "designed_filter"


def _fir_taps_section(fd: FilterDesign) -> list[str]:
    firdes_call = _firdes_call(fd)
    lines: list[str] = []

    if firdes_call is not None:
        lines.extend(
            [
                "def make_taps():",
                '    """Build the taps with GNU Radio\'s own firdes.',
                "",
                "    Edit the arguments here to retune the filter; the block",
                "    below picks the change up automatically.",
                '    """',
                *[f"    {line}" for line in firdes_call],
                "",
                "",
                f"#: The {fd.num_taps} taps this design produced, for reference.",
                "#: firdes should reproduce them; see the __main__ check below.",
            ]
        )
    else:
        method = fd.spec.method
        lines.extend(
            [
                f"# GNU Radio's firdes has no {method} design, so the taps are",
                "# emitted directly. To retune this filter, change the",
                "# specification in the filter engine and regenerate.",
            ]
        )

    lines.extend(
        [
            "TAPS = [",
            format_floats(fd.b),
            "]",
            "",
        ]
    )
    if firdes_call is None:
        lines.extend(["", "def make_taps():", "    return TAPS", ""])
    lines.append("")
    return lines


def _firdes_call(fd: FilterDesign) -> list[str] | None:
    """The firdes call matching this design, or ``None`` if there is none."""
    s = fd.spec
    fs = s.sample_rate
    gain = s.gain

    if s.response is Response.RRC:
        return [
            "return firdes.root_raised_cosine(",
            f"    {gain!r}, {fs!r}, {s.symbol_rate!r}, {s.rolloff!r}, {fd.num_taps},",
            ")",
        ]
    if s.response is Response.GAUSSIAN:
        return [
            "return firdes.gaussian(",
            f"    {gain!r}, {s.samples_per_symbol!r}, {s.bt!r}, {fd.num_taps},",
            ")",
        ]
    if s.response is Response.RC:
        return None  # firdes has no raised-cosine, only the root
    if s.response is Response.DIFFERENTIATOR:
        return None
    if s.response is Response.HILBERT:
        return [
            "return firdes.hilbert(",
            f"    {fd.num_taps}, {_GR_WINDOWS.get(s.window, 'window.WIN_HAMMING')},"
            f" {s.window_param!r},",
            ")",
        ]

    if s.family is FilterFamily.IIR:
        return None
    if s.fir_method is not FirMethod.WINDOW:
        # firdes is windowed-sinc only; remez/firls/firwin2 have no analogue.
        return None

    win = _GR_WINDOWS.get(s.window, "window.WIN_HAMMING")
    beta = s.window_param
    common = f"{gain!r}, {fs!r}"
    tw = s.transition_width

    if s.response is Response.LOWPASS:
        return [
            "return firdes.low_pass(",
            f"    {common}, {s.f_low!r}, {tw!r}, {win}, {beta!r},",
            ")",
        ]
    if s.response is Response.HIGHPASS:
        return [
            "return firdes.high_pass(",
            f"    {common}, {s.f_low!r}, {tw!r}, {win}, {beta!r},",
            ")",
        ]
    if s.response is Response.BANDPASS:
        return [
            "return firdes.band_pass(",
            f"    {common}, {s.f_low!r}, {s.f_high!r}, {tw!r}, {win}, {beta!r},",
            ")",
        ]
    if s.response is Response.BANDSTOP:
        return [
            "return firdes.band_reject(",
            f"    {common}, {s.f_low!r}, {s.f_high!r}, {tw!r}, {win}, {beta!r},",
            ")",
        ]
    return None


def _fir_block(fd: FilterDesign, name: str, complex_io: bool) -> list[str]:
    io_type = "gr.sizeof_gr_complex" if complex_io else "gr.sizeof_float"
    block = "fir_filter_ccf" if complex_io else "fir_filter_fff"
    stream = "complex" if complex_io else "float"

    return [
        f"class {name}(gr.hier_block2):",
        f'    """{fd.spec.response.value} FIR, {fd.num_taps} taps, '
        f'{stream} in/out.',
        "",
        f"    Group delay is {(fd.num_taps - 1) / 2.0:g} samples; account for it",
        "    when aligning this path against another.",
        '    """',
        "",
        "    def __init__(self, decimation=1, taps=None):",
        "        gr.hier_block2.__init__(",
        f'            self, "{name}",',
        f"            gr.io_signature(1, 1, {io_type}),",
        f"            gr.io_signature(1, 1, {io_type}),",
        "        )",
        "        self.taps = make_taps() if taps is None else taps",
        "        self.decimation = decimation",
        f"        self.filt = gr_filter.{block}(decimation, self.taps)",
        "        self.connect((self, 0), (self.filt, 0))",
        "        self.connect((self.filt, 0), (self, 0))",
        "",
        "    def set_taps(self, taps):",
        '        """Swap the taps at runtime."""',
        "        self.taps = taps",
        "        self.filt.set_taps(taps)",
        "",
        "    def get_taps(self):",
        "        return self.taps",
        "",
    ]


def _iir_section(fd: FilterDesign, name: str) -> list[str]:
    """GNU Radio IIR blocks take feed-forward and feedback taps, not sections.

    ``iir_filter_ffd`` is a single direct-form stage, so a high-order design
    has to be built as a cascade of biquads -- which is what you want anyway.
    """
    assert fd.sos is not None
    rows = []
    for section in fd.sos:
        b = ", ".join(repr(float(v)) for v in section[:3])
        a = ", ".join(repr(float(v)) for v in section[3:])
        rows.append(f"    ([{b}], [{a}]),")

    return [
        f"#: {fd.num_sections} biquad sections as (feed-forward, feedback) tap pairs.",
        "#:",
        "#: GNU Radio's iir_filter_ffd is one direct-form stage, so the cascade",
        f"#: is built explicitly below. Do not flatten these into one order-{fd.order}",
        "#: stage: the expanded polynomial is numerically unstable.",
        "SECTIONS = [",
        *rows,
        "]",
        "",
        "",
        f"class {name}(gr.hier_block2):",
        f'    """{fd.spec.response.value} IIR, order {fd.order}, '
        f'as {fd.num_sections} cascaded biquads."""',
        "",
        "    def __init__(self):",
        "        gr.hier_block2.__init__(",
        f'            self, "{name}",',
        "            gr.io_signature(1, 1, gr.sizeof_float),",
        "            gr.io_signature(1, 1, gr.sizeof_float),",
        "        )",
        "        self.stages = [",
        "            gr_filter.iir_filter_ffd(list(ff), list(fb), False)",
        "            for ff, fb in SECTIONS",
        "        ]",
        "        self.connect((self, 0), (self.stages[0], 0))",
        "        for first, second in zip(self.stages, self.stages[1:]):",
        "            self.connect((first, 0), (second, 0))",
        "        self.connect((self.stages[-1], 0), (self, 0))",
        "",
    ]


def _demo_block(fd: FilterDesign, name: str, complex_io: bool) -> list[str]:
    lines = [
        "",
        'if __name__ == "__main__":',
        "    # Check the block builds, and compare the firdes taps against the",
        "    # ones this filter was designed with.",
    ]
    if fd.is_fir:
        lines.extend(
            [
                "    taps = make_taps()",
                "    print(f\"{len(taps)} taps\")",
                "    if len(taps) == len(TAPS):",
                "        worst = max(abs(a - b) for a, b in zip(taps, TAPS))",
                '        print(f"max difference from the stored taps: {worst:.3g}")',
                "    else:",
                "        print(",
                '            f"firdes returned {len(taps)} taps but this design "',
                '            f"used {len(TAPS)}; GNU Radio picks its own length "',
                '            "from the transition width."',
                "        )",
            ]
        )
    else:
        lines.append(f'    print(f"{{len(SECTIONS)}} biquad sections")')
    lines.extend(
        [
            f"    block = {name}()",
            '    print(f"built {block.name()} OK")',
        ]
    )
    return lines
