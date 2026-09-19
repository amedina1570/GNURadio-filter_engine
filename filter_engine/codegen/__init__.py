"""Code generation targets.

Each target turns a design into text.  :data:`TARGETS` is the registry the GUI
builds its export menu from, so adding a generator here makes it appear in the
application without touching the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..core.design import FilterDesign
from ..core.quantize import QuantizedFilter
from . import gnuradio, hdl, python_scipy

__all__ = ["Target", "TARGETS", "generate", "target_names"]


@dataclass(frozen=True)
class Target:
    """One code-generation target."""

    key: str
    label: str
    #: Suggested file extension, including the dot.
    extension: str
    #: Language name for the GUI's syntax highlighting and for humans.
    language: str
    #: True when the generator needs quantized (fixed-point) coefficients.
    needs_quantized: bool
    description: str
    _fn: Callable[..., str]
    #: True when the format can only describe an FIR tap set. The GUI uses
    #: this to disable the target for an IIR design rather than letting the
    #: user hit an error.
    fir_only: bool = False

    def __call__(self, design_or_quantized, **kwargs) -> str:
        return self._fn(design_or_quantized, **kwargs)

    def supports(self, design: FilterDesign) -> bool:
        return design.is_fir or not self.fir_only


TARGETS: tuple[Target, ...] = (
    Target(
        key="python",
        label="Python (NumPy / SciPy)",
        extension=".py",
        language="python",
        needs_quantized=False,
        description=(
            "A standalone module with the taps, the scipy call that produced "
            "them, and a streaming filter class."
        ),
        _fn=python_scipy.generate,
    ),
    Target(
        key="gnuradio",
        label="GNU Radio (hier_block2)",
        extension=".py",
        language="python",
        needs_quantized=False,
        description=(
            "A hierarchical block for a flowgraph, using firdes where the "
            "design maps onto it."
        ),
        _fn=gnuradio.generate,
    ),
    Target(
        key="coe",
        label="Xilinx .coe (FIR Compiler)",
        extension=".coe",
        language="text",
        needs_quantized=True,
        description="Integer coefficients for the Vivado FIR Compiler IP.",
        _fn=hdl.generate_coe,
        fir_only=True,
    ),
    Target(
        key="mif",
        label="Memory init file (.mif)",
        extension=".mif",
        language="text",
        needs_quantized=True,
        description="One coefficient word per line, for a coefficient ROM.",
        _fn=hdl.generate_mif,
        fir_only=True,
    ),
    Target(
        key="verilog",
        label="Verilog (transposed FIR)",
        extension=".v",
        language="verilog",
        needs_quantized=True,
        description=(
            "A synthesisable transposed-form FIR with rounding and saturation."
        ),
        _fn=hdl.generate_verilog,
    ),
    Target(
        key="vhdl",
        label="VHDL (package + entity)",
        extension=".vhd",
        language="vhdl",
        needs_quantized=True,
        description="The same filter as a VHDL package and entity.",
        _fn=hdl.generate_vhdl,
    ),
)

_BY_KEY = {t.key: t for t in TARGETS}


def target_names() -> list[str]:
    return [t.key for t in TARGETS]


def get_target(key: str) -> Target:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(
            f"unknown target {key!r}; known: {target_names()}"
        ) from None


def generate(
    key: str,
    design: FilterDesign,
    quantized: QuantizedFilter | None = None,
    **kwargs,
) -> str:
    """Run one target.

    Fixed-point targets need ``quantized``; floating-point ones ignore it.
    """
    target = get_target(key)
    if target.needs_quantized:
        if quantized is None:
            raise ValueError(
                f"the {target.label} target needs quantized coefficients; "
                "quantize the design first"
            )
        return target(quantized, **kwargs)
    return target(design, **kwargs)
