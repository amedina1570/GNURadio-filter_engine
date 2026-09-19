"""Generate HDL and Vivado coefficient files from a quantized design.

Everything here consumes a :class:`~filter_engine.core.quantize.QuantizedFilter`
rather than a floating-point design, because there is no such thing as a
floating-point tap in an FPGA.  Quantize first, look at what it did to the
stopband, and only then export.

Four outputs:

``coe``      Xilinx coefficient file for the Vivado FIR Compiler.
``mif``      Memory initialisation file, for a coefficient ROM.
``verilog``  A synthesisable transposed-form FIR plus its coefficient ROM.
``vhdl``     The same, as a package and entity.
"""

from __future__ import annotations

import math

import numpy as np

from ..core.quantize import QuantizedFilter
from ._common import format_ints, spec_comment, timestamp

__all__ = ["generate_coe", "generate_mif", "generate_verilog", "generate_vhdl"]


def _identifier(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw).strip("_").lower()
    if not cleaned:
        cleaned = "fir_filter"
    if cleaned[0].isdigit():
        cleaned = f"f_{cleaned}"
    return cleaned


def _twos_complement_bits(value: int, width: int) -> str:
    """Two's-complement binary string of ``value`` in ``width`` bits."""
    return format(int(value) & ((1 << width) - 1), f"0{width}b")


def _twos_complement_hex(value: int, width: int) -> str:
    digits = (width + 3) // 4
    return format(int(value) & ((1 << width) - 1), f"0{digits}X")


def _require_fir(q: QuantizedFilter, what: str) -> np.ndarray:
    if q.int_sos is not None:
        raise ValueError(
            f"{what} describes a single FIR tap set, but this is an IIR "
            "design. Export the biquad coefficients with the Verilog or VHDL "
            "generator, which emits the cascade."
        )
    return np.asarray(q.int_taps, dtype=np.int64)


# --------------------------------------------------------------------------
# Vivado coefficient files
# --------------------------------------------------------------------------
def generate_coe(q: QuantizedFilter, radix: int = 10) -> str:
    """Xilinx ``.coe`` file for the Vivado FIR Compiler.

    The FIR Compiler takes *integer* coefficients and applies the binary
    point itself, so the ``Q`` format is recorded in the header comment for
    whoever sets that up.
    """
    taps = _require_fir(q, "A .coe file")
    if radix not in (2, 10, 16):
        raise ValueError("radix must be 2, 10 or 16")

    if radix == 10:
        values = [str(int(v)) for v in taps]
    elif radix == 16:
        values = [_twos_complement_hex(v, q.fmt.total_bits) for v in taps]
    else:
        values = [_twos_complement_bits(v, q.fmt.total_bits) for v in taps]

    header = spec_comment(q.design, prefix="; ")
    body = ",\n ".join(values)
    return "\n".join(
        [
            header,
            ";",
            f"; Coefficient format: {q.fmt} -- the stored values are integers;",
            f"; divide by 2^{q.fmt.frac_bits} to recover the real tap values.",
            f"; Quantization error floor: {q.error_floor_db:.1f} dB.",
            ";",
            f"radix = {radix};",
            f"coefdata =",
            f" {body};",
            "",
        ]
    )


def generate_mif(q: QuantizedFilter, radix: int = 2) -> str:
    """Memory initialisation file: one coefficient word per line."""
    taps = _require_fir(q, "A .mif file")
    width = q.fmt.total_bits
    if radix == 2:
        values = [_twos_complement_bits(v, width) for v in taps]
    elif radix == 16:
        values = [_twos_complement_hex(v, width) for v in taps]
    else:
        raise ValueError("radix must be 2 or 16")
    return "\n".join(values) + "\n"


# --------------------------------------------------------------------------
# Verilog
# --------------------------------------------------------------------------
def generate_verilog(
    q: QuantizedFilter, module_name: str | None = None, data_bits: int = 16
) -> str:
    """A synthesisable transposed-form FIR, with its coefficients inlined."""
    if q.int_sos is not None:
        return _verilog_biquads(q, module_name)

    taps = np.asarray(q.int_taps, dtype=np.int64)
    name = _identifier(module_name or q.design.spec.name)
    n = taps.size
    coeff_w = q.fmt.total_bits
    growth = math.ceil(math.log2(max(n, 2)))
    acc_w = data_bits + coeff_w + growth

    coeff_lines = [
        f"        coeff[{i}] = {coeff_w}'sd{v};"
        if v >= 0
        else f"        coeff[{i}] = -{coeff_w}'sd{abs(int(v))};"
        for i, v in enumerate(taps)
    ]

    return "\n".join(
        [
            "// " + "-" * 70,
            spec_comment(q.design, prefix="// "),
            "//",
            f"// Coefficient format: {q.fmt}",
            f"// Quantization error floor: {q.error_floor_db:.1f} dB",
            "//",
            "// Transposed direct form: the adder chain is broken by a register",
            "// at every tap, so the critical path is one multiply plus one add",
            "// regardless of length. That is what lets a long filter still",
            "// close timing on an Artix-7.",
            "//",
            f"// {'Taps are symmetric -- a folded form would halve the multiplier count.' if q.symmetric else 'Taps are not symmetric; no folding is possible.'}",
            "// " + "-" * 70,
            "",
            "`default_nettype none",
            "",
            f"module {name} #(",
            f"    parameter integer DATA_W  = {data_bits},",
            f"    parameter integer COEF_W  = {coeff_w},",
            f"    parameter integer NTAPS   = {n},",
            f"    parameter integer ACC_W   = {acc_w},",
            f"    parameter integer COEF_FRAC = {q.fmt.frac_bits}",
            ") (",
            "    input  wire                      clk,",
            "    input  wire                      rst_n,",
            "    input  wire                      in_valid,",
            "    input  wire signed [DATA_W-1:0]  in_data,",
            "    output wire                      out_valid,",
            "    output wire signed [DATA_W-1:0]  out_data,",
            "    output wire signed [ACC_W-1:0]   out_full",
            ");",
            "",
            "    // Coefficient ROM. Vivado infers a ROM or distributed LUTs",
            "    // from this initial block.",
            "    reg signed [COEF_W-1:0] coeff [0:NTAPS-1];",
            "    initial begin",
            *coeff_lines,
            "    end",
            "",
            "    reg signed [ACC_W-1:0] acc [0:NTAPS-1];",
            "",
            f"    // Output valid once the chain has filled ({n - 1} cycles).",
            "    reg [NTAPS-1:0] valid_sr;",
            "",
            "    integer i;",
            "    always @(posedge clk) begin",
            "        if (!rst_n) begin",
            "            for (i = 0; i < NTAPS; i = i + 1)",
            "                acc[i] <= {ACC_W{1'b0}};",
            "            valid_sr <= {NTAPS{1'b0}};",
            "        end else if (in_valid) begin",
            "            // Last stage starts a fresh partial sum; every other",
            "            // stage adds its product to the one behind it.",
            "            acc[NTAPS-1] <= in_data * coeff[NTAPS-1];",
            "            for (i = NTAPS-2; i >= 0; i = i - 1)",
            "                acc[i] <= acc[i+1] + in_data * coeff[i];",
            "            valid_sr <= {valid_sr[NTAPS-2:0], 1'b1};",
            "        end",
            "    end",
            "",
            "    assign out_full  = acc[0];",
            "    assign out_valid = valid_sr[NTAPS-1];",
            "",
            "    // Rescale to the input word width: shift the binary point back",
            "    // by COEF_FRAC, rounding to nearest rather than truncating so",
            "    // the filter does not acquire a DC offset.",
            "    wire signed [ACC_W-1:0] rounded =",
            "        (acc[0] + (1 <<< (COEF_FRAC-1))) >>> COEF_FRAC;",
            "",
            "    // Saturate instead of wrapping: a wrapped overflow turns a loud",
            "    // sample into a full-scale sample of the opposite sign, which",
            "    // sounds and looks far worse than clipping.",
            "    localparam signed [DATA_W-1:0] MAXV = {1'b0, {(DATA_W-1){1'b1}}};",
            "    localparam signed [DATA_W-1:0] MINV = {1'b1, {(DATA_W-1){1'b0}}};",
            "    assign out_data =",
            "        (rounded > $signed({{(ACC_W-DATA_W){1'b0}}, MAXV})) ? MAXV :",
            "        (rounded < $signed({{(ACC_W-DATA_W){1'b1}}, MINV})) ? MINV :",
            "        rounded[DATA_W-1:0];",
            "",
            "endmodule",
            "",
            "`default_nettype wire",
            "",
        ]
    )


def _verilog_biquads(q: QuantizedFilter, module_name: str | None) -> str:
    """Coefficient tables for an IIR cascade.

    No filter body is emitted: a fixed-point IIR needs per-section scaling
    and overflow handling decided against the real signal levels, and a
    generated guess at that would be worse than none.
    """
    assert q.int_sos is not None
    name = _identifier(module_name or q.design.spec.name)
    w = q.fmt.total_bits
    lines = [
        "// " + "-" * 70,
        spec_comment(q.design, prefix="// "),
        "//",
        f"// Coefficient format: {q.fmt}",
        f"// Stable after rounding: {'yes' if q.stable else 'NO'}",
        "//",
        "// Biquad coefficients only. A fixed-point IIR needs per-section",
        "// input scaling and overflow handling chosen against your actual",
        "// signal levels -- the feedback path means a single overflow can",
        "// start a limit cycle that never decays. Implement the cascade",
        "// section by section, in this order, and simulate it against the",
        "// generated Python before trusting it.",
        "// " + "-" * 70,
        "",
        f"localparam integer {name.upper()}_SECTIONS = {q.int_sos.shape[0]};",
        f"localparam integer {name.upper()}_COEF_W   = {w};",
        f"localparam integer {name.upper()}_COEF_FRAC = {q.fmt.frac_bits};",
        "",
    ]
    for i, section in enumerate(q.int_sos):
        b0, b1, b2, a0, a1, a2 = (int(v) for v in section)
        lines.extend(
            [
                f"// Section {i}: b = [{b0}, {b1}, {b2}], a = [{a0}, {a1}, {a2}]",
                f"localparam signed [{w-1}:0] {name.upper()}_S{i}_B0 = {w}'sd{b0};"
                if b0 >= 0
                else f"localparam signed [{w-1}:0] {name.upper()}_S{i}_B0 = -{w}'sd{abs(b0)};",
            ]
        )
        for label, value in (("B1", b1), ("B2", b2), ("A1", a1), ("A2", a2)):
            literal = (
                f"{w}'sd{value}" if value >= 0 else f"-{w}'sd{abs(value)}"
            )
            lines.append(
                f"localparam signed [{w-1}:0] {name.upper()}_S{i}_{label} = {literal};"
            )
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# VHDL
# --------------------------------------------------------------------------
def generate_vhdl(
    q: QuantizedFilter, entity_name: str | None = None, data_bits: int = 16
) -> str:
    """A VHDL package of coefficients plus a transposed-form FIR entity."""
    name = _identifier(entity_name or q.design.spec.name)
    coeff_w = q.fmt.total_bits

    if q.int_sos is not None:
        values = [int(v) for v in np.asarray(q.int_sos).ravel()]
        table = ",\n        ".join(str(v) for v in values)
        return "\n".join(
            [
                spec_comment(q.design, prefix="-- "),
                "--",
                f"-- Coefficient format: {q.fmt}",
                f"-- Stable after rounding: {'yes' if q.stable else 'NO'}",
                "--",
                "-- Biquad coefficients in row order [b0, b1, b2, a0, a1, a2].",
                "-- No filter body is generated: see the Verilog export for why",
                "-- a fixed-point IIR body has to be written against real signal",
                "-- levels rather than generated.",
                "",
                "library ieee;",
                "use ieee.std_logic_1164.all;",
                "use ieee.numeric_std.all;",
                "",
                f"package {name}_pkg is",
                f"    constant SECTIONS  : integer := {q.int_sos.shape[0]};",
                f"    constant COEF_W    : integer := {coeff_w};",
                f"    constant COEF_FRAC : integer := {q.fmt.frac_bits};",
                "    type coef_array is array (natural range <>) of integer;",
                f"    constant COEFFS : coef_array(0 to {len(values) - 1}) := (",
                f"        {table}",
                "    );",
                f"end package {name}_pkg;",
                "",
            ]
        )

    taps = np.asarray(q.int_taps, dtype=np.int64)
    n = taps.size
    growth = math.ceil(math.log2(max(n, 2)))
    acc_w = data_bits + coeff_w + growth
    table = ",\n        ".join(
        ", ".join(str(int(v)) for v in taps[i : i + 8]) for i in range(0, n, 8)
    )

    return "\n".join(
        [
            spec_comment(q.design, prefix="-- "),
            "--",
            f"-- Coefficient format: {q.fmt}",
            f"-- Quantization error floor: {q.error_floor_db:.1f} dB",
            "",
            "library ieee;",
            "use ieee.std_logic_1164.all;",
            "use ieee.numeric_std.all;",
            "",
            f"package {name}_pkg is",
            f"    constant NTAPS     : integer := {n};",
            f"    constant COEF_W    : integer := {coeff_w};",
            f"    constant DATA_W    : integer := {data_bits};",
            f"    constant ACC_W     : integer := {acc_w};",
            f"    constant COEF_FRAC : integer := {q.fmt.frac_bits};",
            "    type coef_array is array (natural range <>) of integer;",
            f"    constant COEFFS : coef_array(0 to NTAPS-1) := (",
            f"        {table}",
            "    );",
            f"end package {name}_pkg;",
            "",
            "",
            "library ieee;",
            "use ieee.std_logic_1164.all;",
            "use ieee.numeric_std.all;",
            "use work." + name + "_pkg.all;",
            "",
            "-- Transposed direct form: one register per tap keeps the critical",
            "-- path to a single multiply-add however long the filter gets.",
            f"entity {name} is",
            "    port (",
            "        clk       : in  std_logic;",
            "        rst_n     : in  std_logic;",
            "        in_valid  : in  std_logic;",
            "        in_data   : in  signed(DATA_W-1 downto 0);",
            "        out_valid : out std_logic;",
            "        out_data  : out signed(DATA_W-1 downto 0)",
            "    );",
            f"end entity {name};",
            "",
            f"architecture rtl of {name} is",
            "    type acc_array is array (0 to NTAPS-1) of signed(ACC_W-1 downto 0);",
            "    signal acc      : acc_array := (others => (others => '0'));",
            "    signal valid_sr : std_logic_vector(NTAPS-1 downto 0) := (others => '0');",
            "begin",
            "",
            "    process (clk)",
            "        variable prod : signed(DATA_W+COEF_W-1 downto 0);",
            "    begin",
            "        if rising_edge(clk) then",
            "            if rst_n = '0' then",
            "                acc      <= (others => (others => '0'));",
            "                valid_sr <= (others => '0');",
            "            elsif in_valid = '1' then",
            "                prod := in_data * to_signed(COEFFS(NTAPS-1), COEF_W);",
            "                acc(NTAPS-1) <= resize(prod, ACC_W);",
            "                for i in NTAPS-2 downto 0 loop",
            "                    prod := in_data * to_signed(COEFFS(i), COEF_W);",
            "                    acc(i) <= acc(i+1) + resize(prod, ACC_W);",
            "                end loop;",
            "                valid_sr <= valid_sr(NTAPS-2 downto 0) & '1';",
            "            end if;",
            "        end if;",
            "    end process;",
            "",
            "    out_valid <= valid_sr(NTAPS-1);",
            "",
            "    -- Round to nearest, then saturate rather than wrap.",
            "    process (acc)",
            "        variable rounded : signed(ACC_W-1 downto 0);",
            "        constant MAXV : signed(DATA_W-1 downto 0) :=",
            "            to_signed(2**(DATA_W-1) - 1, DATA_W);",
            "        constant MINV : signed(DATA_W-1 downto 0) :=",
            "            to_signed(-(2**(DATA_W-1)), DATA_W);",
            "    begin",
            "        rounded := shift_right(acc(0) + to_signed(2**(COEF_FRAC-1), ACC_W),",
            "                               COEF_FRAC);",
            "        if rounded > resize(MAXV, ACC_W) then",
            "            out_data <= MAXV;",
            "        elsif rounded < resize(MINV, ACC_W) then",
            "            out_data <= MINV;",
            "        else",
            "            out_data <= resize(rounded, DATA_W);",
            "        end if;",
            "    end process;",
            "",
            "end architecture rtl;",
            "",
        ]
    )
