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
from ._common import spec_comment

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


def _require_fir(
    q: QuantizedFilter, what: str, component: str | None = None
) -> np.ndarray:
    """Integer taps for a coefficient file, choosing an I/Q component.

    A complex filter has two coefficient sets. ``component`` picks one; with
    ``None`` they are concatenated, in-phase first, which is what the Vivado
    FIR Compiler expects when told there are two coefficient sets.
    """
    if q.int_sos is not None:
        raise ValueError(
            f"{what} describes a single FIR tap set, but this is an IIR "
            "design. Export the biquad coefficients with the Verilog or VHDL "
            "generator, which emits the cascade."
        )
    if not q.is_complex:
        return np.asarray(q.int_taps, dtype=np.int64)

    if component == "i":
        return q.int_taps_i
    if component == "q":
        return q.int_taps_q
    if component is None:
        return np.concatenate([q.int_taps_i, q.int_taps_q])
    raise ValueError(f"component must be 'i', 'q' or None, not {component!r}")


# --------------------------------------------------------------------------
# Vivado coefficient files
# --------------------------------------------------------------------------
def generate_coe(
    q: QuantizedFilter, radix: int = 10, component: str | None = None
) -> str:
    """Xilinx ``.coe`` file for the Vivado FIR Compiler.

    The FIR Compiler takes *integer* coefficients and applies the binary
    point itself, so the ``Q`` format is recorded in the header comment for
    whoever sets that up.

    A complex filter writes both coefficient sets, in-phase first.
    """
    taps = _require_fir(q, "A .coe file", component)
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
            *_complex_coe_notes(q, component),
            ";",
            f"radix = {radix};",
            "coefdata =",
            f" {body};",
            "",
        ]
    )


def _complex_coe_notes(q: QuantizedFilter, component: str | None) -> list[str]:
    """Header lines explaining how a complex coefficient file is laid out."""
    if not q.is_complex:
        return []
    if component is not None:
        return [
            ";",
            f"; COMPLEX FILTER: this is the {component.upper()} coefficient "
            "set only.",
        ]
    return [
        ";",
        f"; COMPLEX FILTER: two coefficient sets of {q.int_taps_i.size} taps",
        "; each, in-phase first and then quadrature. In the Vivado FIR",
        "; Compiler set 'Number of Coefficient Sets' to 2, or export the I and",
        "; Q sets separately and instantiate two filters.",
        "; Filtering a complex stream takes four such convolutions (three if",
        "; you use the Karatsuba identity to trade a multiply for two adds).",
    ]


def generate_mif(
    q: QuantizedFilter, radix: int = 2, component: str | None = None
) -> str:
    """Memory initialisation file: one coefficient word per line.

    A complex filter writes every in-phase word and then every quadrature
    word, so one ROM holds both halves back to back.
    """
    taps = _require_fir(q, "A .mif file", component)
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
    if q.is_complex:
        return _verilog_complex_taps(q, module_name)

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
            "    //",
            "    // ROUND_ADD is built at the accumulator's width on purpose. A",
            "    // bare `1 <<< (COEF_FRAC-1)` is a 32-bit signed literal, which",
            "    // overflows to negative at COEF_FRAC = 32 and silently turns",
            "    // rounding into a large subtraction.",
            "    localparam signed [ACC_W-1:0] ROUND_ADD = (COEF_FRAC == 0)",
            "        ? {ACC_W{1'b0}}",
            "        : ({{(ACC_W-1){1'b0}}, 1'b1} <<< (COEF_FRAC - 1));",
            "",
            "    wire signed [ACC_W-1:0] rounded = (acc[0] + ROUND_ADD) >>> COEF_FRAC;",
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


def _verilog_complex_taps(q: QuantizedFilter, module_name: str | None) -> str:
    """I and Q coefficient ROMs for a complex FIR, without a filter body.

    No body is generated on purpose. A complex FIR has a structural choice to
    make -- four real multiplies per tap, or three using the Karatsuba
    identity -- and pulse compression filters are usually long enough that
    neither belongs in fabric at all: fast convolution wins past a few hundred
    taps. Emitting one arbitrary structure would hide that decision rather
    than inform it.
    """
    name = _identifier(module_name or q.design.spec.name)
    upper = name.upper()
    width = q.fmt.total_bits
    taps_i, taps_q = q.int_taps_i, q.int_taps_q
    n = taps_i.size

    def rom(label: str, values: np.ndarray) -> list[str]:
        lines = [
            f"    reg signed [{upper}_COEF_W-1:0] {name}_coeff_{label} "
            f"[0:{upper}_NTAPS-1];",
            "    initial begin",
        ]
        for i, v in enumerate(values):
            literal = (
                f"{width}'sd{int(v)}" if v >= 0 else f"-{width}'sd{abs(int(v))}"
            )
            lines.append(f"        {name}_coeff_{label}[{i}] = {literal};")
        lines.extend(["    end", ""])
        return lines

    fft_size = 1 << max(int(max(n - 1, 1)).bit_length() + 1, 8)
    return "\n".join(
        [
            "// " + "-" * 70,
            spec_comment(q.design, prefix="// "),
            "//",
            f"// Coefficient format: {q.fmt}",
            f"// Quantization error floor: {q.error_floor_db:.1f} dB",
            "//",
            "// COMPLEX FILTER -- coefficient tables only, no filter body.",
            "//",
            "// Filtering a complex stream with complex taps is four real",
            "// convolutions:",
            "//     y_i = x_i * h_i - x_q * h_q",
            "//     y_q = x_i * h_q + x_q * h_i",
            "// Three multiplies are possible via the Karatsuba identity, at",
            "// the cost of extra adders.",
            "//",
            f"// At {n} taps, consider not doing this in fabric at all:",
            f"// overlap-save fast convolution with a {fft_size}-point FFT",
            "// costs far less, and is how pulse compression is normally",
            "// implemented.",
            "// " + "-" * 70,
            "",
            f"localparam integer {upper}_NTAPS     = {n};",
            f"localparam integer {upper}_COEF_W    = {width};",
            f"localparam integer {upper}_COEF_FRAC = {q.fmt.frac_bits};",
            "",
            "// In-phase coefficients",
            *rom("i", taps_i),
            "// Quadrature coefficients",
            *rom("q", taps_q),
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

    if q.is_complex:
        # Same reasoning as the Verilog export: emit both coefficient sets and
        # leave the four-multiply structure to the implementer.
        values_i = [int(v) for v in q.int_taps_i]
        values_q = [int(v) for v in q.int_taps_q]

        def table_of(values: list[int]) -> str:
            return ",\n        ".join(
                ", ".join(str(v) for v in values[i : i + 8])
                for i in range(0, len(values), 8)
            )

        return "\n".join(
            [
                spec_comment(q.design, prefix="-- "),
                "--",
                f"-- Coefficient format: {q.fmt}",
                f"-- Quantization error floor: {q.error_floor_db:.1f} dB",
                "--",
                "-- COMPLEX FILTER -- coefficient tables only. Filtering a",
                "-- complex stream takes four real convolutions:",
                "--     y_i = x_i*h_i - x_q*h_q",
                "--     y_q = x_i*h_q + x_q*h_i",
                "",
                "library ieee;",
                "use ieee.std_logic_1164.all;",
                "use ieee.numeric_std.all;",
                "",
                f"package {name}_pkg is",
                f"    constant NTAPS     : integer := {len(values_i)};",
                f"    constant COEF_W    : integer := {coeff_w};",
                f"    constant COEF_FRAC : integer := {q.fmt.frac_bits};",
                "    type coef_array is array (natural range <>) of integer;",
                "    constant COEFFS_I : coef_array(0 to NTAPS-1) := (",
                f"        {table_of(values_i)}",
                "    );",
                "    constant COEFFS_Q : coef_array(0 to NTAPS-1) := (",
                f"        {table_of(values_q)}",
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
            "    constant COEFFS : coef_array(0 to NTAPS-1) := (",
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
