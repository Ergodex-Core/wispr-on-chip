"""Process-node presets.

All numbers are first-order public-domain estimates meant to be overridden;
they are chosen so that a single-die plan reproduces published hardwired-model
chips (see ``docs/ARCHITECTURE.md``) to within a small factor.

Densities are *usable* densities after routing, power grid and array overheads,
not marketing peak densities. Energies are per operation at nominal voltage.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Process:
    name: str
    logic_density_mtr_mm2: float
    """Usable logic transistors per mm² (millions)."""
    sram_density_mbit_mm2: float
    """Usable SRAM density (Mbit / mm²) including periphery."""
    weight_cell_density_mbit_mm2: float
    """Density of via/metal-programmed constant cells (Mbit / mm²). Denser than SRAM: no
    cross-coupled latch, one access device per bit, programmed in the via masks."""
    mac_energy_pj_int8: float
    """Energy of one 8x8-bit multiply-accumulate (pJ). Scaled by bit product for other widths."""
    weight_read_energy_pj_bit: float
    """Energy to read one hardwired weight bit from the local constant array (pJ)."""
    sram_read_energy_pj_bit: float
    """Energy to read one bit from a tile-local SRAM (pJ)."""
    hop_energy_pj_bit: float
    """Energy to move one bit across one tile-to-tile link (pJ)."""
    leakage_mw_mm2: float
    """Static power density at nominal voltage and temperature (mW / mm²)."""
    clock_ghz: float
    """Nominal fabric clock."""
    activity_overhead: float = 2.0
    """Multiplier on datapath energy for clock tree, control, pipeline registers and glue."""
    reticle_stitch_supported: bool = True

    def with_(self, **changes) -> "Process":
        return replace(self, **changes)


PROCESSES: dict[str, Process] = {
    "n7": Process(
        name="n7",
        logic_density_mtr_mm2=60.0,
        sram_density_mbit_mm2=22.0,
        weight_cell_density_mbit_mm2=60.0,
        mac_energy_pj_int8=0.24,
        weight_read_energy_pj_bit=0.006,
        sram_read_energy_pj_bit=0.03,
        hop_energy_pj_bit=0.05,
        leakage_mw_mm2=45.0,
        clock_ghz=0.9,
    ),
    "n6": Process(
        name="n6",
        logic_density_mtr_mm2=75.0,
        sram_density_mbit_mm2=27.0,
        weight_cell_density_mbit_mm2=80.0,
        mac_energy_pj_int8=0.20,
        weight_read_energy_pj_bit=0.005,
        sram_read_energy_pj_bit=0.027,
        hop_energy_pj_bit=0.045,
        leakage_mw_mm2=45.0,
        clock_ghz=1.0,
    ),
    "n5": Process(
        name="n5",
        logic_density_mtr_mm2=90.0,
        sram_density_mbit_mm2=32.0,
        weight_cell_density_mbit_mm2=110.0,
        mac_energy_pj_int8=0.15,
        weight_read_energy_pj_bit=0.004,
        sram_read_energy_pj_bit=0.022,
        hop_energy_pj_bit=0.04,
        leakage_mw_mm2=50.0,
        clock_ghz=1.1,
    ),
    "n3": Process(
        name="n3",
        logic_density_mtr_mm2=130.0,
        sram_density_mbit_mm2=34.0,
        weight_cell_density_mbit_mm2=140.0,
        mac_energy_pj_int8=0.11,
        weight_read_energy_pj_bit=0.003,
        sram_read_energy_pj_bit=0.02,
        hop_energy_pj_bit=0.035,
        leakage_mw_mm2=55.0,
        clock_ghz=1.2,
    ),
}


def get_process(name: str) -> Process:
    key = name.lower()
    if key not in PROCESSES:
        raise KeyError(f"unknown process {name!r}; known: {', '.join(PROCESSES)}")
    return PROCESSES[key]
