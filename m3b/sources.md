# M3b: published memory-energy figures (opened and read, 2026-11-19)

Rule for this module: off-chip memory energy comes only from sources we opened ourselves. Each number below was read from the PDF, with its page and context. The PDFs and their extracted text are not included here (not ours to redistribute).

## 1. Horowitz, ISSCC 2014

M. Horowitz, "1.1 Computing's energy problem (and what we can do about it)", ISSCC 2014 Digest of Technical Papers, pp. 10-14. DOI 10.1109/ISSCC.2014.6757323. Copy read: gwern.net/doc/cs/hardware/2014-horowitz-2.pdf (5 pages).

**Figure 1.1.9, "Rough energy costs for various operations in 45nm 0.9V"** (page 5). The text layer of this figure uses a scrambled font, so we rendered the page to an image and read the table from it:

| Operation | Energy |
|---|---|
| Integer add, 8 bit | 0.03 pJ |
| Integer add, 32 bit | 0.1 pJ |
| Integer multiply, 8 bit | 0.2 pJ |
| Integer multiply, 32 bit | 3.1 pJ |
| FP add, 16 / 32 bit | 0.4 / 0.9 pJ |
| FP multiply, 16 / 32 bit | 1.1 / 3.7 pJ |
| Cache read, 64 bit: 8 KB / 32 KB / 1 MB | 10 / 20 / 100 pJ |
| DRAM, 64 bit | 1.3-2.6 nJ |

Same figure: instruction energy breakdown of 70 pJ, with I-cache access 25 pJ and register file access 6 pJ.

**Body text (page 2, right column)**:
- A DRAM access costs "1 to 2nJ", a couple of orders of magnitude above a cache access or operation (10 pJ).
- DRAM I/O "takes over 20pJ/bit".
- Even with better I/O, a DRAM access stays large: "(10pJ/bit, 0.6nJ/8B)".

Per bit, the table's DRAM row is 1.3-2.6 nJ / 64 = **20-41 pJ/bit**.

**Caveats**:
- The figures are labelled "rough".
- The node is 45 nm at 0.9 V, and the DRAM generation is not stated (2014 commodity DRAM).

## 2. O'Connor et al., MICRO 2017

M. O'Connor, N. Chatterjee, D. Lee, J. Wilson, A. Agrawal, S. W. Keckler, W. J. Dally, "Fine-Grained DRAM: Energy-Efficient DRAM for Extreme Bandwidth Systems", MICRO-50, 2017. DOI 10.1145/3123939.3124545. Copy read: research.nvidia.com (oconnor_and_chatterjee.micro2017.pdf).

- **Section 1 (page 1)**: "The energy to access a bit in HBM2 is approximately 3.97 pJ/bit". Most of it is data movement (row buffer to I/O pins) and row activation.
- **Section 2 (page 2)** breaks one HBM2 access down, using the authors' energy model and real application toggle rates:
  - 2.24 pJ/bit to move data on the DRAM die and stack;
  - 0.3 pJ/bit for the interposer I/O;
  - 1.21 pJ/bit average activation energy;
  - total "3.92 pJ/bit of energy (including ECC overhead)".
- **Figure 1a (page 1)** labels GDDR5 at 14.0 pJ/bit and HBM2 at 3.9 pJ/bit.
- Their proposal (FGDRAM) reaches about 1.95-1.97 pJ/bit. That is a proposed design, not a shipping part, so we do not use it as a current figure.

**Caveats**:
- These figures come from NVIDIA's energy model, not from a meter on a shipping chip.
- The paper itself gives both 3.97 and 3.92 pJ/bit for HBM2. We carry **3.9-4.0 pJ/bit**.

## 3. Micron, via Tom's Hardware, 2020 (vendor claim, second hand)

Anton Shilov, "Micron Reveals GDDR6X Details: The Future of Memory, or a Proprietary DRAM?", Tom's Hardware, 2020-09-06.

The article says GDDR6X is "15% more power-efficient than GDDR6 (7.25 pj/bit vs 7.5 pj/bit) at the device level, according to Micron".

**Caveats**:
- This is a vendor figure reported by the press, not a paper.
- "Device level" probably excludes the GPU-side controller and PHY.
- The two numbers differ by about 3%, not 15%, as written. We carry the numbers, not the percentage, and flag the inconsistency.

## What we use in M4

| Memory | Energy per bit | Source | Status |
|---|---|---|---|
| HBM2 (stacked DRAM next to a GPU) | 3.9-4.0 pJ | O'Connor 2017 | Modelled, 2017 |
| GDDR5 | 14.0 pJ | O'Connor 2017, Figure 1a | Modelled, 2017 |
| GDDR6 / GDDR6X (device only) | 7.5 / 7.25 pJ | Micron via Tom's Hardware 2020 | Vendor claim |
| Commodity DRAM, 64-bit access | 20-41 pJ | Horowitz 2014 | Rough, 45 nm era |
| On-chip SRAM | from our own sky130 simulation | M3b | Simulated |

**Not used**: HBM3, HBM3e and GDDR7. We have not opened a primary source for their energy per bit yet, so no figure is claimed. Deng et al.'s GPUs use GDDR6 (L40S) and GDDR7 (RTX 6000 Pro). For M4 we carry a GDDR range of 7.25-14 pJ/bit and label it as spanning a vendor claim and a 2017 model. This gap is listed as open.
