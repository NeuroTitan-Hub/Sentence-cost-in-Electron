# M0 tool smoke test, 2026-11-19

Purpose: prove the toolchain works on a Windows laptop. Not a result for the post.

Tools (not included in this repo):
- ngspice 47, Windows 64-bit build `ngspice-47_64.7z` from the official SourceForge project. MD5 550f3a5a278ddcb464470878aa173482 matches SourceForge's published hash.
- volare 0.20.6 (PyPI, Apache-2.0).
- sky130 PDK, volare version c6d73a35f524070e85faff4a6a9eef49553ebc2b (released 2025-08-09), libraries `common` + `sky130_fd_pr` only (about 21 MB download). Standard cells (`sky130_fd_sc_hd`, 127 MB) not fetched yet; needed later in M3a.
- WSL Ubuntu 22.04 on the laptop is broken (its disk file is missing); not used.

Run: `ngspice_con -b inverter.cir` in this folder (`.spiceinit` sets `ngbehavior=hsa`, needed for the sky130 models). The `.lib` line uses an absolute path on this laptop; M3a will make it relative.

Smoke-test output (tt corner, 27 C, 1.8 V, W/L = 1.0/0.15 um PMOS and 0.5/0.15 um NMOS, 2 fF load, 20 ps input edges):
- fall delay 17.7 ps, rise delay 20.7 ps
- energy drawn from the supply over one full cycle: 9.4 fJ
- sanity check: C x V^2 for the 2 fF load alone is 6.5 fJ; the rest is the transistors' own capacitance and short-circuit current, so the size is plausible
- wall time 18 s, mostly loading the model library

Known warning: "m=xx on .subckt line will override multiplier m hierarchy!" comes from the PDK model files; it does not affect a single-device simulation. Confirmed harmless before M3a.
