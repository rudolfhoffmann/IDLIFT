# IDLIFT — Interval-based Dynamic LIFT

IDLIFT learns the structure of **Dynamic Fault Trees (DFTs)** from observational data. Input data is
an **Interval-based Temporal Event Table (ITET)** — a dataset where each row represents one observation window and each cell 
records the time intervals during which an event was active. IDLIFT uses the **Partial Association Mantel-Haenszel (PAMH)** 
statistical test  to identify gate relationships (OR, AND, PAND, SEQ) between events layer by layer.

Based on the LIFT algorithm by M. Nauta (https://github.com/M-Nauta/LIFT), extended to interval-valued temporal data.

---

## Requirements

```
pip install pandas numpy scipy
```

---

## DFT Format

A DFT is represented as a Python dictionary:

```python
dft = {
    'TE':  [('IE1', 'IE2'), 'OR'],
    'IE1': [('BE1', 'BE2'), 'AND'],
    'IE2': [('BE3', 'BE4'), 'PAND'],
    'BE1': [(), 'BE'],
    'BE2': [(), 'BE'],
    'BE3': [(), 'BE'],
    'BE4': [(), 'BE'],
}
```

Each entry: `node: [(children), gate_type]`  
Gate types: `OR`, `AND`, `PAND`, `SEQ`, `BE` (Basic Event — no children)

---

## ITET Format

An ITET is a pandas DataFrame where each row is one observation window.  
Each cell is either `0` (event did not occur) or a **frozenset of `(t_start, t_end)` tuples** — one per occurrence of the event.  
The label column (Top Event) is binary (`0` or `1`).

```python
# Example ITET (3 rows, 4 events + Top Event)
#
#    BE1                  BE2              BE3   BE4              TE
# 0  {(0.0, 2.25)}        {(1.5, 4.0)}    0     {(3.8, 6.1)}    1
# 1  0                    {(0.5, 3.0)}    0     0                0
# 2  {(1.0, 5.5)}         {(1.0, 3.0),
#                          (7.0, 9.0)}    0     {(2.8, 4.2)}    1
Multiple tuples per cell: event fired more than once in that window (e.g., BE2 in row 2)
0: event did not occur in that window
TE column: always binary — 1 if the Top Event fired, 0 otherwise
Use ITET_simulation() to generate synthetic ITET data from a known DFT (see Usage below).

---

## Usage

```python
from IDLIFT import ITET_simulation, learnFTandcheck

# 1. Define the reference DFT
dft = {
    'TE':  [('IE1', 'IE2'), 'OR'],
    'IE1': [('BE1', 'BE2'), 'AND'],
    'IE2': [('BE3', 'BE4'), 'PAND'],
    'BE1': [(), 'BE'],
    'BE2': [(), 'BE'],
    'BE3': [(), 'BE'],
    'BE4': [(), 'BE'],
}

# 2. Generate synthetic ITET training data
df = ITET_simulation(
    dft,
    window_seconds=10.0,  # observation window length; based on your system
    n_samples=1000,
    p_TE=0.5,
    interval_tolerance=0.05,  # based on sensor sampling rate
    random_seed=42,
)

# 3. Learn the DFT structure from the data
match, learned_dft = learnFTandcheck(
    {},
    df,
    significance=0.95,
    top_event='TE',
    interval_tolerance=0.05,
)

print(learned_dft)
```

---

## Main Functions

| Function | Description |
|---|---|
| `ITET_simulation(dft, ...)` | Generate a synthetic ITET DataFrame from a known DFT |
| `learnFTandcheck(tree, df, significance, ...)` | Learn a full DFT layer by layer from an ITET |
| `learn_depthN_DFT_with_most_significant_relationship(df, significance, n_depth, ...)` | Depth-limited learning |
| `learn_depth1_DFT_with_N_significant_relationships(df, significance, N, ...)` | Depth-1 learning, top-N relationships |

---

## License

MIT License — free to use with attribution.  
Copyright (c) 2026 Rudolf Hoffmann

---

## Author

Rudolf Hoffmann and Christoph Reich
