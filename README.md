# Levin Tree Search for The Witness Puzzle

Implementation of **Levin Tree Search (LevinTS)** with a **Bootstrap learning** framework to solve puzzles from the game *The Witness*, built in Python as part of a Search & Planning in AI course at the University of Alberta.

This project combines **heuristic search** with **online policy learning** — the system gets better at solving puzzles the more it searches, using solution paths to train a learned policy via gradient descent.

---

## Overview

The Witness puzzle requires drawing a line on a grid from a start to a goal location, while partitioning colored bullets into separate regions. The puzzle is solved when:
- The line connects start to goal
- No region contains bullets of more than one color

Agents can move **up, down, left, or right** (no diagonals). The line cannot cross itself.

---

## System Architecture

### 1. Levin Tree Search (LevinTS)

LevinTS is a best-first search algorithm that sorts nodes by the **Levin cost**:

$$\text{cost}(n) = \frac{d(n)}{\pi(n)}$$

where `d(n)` is the depth of node `n` and `π(n)` is the probability of reaching `n` under the current policy. Nodes that are deep *and* likely under the policy are prioritized.

For numerical stability, all computations are done in **log space**:

$$\text{cost}(n) = \log(d(n)) - \log(\pi(n))$$

Key properties:
- Uses a **CLOSED list** to avoid revisiting states (transposition table)
- Uses **parent pruning** to avoid immediately undoing the previous move
- Returns the first solution found (no optimality guarantee — speed is the goal)
- Supports a **node expansion budget** — if exceeded, returns failure and doubles the budget

### 2. Context Model + Policy Learning

The policy `π` is represented as a **context model** — a lookup table mapping local puzzle contexts to action probability distributions.

A **context** is a 4-tuple describing the cells around the current tip of the line (top-left, top-right, bottom-left, bottom-right). Each cell is encoded as:
- `0` — empty
- `1` — red bullet
- `2` — blue bullet  
- `-1` — outside the grid

The model maps each context to a vector of logits (one per action), converted to probabilities via **Softmax**. The model is updated via **gradient descent on the log-loss**:

```
gradient[a] = π(a | context) - 1   (for the action taken)
gradient[b] = π(b | context)        (for all other actions)
T[context] -= α * gradient
```

A **symmetry augmentation** doubles the training signal: since red and blue are symmetric in this puzzle, each solution path is used to update both the regular context and a color-swapped "reversed" context.

### 3. Bootstrap Training

The Bootstrap framework trains the model iteratively:

1. For each unsolved puzzle, run LevinTS with budget `B`
2. If solved, update the model 10 times using the solution path
3. If no puzzles were solved in an entire sweep, double `B`
4. Repeat until all puzzles are solved

This compensates for a weak early policy with more search, and gradually shifts work to the learned policy as it improves.

---

## Results

Training on 4×4 Witness puzzles with 2 colors:

- **Early iterations**: few problems solved per sweep — the uniform policy provides no useful guidance, so search does most of the work
- **Later iterations**: solve rate increases significantly as the learned policy steers search toward promising paths
- **Symmetry augmentation (version 2)** consistently solves more problems per iteration than the baseline (version 1), because each solution path produces twice the training signal

Training progress is logged automatically to `training_logs/training_bootstrap_<timestamp>.csv`.

---

## Project Structure

```
├── main.py                  # Entry point: runs easy instance + Bootstrap training
├── levin_tree_search.py     # LevinTS search algorithm + TreeNode + Trajectory
├── model.py                 # Context model with Softmax + gradient descent update
├── bootstrap.py             # Bootstrap training loop with budget doubling
├── witness.py               # WitnessState puzzle implementation
├── puzzle_generator.py      # Random puzzle generation with configurable parameters
├── problems/
│   └── puzzles_4x4/         # Pre-generated 4×4 puzzle instances
└── training_logs/           # CSV logs generated during Bootstrap training
```

---

## Getting Started

**Requirements:** Python 3

```bash
# (Optional) Create a virtual environment
virtualenv -p python3 venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python3 main.py
```

Running `main.py` will:
1. Solve an easy puzzle instance using a uniform policy (no learning) to verify LevinTS works
2. Load all 4×4 puzzle instances
3. Run Bootstrap training, printing solve status for each puzzle per iteration
4. Save training logs to `training_logs/`

---

## Generating New Puzzles

```bash
python3 puzzle_generator.py -l 4 -c 4 -colors 2 -n 1000 -folder problems/puzzles_4x4
```

| Flag | Description | Default |
|---|---|---|
| `-l` | Number of rows | 4 |
| `-c` | Number of columns | 4 |
| `-colors` | Number of bullet colors | 2 |
| `-n` | Number of puzzles to generate | 1000 |
| `-p` | Probability of placing a bullet | 0.6 |
| `-time` | Time limit in seconds | 300 |

---

## Key Concepts

- **Levin cost**: balances search depth against policy probability — prefers shallow nodes that the policy considers likely
- **Log-space arithmetic**: prevents floating-point underflow when multiplying many small probabilities
- **Context model**: lightweight tabular policy; generalizes across puzzle states that share local structure
- **Bootstrap**: a search-and-learn loop — search finds solutions, solutions improve the policy, better policy makes search faster
- **Symmetry augmentation**: exploits color symmetry to double training data without solving additional puzzles

---

## Technologies

- **Python 3**
- `numpy` — Softmax, log-space arithmetic, gradient updates
- `heapq` — priority queue for LevinTS OPEN list
- `copy.deepcopy` — state copying during tree expansion
