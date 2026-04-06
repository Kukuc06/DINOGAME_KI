# DINOGAME_KI

A neural network that learns to play the Chrome Dino game using a **genetic algorithm**. No pre-training, no hand-coded rules — it starts completely random and evolves better strategies through natural selection.

## How it works

### Neural network

A feedforward network (**12 inputs → 8 hidden → 2 outputs**) with sigmoid activations controls the dino. All inputs are normalised to [0, 1].

**Inputs:**

| # | Name | 0 | 1 |
|---|------|---|---|
| 0 | `dist1` | Obstacle is right at the dino | Obstacle is at the far edge of the canvas |
| 1 | `ttc` | Impact imminent | ~20 frames or more away |
| 2 | `type1` | Cactus | Pterodactyl |
| 3 | `obs-y` | Obstacle at top of canvas | Obstacle at bottom |
| 4 | `hgt` | No height | Spans full canvas height |
| 5 | `dist2` | 2nd obstacle right behind the 1st | Far away / none |
| 6 | `type2` | Cactus | Pterodactyl |
| 7 | `gap` | No gap between obstacles | Large gap between 1st and 2nd |
| 8 | `speed` | Minimum game speed | Maximum game speed (13) |
| 9 | `dino-y` | Dino on the ground | Dino at top of jump |
| 10 | `vel-y` | Maximum downward velocity | Maximum upward velocity |
| 11 | `duck` | Not ducking | Currently ducking |

**Outputs:** `[jump, duck]`

The winning output must exceed **30%** and beat the other output — jump and duck are mutually exclusive. Four possible actions per tick:

| Action | Condition |
|--------|-----------|
| `JUMP` | `jump > duck` and `jump > 0.3` (while on ground) |
| `DUCK` | `duck > jump` and `duck > 0.3` |
| `FALL` | `duck > jump` and `duck > 0.3` (while already airborne — triggers speed-drop for a fast landing) |
| `run` | Both outputs ≤ 0.3 |

---

### Fitness / reward shaping

Raw distance score alone is a weak signal. Each individual's fitness is:

```
fitness = score + speedIntegral × 0.5 + dodgeBonus
```

**speedIntegral** — accumulates `currentSpeed / MAX_SPEED` every tick. Rewards surviving longer at higher speeds.

**dodgeBonus** — awarded once per obstacle the moment it fully clears the dino's x-position (tracked by a `Set` so each obstacle is counted exactly once):

| Obstacle | Condition | Bonus |
|----------|-----------|------:|
| Cactus | Dino is jumping | +100 |
| Cactus | Dino is grounded (edge case) | +20 |
| Low pterodactyl (`yPos > 75`) | Ducking | +100 |
| Low pterodactyl (`yPos > 75`) | Not ducking | +10 |
| Mid pterodactyl (`40 < yPos ≤ 75`) | Ducking or jumping | +100 |
| High pterodactyl (`yPos ≤ 40`) | Running (no action) | +100 |
| High pterodactyl (`yPos ≤ 40`) | Jumping or ducking (unnecessary) | −20 |

---

### Genetic algorithm

Each **generation** runs **20 individuals** one after another. After all have played:

1. **Elitism** — top 3 survive unchanged into the next generation
2. **Tournament selection** — parents are chosen by picking 3 random individuals and keeping the best (k = 3)
3. **Crossover** — two parents produce a child via single-point genome crossover (65% of offspring slots)
4. **Mutation** — every weight has a 12% chance of being perturbed by Gaussian noise (Box-Muller transform, std-dev 0.25); applied to all crossover children and the remaining 35% of slots (cloned from a tournament winner then mutated)
5. **Adaptive mutation rate** — adjusts automatically after each generation:
   - New all-time best found → rate × 0.9 (min 5%)
   - No improvement for 4 generations → rate × 1.3 (max 40%)
   - If rate is already ≥ 30% (75% of the 40% ceiling) and still stagnant → **partial population reset**: bottom 25% (5 individuals) replaced with fresh random networks, stagnation counter reset
6. **Diversity maintenance** — after every evolution step, average pairwise L1 genome distance is computed across all non-elite individuals. If it drops below 0.05 (genomes too similar), the bottom 25% are replaced with fresh random networks regardless of stagnation

---

## Usage

1. Open the Chrome Dino game:
   - Navigate to `chrome://dino`, or
   - Disconnect your internet and open any page in Chrome
2. Open DevTools: `F12` → **Console** tab
3. Paste the contents of `dino_bot.js` into the console and press **Enter**
4. Press **Space once** to start the game — the bot takes over immediately

The bot polls the game state every **50 ms**. After a crash it waits **600 ms** before restarting.

---

## HUD

The top-right overlay shows all inputs and outputs live:

```
DINO BOT
mut: 10.8%

── INPUTS ──────────────────
  obstacle 1 — cactus
dist1  [||||......] 42%
ttc    [||||||....] 61%
obs-y  [..........] 0%
hgt    [|||.......] 28%
  obstacle 2 — cactus
dist2  [||||||||||] 100%
gap    [||||||||||] 100%
  dino state
speed  [|||.......] 31%
dino-y [..........] 0%
vel-y  [|||||.....] 50%
duck   [..........] 0%

── OUTPUTS ─────────────────
jump   [|||||||...] 72%
duck   [|.........] 8%

Action: JUMP
```

---

## Overlays

Five panels appear on screen during training:

| Position | Panel | Description |
|----------|-------|-------------|
| Top-right | **DINO BOT** | Live inputs, outputs, current action, mutation rate |
| Top-left | **GEN N** | Per-individual scores and origin badges for the current generation |
| Bottom-left | **TRAINING STATS** | Fitness chart, network diagram, or evolution origins (3 tabs) |
| Bottom-centre | **HALL OF FAME** | Top 5 all-time best genomes with download |
| Bottom-right | **CHECKPOINTS** | Auto-saved snapshots every 5 generations with load/save |

### Generation panel badges

| Badge | Meaning |
|-------|---------|
| `[E]` | Elite — carried over unchanged |
| `[M]` | Mutated offspring — cloned from a tournament winner then mutated |
| `[X]` | Crossover child — genome spliced from two parents |

The `←score` annotation shows the parent's score from the previous generation.

### Training Stats tabs

- **Fitness** — bar chart of generation best scores with a green step-line tracking the all-time record
- **Network** — live weight and activation diagram of the currently playing network (green = positive weights, red = negative)
- **Origins** — visualisation of the parent→child lineage for the current generation: previous gen individuals (left, ranked by fitness) connected to their offspring (right) by coloured lines

### Special modes

**Competition mode** — runs all 20 individuals of the current generation without evolving, then displays a ranked leaderboard. Accessible via the ⚔ Compete button in the GEN panel.

**Replay mode** — plays a saved genome without affecting training. Triggered via `DinoBot.replay(...)` or the Hall of Fame panel.

**Duel mode** — loads two genome files and runs them back-to-back, then displays a head-to-head score comparison.

---

## Console commands

```js
DinoBot.stop()                            // pause the bot
DinoBot.start()                           // resume
DinoBot.save()                            // download best genome as dino_genome.json
DinoBot.load()                            // import a saved genome file and continue training
DinoBot.replay(DinoBot.ga.allTimeBest)    // watch the best genome play without affecting training
DinoBot.duel()                            // load two genome files and compare them head-to-head
DinoBot.ga.reset()                        // wipe all training and start from scratch
DinoBot.forceRunner(Runner.getInstance()) // manually set runner if auto-detect fails
```

---

## Saving and resuming training

The best genome is auto-saved to `localStorage` whenever a new all-time best is reached and restored automatically on page reload.

```js
DinoBot.save()  // downloads dino_genome.json
DinoBot.load()  // opens a file picker — select dino_genome.json to resume
```

The GEN panel **Save** button exports the full population (all 20 genomes + current state). It can be re-imported via the **Import** button in the Checkpoints panel.

Checkpoints (best genome snapshot) are auto-saved every 5 generations and kept in memory (up to 10). Each can be downloaded or loaded from the Checkpoints panel.

---

## Project structure

```
DINOGAME_KI/
├── src/
│   ├── NeuralNetwork.js     — feedforward NN, sigmoid activation, genome serialisation
│   ├── GeneticAlgorithm.js  — selection, crossover, mutation, diversity maintenance, checkpoints
│   └── DinoBot.js           — game interface, input vector, reward shaping, all overlays, main loop
└── dino_bot.js              — single bundled file (paste this into the browser console)
```

After editing `src/` files, rebuild the bundle:

```bash
cat src/NeuralNetwork.js src/GeneticAlgorithm.js src/DinoBot.js > dino_bot.js
```

---

## Tuning

All parameters are at the top of `DinoBot.js` (constants) and in the `GeneticAlgorithm` constructor call:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `populationSize` | 20 | Individuals per generation |
| `mutationRate` | 0.12 | Starting per-weight mutation probability |
| `mutationScale` | 0.25 | Gaussian std-dev for each mutation |
| `eliteCount` | 3 | Top individuals carried over unchanged |
| `crossoverRate` | 0.65 | Fraction of offspring slots using crossover |
| `adaptiveMutationMin` | 0.05 | Floor for mutation rate |
| `adaptiveMutationMax` | 0.40 | Ceiling before partial reset kicks in |
| `stagnationThreshold` | 4 | Stagnant generations before boosting mutation |
| `topology` | `[12, 8, 2]` | Network shape — add neurons/layers for more capacity |
| `autoSaveEvery` | 5 | Checkpoint interval in generations |
| `POLL_MS` | 50 | Game polling interval in milliseconds |
| `CRASH_PAUSE_MS` | 600 | Pause after a crash before restarting |
