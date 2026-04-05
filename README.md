# DINOGAME_KI

A neural network that learns to play the Chrome Dino game using a **genetic algorithm**. No pre-training, no hand-coded rules — it starts completely random and gradually evolves better strategies through natural selection.

## How it works

### Neural network
A small feedforward network (7 inputs → 8 hidden → 2 outputs) with sigmoid activations controls the dino.

**Inputs** (all normalised to 0–100%):

| # | Input | 0% | 100% |
|---|-------|----|------|
| 0 | `dist` | Obstacle is right on the dino | Obstacle is far away / no obstacle |
| 1 | `width` | Obstacle has no width | Obstacle spans the full canvas |
| 2 | `height` | Obstacle has no height | Obstacle spans the full canvas height |
| 3 | `type` | Cactus | Pterodactyl |
| 4 | `obs-y` | Obstacle is at the top of the canvas | Obstacle is at the bottom |
| 5 | `speed` | Minimum game speed | Maximum game speed (~13) |
| 6 | `dino-y` | Dino is on the ground | Dino is at the top of its jump |

**Outputs**: `[jump, duck]`
- The higher output wins if it exceeds 50% — jump and duck are mutually exclusive
- If both are below 50%, the dino runs

### Genetic algorithm
Each **generation** runs 15 individuals one after another. When an individual crashes, its `distanceRan` score becomes its **fitness**. After all 15 have played:

1. **Elitism** — top 2 survive unchanged into the next generation
2. **Crossover** — two parents selected via tournament (groups of 3) produce a child by single-point genome crossover (60% chance per offspring slot)
3. **Tournament selection + mutation** — remaining slots are filled by cloning a tournament winner and applying Gaussian mutation
4. **Adaptive mutation** — mutation rate auto-adjusts between 5% and 50% based on stagnation: it decreases after improvement, increases after 5 stagnant generations

The score is only used as fitness — it is never an input to the neural network.

## Usage

1. Open the Chrome Dino game:
   - Navigate to `chrome://dino`, or
   - Disconnect your internet and open any page in Chrome
2. Open DevTools: `F12` → **Console** tab
3. Paste the contents of `dino_bot.js` into the console and press **Enter**
4. Press **Space once** to start the game — the bot takes over immediately

The HUD in the top-right corner shows all inputs and outputs live:

```
DINO BOT
Gen 4  #7 / 15

Score    : 312
Gen best : 489
All-time : 831
mut: 13.5%

── INPUTS ──────────────────
dist   [||||......] 42%
width  [||........] 18%
height [|||.......] 28%
type   [..........] 0%  cactus
obs-y  [..........] 0%
speed  [|||.......] 31%
dino-y [..........] 0%

── OUTPUTS ─────────────────
jump  [|||||||...] 72%
duck  [|.........] 8%

Action: JUMP
```

## Overlays

Four panels appear on screen during training:

| Position | Panel | Description |
|----------|-------|-------------|
| Top-right | **DINO BOT** | Live inputs, outputs, current action |
| Top-left | **GEN N** | Per-individual scores and origin badges for the current generation |
| Bottom-left | **TRAINING STATS** | Fitness chart, network diagram, or evolution origins (3 tabs) |
| Bottom (next to stats) | **HALL OF FAME** | Top 5 all-time best genomes with replay and download |
| Bottom-right | **CHECKPOINTS** | Auto-saved snapshots every 5 generations |

### Generation panel badges

Each individual in the GEN panel shows where it came from:

| Badge | Meaning |
|-------|---------|
| `[E]` | Elite — carried over unchanged from the previous generation |
| `[M]` | Mutated offspring — cloned from a tournament winner then mutated |
| `[X]` | Crossover child — genome spliced from two parents |

The `←score` annotation shows the parent's score from the previous generation.

### Training Stats tabs

- **Fitness** — line chart of all-time best (green) and generation best (yellow dashed) over time
- **Network** — live weight and activation diagram of the currently playing network
- **Origins** — canvas visualisation of the parent→child lineage for the current generation: previous gen individuals (left, ranked by fitness) connected by coloured lines to their offspring (right)

### Console commands

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

### Saving and resuming training

The best genome is auto-saved to `localStorage` whenever a new all-time best is reached. On page reload it is restored automatically.

For explicit saves or moving training between tabs:

```js
DinoBot.save()  // downloads dino_genome.json — do this before closing the tab
DinoBot.load()  // opens a file picker — select your dino_genome.json to resume
```

You can also save the full population (all 15 genomes + current state) from the GEN panel's **Save** button, and reload it later via **Import** in the Checkpoints panel.

## Project structure

```
DINOGAME_KI/
├── src/
│   ├── NeuralNetwork.js     — feedforward NN with genome serialisation
│   ├── GeneticAlgorithm.js  — population management, crossover, mutation, checkpoints
│   └── DinoBot.js           — game interface, all overlays, main loop
└── dino_bot.js              — bundled single file (paste this in the console)
```

## Tuning

All parameters are at the top of `DinoBot.js` / `GeneticAlgorithm.js`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `populationSize` | 15 | Larger = more exploration per generation, slower |
| `mutationRate` | 0.15 | Starting probability of mutating each weight |
| `mutationScale` | 0.4 | Size of each mutation (Gaussian std-dev) |
| `eliteCount` | 2 | How many top individuals carry over unchanged |
| `crossoverRate` | 0.6 | Probability of crossover vs pure mutation per offspring slot |
| `adaptiveMutationMin` | 0.05 | Minimum mutation rate (after improvement) |
| `adaptiveMutationMax` | 0.50 | Maximum mutation rate (after stagnation) |
| `stagnationThreshold` | 5 | Generations without improvement before boosting mutation |
| `TOPOLOGY` | `[7, 8, 2]` | Network shape — add neurons/layers for more capacity |
| `CRASH_PAUSE_MS` | 600 | How long to pause after a crash before restarting |

After editing `src/` files, rebuild the bundle:

```bash
cat src/NeuralNetwork.js src/GeneticAlgorithm.js src/DinoBot.js > dino_bot.js
```
