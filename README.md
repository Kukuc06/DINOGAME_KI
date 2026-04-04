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
2. **Tournament selection** — random groups of 3 compete; the winner breeds the next individual
3. **Gaussian mutation** — each gene has a 15% chance of being nudged by Gaussian noise

The score is only used as fitness for the GA — it is never an input to the neural network.

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

### Console commands

```js
DinoBot.stop()                         // pause the bot
DinoBot.start()                        // resume
DinoBot.save()                         // download current best genome as dino_genome.json
DinoBot.load()                         // load a saved dino_genome.json and continue training
DinoBot.ga.reset()                     // wipe all training and start from scratch
DinoBot.forceRunner(Runner.getInstance()) // manually set runner if auto-detect fails
```

### Saving and resuming training

Because `chrome://dino` blocks `localStorage`, the bot cannot auto-save. Use the file commands instead:

```js
DinoBot.save()  // downloads dino_genome.json — do this before closing the tab
DinoBot.load()  // opens a file picker — select your dino_genome.json to resume
```

Training picks up exactly where it left off: same generation count, same all-time best, and the population is seeded from that best genome.

## Project structure

```
DINOGAME_KI/
├── src/
│   ├── NeuralNetwork.js     — feedforward NN, genome serialisation
│   ├── GeneticAlgorithm.js  — population, selection, mutation, save/load
│   └── DinoBot.js           — game interface, HUD, main loop
└── dino_bot.js              — bundled single file (paste this in the console)
```

## Tuning

All parameters are at the top of `DinoBot.js`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `populationSize` | 15 | Larger = more exploration per generation, slower |
| `mutationRate` | 0.15 | Probability of mutating each weight |
| `mutationScale` | 0.4 | Size of each mutation (Gaussian std-dev) |
| `eliteCount` | 2 | How many top individuals carry over unchanged |
| `TOPOLOGY` | `[7, 8, 2]` | Network shape — add neurons/layers for more capacity |
| `CRASH_PAUSE_MS` | 600 | How long to pause after a crash before restarting |

After editing `src/` files, rebuild the bundle:

```bash
cat src/NeuralNetwork.js src/GeneticAlgorithm.js src/DinoBot.js > dino_bot.js
```
