/**
 * NeuralNetwork
 *
 * A minimal feedforward neural network with sigmoid activations.
 * Weights and biases are stored as flat arrays (the "genome") so
 * they can be easily exchanged with the genetic algorithm.
 *
 * topology: e.g. [7, 8, 2]
 *   → 7 inputs, one hidden layer of 8 neurons, 2 outputs
 */
class NeuralNetwork {
  constructor(topology) {
    this.topology = topology;
    this.weights = []; // weights[layer] = 2D array [outputNeuron][inputNeuron]
    this.biases = [];  // biases[layer] = 1D array [outputNeuron]

    for (let i = 0; i < topology.length - 1; i++) {
      const ins = topology[i];
      const outs = topology[i + 1];
      this.weights.push(
        Array.from({ length: outs }, () =>
          Array.from({ length: ins }, () => Math.random() * 2 - 1)
        )
      );
      this.biases.push(
        Array.from({ length: outs }, () => Math.random() * 2 - 1)
      );
    }
  }

  /** Run a forward pass and return the output activations. */
  forward(inputs) {
    return this.forwardWithActivations(inputs).output;
  }

  /** Forward pass that also returns per-layer activations (needed by network diagram). */
  forwardWithActivations(inputs) {
    const layerActivations = [inputs.slice()];
    let activation = inputs.slice();
    for (let l = 0; l < this.weights.length; l++) {
      const next = [];
      for (let j = 0; j < this.weights[l].length; j++) {
        let sum = this.biases[l][j];
        for (let k = 0; k < this.weights[l][j].length; k++) {
          sum += this.weights[l][j][k] * activation[k];
        }
        next.push(NeuralNetwork.sigmoid(sum));
      }
      layerActivations.push(next);
      activation = next;
    }
    return { output: activation, layerActivations };
  }

  static sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  /** Flatten all weights and biases into a single 1-D array (the genome). */
  toGenome() {
    const g = [];
    for (const wMatrix of this.weights)
      for (const row of wMatrix) g.push(...row);
    for (const bVec of this.biases) g.push(...bVec);
    return g;
  }

  /** Load weights and biases from a flat genome array. */
  fromGenome(genome) {
    let idx = 0;
    for (let l = 0; l < this.weights.length; l++) {
      for (let j = 0; j < this.weights[l].length; j++) {
        for (let k = 0; k < this.weights[l][j].length; k++) {
          this.weights[l][j][k] = genome[idx++];
        }
      }
    }
    for (let l = 0; l < this.biases.length; l++) {
      for (let j = 0; j < this.biases[l].length; j++) {
        this.biases[l][j] = genome[idx++];
      }
    }
    return this;
  }

  clone() {
    return new NeuralNetwork(this.topology).fromGenome(this.toGenome());
  }
}
/**
 * GeneticAlgorithm
 *
 * Evolves a population of NeuralNetwork genomes across episodes.
 *
 * Workflow:
 *   1. Call getCurrentNetwork() to get the network for the current individual.
 *   2. Let it play until it dies, then call recordFitness(score).
 *   3. Repeat until the whole population has been evaluated.
 *   4. _evolve() is called automatically — selects elites, produces offspring
 *      via crossover + tournament selection + Gaussian mutation.
 *   5. A checkpoint is stored in memory every autoSaveEvery generations.
 *
 * New in v2:
 *   - crossoverRate: probability of crossover vs pure mutation (default 0.6)
 *   - adaptiveMutation: auto-adjusts mutationRate based on stagnation
 *   - hallOfFame: top-5 all-time best genomes with timestamps
 */
class GeneticAlgorithm {
  constructor({
    populationSize        = 20,
    mutationRate          = 0.12,
    mutationScale         = 0.25,
    eliteCount            = 3,
    topology              = [8, 8, 2],
    autoSaveEvery         = 5,
    maxCheckpoints        = 10,
    crossoverRate         = 0.65,
    adaptiveMutation      = true,
    adaptiveMutationMin   = 0.05,
    adaptiveMutationMax   = 0.40,
    stagnationThreshold   = 4,
  } = {}) {
    this.populationSize       = populationSize;
    this.mutationRate         = mutationRate;
    this._baseMutationRate    = mutationRate;
    this.mutationScale        = mutationScale;
    this.eliteCount           = eliteCount;
    this.topology             = topology;
    this.autoSaveEvery        = autoSaveEvery;
    this.maxCheckpoints       = maxCheckpoints;
    this.crossoverRate        = crossoverRate;
    this.adaptiveMutation     = adaptiveMutation;
    this.adaptiveMutationMin  = adaptiveMutationMin;
    this.adaptiveMutationMax  = adaptiveMutationMax;
    this.stagnationThreshold  = stagnationThreshold;

    this.generation           = 1;
    this.currentIndex         = 0;
    this.fitnesses            = new Array(populationSize).fill(0);
    this.allTimeBest          = null;
    this.allTimeBestFitness   = -Infinity;
    this.allTimeBestRawScore  = 0;
    this.stagnantGens         = 0;
    this.hallOfFame           = [];   // top-5 all-time bests
    this.checkpoints          = [];   // in-memory list, newest first
    this.evolutionLog         = null; // set after each _evolve()
    this.competitionMode      = false;
    this.onCheckpoint         = null; // fired when checkpoint/HoF list changes
    this.onEvolve             = null; // fired when a new generation starts
    this.onGenerationComplete = null; // fired after all individuals play (competition)

    this.population = Array.from(
      { length: populationSize },
      () => new NeuralNetwork(topology)
    );
  }

  /** The network that should play right now. */
  getCurrentNetwork() {
    return this.population[this.currentIndex];
  }

  /**
   * Record the score for the current individual and advance.
   * Triggers evolution automatically when the generation is complete.
   */
  recordFitness(fitness, rawScore) {
    this.fitnesses[this.currentIndex] = fitness;

    try {
      // Track top-5 by real game score (rawScore), not shaped fitness
      const realScore = rawScore !== undefined ? rawScore : fitness;

      if (fitness > this.allTimeBestFitness) {
        this.allTimeBestFitness  = fitness;
        this.allTimeBestRawScore = realScore;
        this.allTimeBest = this.population[this.currentIndex].clone();
        this._saveToStorage();
      }

      if (realScore > 0) {
        const hofWorst = this.hallOfFame.length < 5
          ? -Infinity
          : this.hallOfFame[this.hallOfFame.length - 1].rawScore;
        if (this.hallOfFame.length < 5 || realScore > hofWorst) {
          this._updateHallOfFame(fitness, this.population[this.currentIndex], realScore);
        }
      }
    } catch (err) {
      console.error('[DinoBot] recordFitness error (individual ' + (this.currentIndex + 1) + '):', err);
    }

    this.currentIndex++;
    if (this.currentIndex >= this.populationSize) {
      if (this.competitionMode) {
        if (this.onGenerationComplete) this.onGenerationComplete();
      } else {
        this._evolve();
      }
    }
  }

  // ── Private ────────────────────────────────────────────────────────────────

  _evolve() {
    this.generation++;

    const fitnessBefore = this.allTimeBestFitness;

    const ranked = this.population
      .map((nn, i) => ({ nn, fitness: this.fitnesses[i] }))
      .sort((a, b) => b.fitness - a.fitness);

    const prevFitnesses = this.fitnesses.slice();
    const origins       = [];
    const next          = [];

    // Elitism: top N survive unchanged
    for (let i = 0; i < this.eliteCount && i < ranked.length; i++) {
      next.push(ranked[i].nn.clone());
      origins.push({ type: 'elite', prevScore: ranked[i].fitness });
    }

    // Fill rest with crossover or mutation
    while (next.length < this.populationSize) {
      if (this.crossoverRate > 0 && Math.random() < this.crossoverRate && ranked.length >= 2) {
        const parentA = this._tournamentSelect(ranked);
        const parentB = this._tournamentSelect(ranked);
        const child   = this._crossover(parentA.nn, parentB.nn);
        this._mutate(child);
        next.push(child);
        origins.push({ type: 'crossover', prevScoreA: parentA.fitness, prevScoreB: parentB.fitness });
      } else {
        const winner = this._tournamentSelect(ranked);
        const child  = winner.nn.clone();
        this._mutate(child);
        next.push(child);
        origins.push({ type: 'offspring', prevScore: winner.fitness });
      }
    }

    this.population   = next;
    this.fitnesses    = new Array(this.populationSize).fill(0);
    this.currentIndex = 0;
    this.evolutionLog = { prevFitnesses, origins };

    // Adaptive mutation rate
    if (this.adaptiveMutation) {
      if (this.allTimeBestFitness > fitnessBefore) {
        this.stagnantGens = 0;
        this.mutationRate = Math.max(this.adaptiveMutationMin, this.mutationRate * 0.9);
      } else {
        this.stagnantGens++;
        if (this.stagnantGens >= this.stagnationThreshold) {
          this.mutationRate = Math.min(this.adaptiveMutationMax, this.mutationRate * 1.3);
        }
      }
    }

    if (this.autoSaveEvery > 0 && this.generation % this.autoSaveEvery === 0) {
      this._addCheckpoint();
    }
    if (this.onEvolve) this.onEvolve();
  }

  _tournamentSelect(ranked, k = 3) {
    let best = null, bestFitness = -Infinity;
    for (let i = 0; i < k; i++) {
      const c = ranked[Math.floor(Math.random() * ranked.length)];
      if (c.fitness > bestFitness) { bestFitness = c.fitness; best = c; }
    }
    return best; // { nn, fitness }
  }

  _crossover(nnA, nnB) {
    const genomeA = nnA.toGenome();
    const genomeB = nnB.toGenome();
    const point   = Math.floor(Math.random() * genomeA.length);
    return new NeuralNetwork(this.topology).fromGenome(
      genomeA.slice(0, point).concat(genomeB.slice(point))
    );
  }

  _mutate(nn) {
    const genome = nn.toGenome();
    for (let i = 0; i < genome.length; i++) {
      if (Math.random() < this.mutationRate) {
        const u1 = Math.random() || 1e-10;
        const u2 = Math.random();
        genome[i] += Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * this.mutationScale;
      }
    }
    nn.fromGenome(genome);
  }

  _updateHallOfFame(fitness, nn, rawScore) {
    this.hallOfFame.push({
      genome:     nn.toGenome(),
      topology:   this.topology,
      fitness,
      rawScore,
      generation: this.generation,
      timestamp:  new Date().toLocaleTimeString(),
    });
    // Sort and trim by real game score so HoF reflects what the user actually sees
    this.hallOfFame.sort((a, b) => b.rawScore - a.rawScore);
    if (this.hallOfFame.length > 5) this.hallOfFame.pop();
    if (this.onCheckpoint) this.onCheckpoint();
  }

  // ── Checkpoints ────────────────────────────────────────────────────────────

  _addCheckpoint() {
    if (!this.allTimeBest) return;
    this.checkpoints.unshift({
      genome:     this.allTimeBest.toGenome(),
      topology:   this.topology,
      fitness:    Math.max(0, this.allTimeBestFitness),
      rawScore:   this.allTimeBestRawScore,
      generation: this.generation,
      timestamp:  new Date().toLocaleTimeString(),
    });
    if (this.checkpoints.length > this.maxCheckpoints) this.checkpoints.pop();
    if (this.onCheckpoint) this.onCheckpoint();
  }

  downloadCheckpoint(index) {
    const cp = this.checkpoints[index];
    if (!cp) return;
    this._download(
      { type: 'best', genome: cp.genome, topology: cp.topology, fitness: cp.fitness, generation: cp.generation },
      `dino_gen${cp.generation}_score${Math.round(cp.fitness)}.json`
    );
  }

  downloadHallOfFame(index) {
    const entry = this.hallOfFame[index];
    if (!entry) return;
    this._download(
      { type: 'best', genome: entry.genome, topology: entry.topology, fitness: entry.fitness, generation: entry.generation },
      `dino_hof${index + 1}_gen${entry.generation}_score${Math.round(entry.fitness)}.json`
    );
  }

  loadCheckpoint(index) {
    const cp = this.checkpoints[index];
    if (!cp) return;
    this._seedFromBest(new NeuralNetwork(cp.topology).fromGenome(cp.genome), cp.fitness, cp.generation, cp.rawScore);
    console.log('[DinoBot] Loaded checkpoint — gen ' + cp.generation + ', score ' + (cp.rawScore || Math.round(cp.fitness)));
    if (this.onCheckpoint) this.onCheckpoint();
    if (this.onEvolve)     this.onEvolve();
  }

  // ── File export ────────────────────────────────────────────────────────────

  exportToFile() {
    if (!this.allTimeBest) { console.warn('[DinoBot] No best genome yet.'); return; }
    this._download(
      { type: 'best', genome: this.allTimeBest.toGenome(), topology: this.topology,
        fitness: this.allTimeBestFitness, generation: this.generation },
      'dino_genome.json'
    );
    console.log('[DinoBot] Saved dino_genome.json (gen ' + this.generation + ', score ' + Math.round(this.allTimeBestFitness) + ')');
  }

  exportPopulationToFile() {
    this._download(
      { type: 'population',
        genomes:             this.population.map(nn => nn.toGenome()),
        fitnesses:           this.fitnesses.slice(),
        topology:            this.topology,
        generation:          this.generation,
        currentIndex:        this.currentIndex,
        allTimeBestGenome:   this.allTimeBest ? this.allTimeBest.toGenome() : null,
        allTimeBestFitness:  this.allTimeBestFitness,
      },
      `dino_gen${this.generation}_population.json`
    );
    console.log('[DinoBot] Saved full population (gen ' + this.generation + ')');
  }

  importFromFile() {
    const input = Object.assign(document.createElement('input'), { type: 'file', accept: '.json' });
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        try {
          const data = JSON.parse(ev.target.result);
          if (JSON.stringify(data.topology) !== JSON.stringify(this.topology)) {
            console.error('[DinoBot] Topology mismatch.'); return;
          }
          if (data.type === 'population') {
            this.population   = data.genomes.map(g => new NeuralNetwork(data.topology).fromGenome(g));
            this.fitnesses    = data.fitnesses.slice();
            this.generation   = data.generation;
            this.currentIndex = data.currentIndex || 0;
            if (data.allTimeBestGenome) {
              this.allTimeBest        = new NeuralNetwork(data.topology).fromGenome(data.allTimeBestGenome);
              this.allTimeBestFitness = data.allTimeBestFitness;
              this._updateHallOfFame(this.allTimeBestFitness, this.allTimeBest);
            }
            console.log('[DinoBot] Loaded full population — gen ' + data.generation);
          } else {
            const best = new NeuralNetwork(data.topology).fromGenome(data.genome);
            this._seedFromBest(best, data.fitness, data.generation);
            this._updateHallOfFame(data.fitness, best);
            console.log('[DinoBot] Loaded best genome — gen ' + data.generation + ', score ' + Math.round(data.fitness));
          }
          this._addCheckpoint();
          if (this.onCheckpoint) this.onCheckpoint();
          if (this.onEvolve)     this.onEvolve();
        } catch (err) { console.error('[DinoBot] Failed to parse file:', err); }
      };
      reader.readAsText(file);
    };
    input.click();
  }

  // ── Reset ──────────────────────────────────────────────────────────────────

  reset() {
    try { localStorage.removeItem('dinoBotGenome'); } catch (_) {}
    this.generation         = 1;
    this.currentIndex       = 0;
    this.fitnesses          = new Array(this.populationSize).fill(0);
    this.allTimeBest         = null;
    this.allTimeBestFitness  = -Infinity;
    this.allTimeBestRawScore = 0;
    this.stagnantGens        = 0;
    this.mutationRate        = this._baseMutationRate;
    this.checkpoints        = [];
    this.hallOfFame         = [];
    this.population = Array.from({ length: this.populationSize }, () => new NeuralNetwork(this.topology));
    console.log('[DinoBot] Training reset.');
    if (this.onCheckpoint) this.onCheckpoint();
    if (this.onEvolve)     this.onEvolve();
  }

  // ── Stats ──────────────────────────────────────────────────────────────────

  getStats() {
    const evaluated = this.fitnesses.slice(0, this.currentIndex);
    return {
      generation:     this.generation,
      individual:     this.currentIndex + 1,
      populationSize: this.populationSize,
      generationBest: evaluated.length ? Math.round(Math.max(...evaluated)) : 0,
      allTimeBest:    Math.round(this.allTimeBestFitness < 0 ? 0 : this.allTimeBestFitness),
      mutationRate:   this.mutationRate,
      stagnantGens:   this.stagnantGens,
    };
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  _seedFromBest(nn, fitness, generation, rawScore) {
    this.allTimeBest         = nn;
    this.allTimeBestFitness  = fitness;
    this.allTimeBestRawScore = rawScore !== undefined ? rawScore : Math.round(fitness);
    this.generation          = generation;
    for (let i = 0; i < this.populationSize; i++) {
      const clone = nn.clone();
      if (i > 0) this._mutate(clone);
      this.population[i] = clone;
    }
    this.currentIndex = 0;
    this.fitnesses    = new Array(this.populationSize).fill(0);
  }

  _download(obj, filename) {
    const a = Object.assign(document.createElement('a'), {
      href:     URL.createObjectURL(new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' })),
      download: filename,
    });
    a.click();
    URL.revokeObjectURL(a.href);
  }

  // ── localStorage (best-effort) ─────────────────────────────────────────────

  _saveToStorage() {
    try {
      localStorage.setItem('dinoBotGenome', JSON.stringify({
        type: 'best', genome: this.allTimeBest.toGenome(),
        topology: this.topology, fitness: this.allTimeBestFitness,
        rawScore: this.allTimeBestRawScore, generation: this.generation,
      }));
    } catch (_) {}
  }

  loadFromStorage() {
    let raw;
    try { raw = localStorage.getItem('dinoBotGenome'); } catch (_) { return false; }
    if (!raw) return false;
    try {
      const data = JSON.parse(raw);
      if (JSON.stringify(data.topology) !== JSON.stringify(this.topology)) return false;
      this._seedFromBest(new NeuralNetwork(data.topology).fromGenome(data.genome), data.fitness, data.generation, data.rawScore);
      console.log('[DinoBot] Loaded saved genome — gen ' + data.generation + ', score ' + (data.rawScore || Math.round(data.fitness)));
      return true;
    } catch (_) { return false; }
  }
}
/**
 * DinoBot — main controller (v2)
 *
 * New features: fitness chart, network diagram, adaptive mutation display,
 * crossover [X] badge, Hall of Fame with replay, replay mode, duel mode.
 *
 * Depends on: NeuralNetwork, GeneticAlgorithm (load them first).
 */
window.DinoBot = (() => {
  // ── Constants ──────────────────────────────────────────────────────────────

  const DINO_X         = 75;
  const CANVAS_WIDTH   = 600;
  const CANVAS_HEIGHT  = 150;
  const MAX_SPEED      = 13;
  const TOPOLOGY       = [8, 8, 2];
  const POLL_MS        = 50;
  const CRASH_PAUSE_MS = 600;
  const MAX_VEL        = 12;   // max dino vertical velocity (canvas units per tick)

  const INPUT_LABELS = ['dist  ', 'width ', 'height', 'type  ', 'obs-y ', 'speed ', 'dino-y', 'vel-y '];

  // ── GA setup ───────────────────────────────────────────────────────────────

  const ga = new GeneticAlgorithm({
    populationSize:       20,
    mutationRate:         0.12,
    mutationScale:        0.25,
    eliteCount:           3,
    topology:             TOPOLOGY,
    autoSaveEvery:        5,
    maxCheckpoints:       10,
    crossoverRate:        0.65,
    adaptiveMutation:     true,
    adaptiveMutationMin:  0.05,
    adaptiveMutationMax:  0.40,
    stagnationThreshold:  4,
  });

  ga.loadFromStorage();

  // ── State ──────────────────────────────────────────────────────────────────

  let intervalId         = null;
  let overlay            = null;   // top-right: main HUD
  let cpPanel            = null;   // bottom-right: checkpoints
  let hofPanel           = null;   // bottom: Hall of Fame (right of training stats)
  let genPanel           = null;   // top-left: generation performance
  let blPanel            = null;   // bottom-left: fitness chart / network diagram
  let blHeader           = null;
  let blCanvas           = null;
  let cpExpanded         = true;
  let hofExpanded        = true;
  let genExpanded        = true;
  let blExpanded         = true;
  let blTab              = 'fitness';   // 'fitness' | 'network'

  let competitionActive  = false;
  let competitionDone    = false;
  let competitionResults = null;

  let replayMode         = false;
  let replayNetwork      = null;
  let replayScore        = 0;

  let duelMode           = false;
  let duelNetworks       = [null, null];
  let duelResults        = [null, null];
  let duelPhase          = 0;   // 0=idle 1=runA 2=runB 3=done

  let genBestHistory     = [];
  let allTimeBestHist    = [];

  let currentScore  = 0;
  let lastAction    = 'waiting...';
  let jumpOut       = 0;
  let duckOut       = 0;
  let isDucking     = false;
  let lastInputs    = [0, 0, 0, 0, 0, 0, 0, 0];
  let lastActivations = null;

  // Fitness shaping accumulators (reset on each run)
  let speedIntegral      = 0;
  let dodgeBonus         = 0;
  let lastObstacleXPos   = Infinity;

  // Vertical velocity tracking
  let lastDinoY = 0;
  let dinoVelY  = 0;

  // Real game scores per individual for display (separate from shaped fitness)
  let displayScores    = new Array(ga.populationSize).fill(0);
  let rawAllTimeBest   = 0;   // best raw game score ever (for chart/legend)
  let crashedAt          = null;
  let forcedRunner       = null;
  let runnerLogged       = false;
  let pendingStop        = false;
  let _lastGenPanelFull  = 0;   // timestamp of last full genPanel rebuild

  // ── Runner detection ───────────────────────────────────────────────────────

  function getRunner() {
    if (forcedRunner) return forcedRunner;
    if (window.Runner && typeof Runner.getInstance === 'function') {
      const r = Runner.getInstance();
      if (r) return r;
    }
    if (window.Runner && Runner.instance_) return Runner.instance_;
    if (window.Runner && typeof Runner.currentSpeed === 'number') return Runner;
    const canvas = document.querySelector('canvas');
    if (canvas && canvas.runner) return canvas.runner;
    return null;
  }

  // ── Score helper ───────────────────────────────────────────────────────────

  function getDisplayScore(runner) {
    if (runner.distanceMeter && runner.distanceMeter.digits && runner.distanceMeter.digits.length) {
      return parseInt(runner.distanceMeter.digits.join(''), 10) || 0;
    }
    return Math.ceil(runner.distanceRan * 0.025);
  }

  // ── Input vector ───────────────────────────────────────────────────────────

  function buildInputs(runner) {
    const obstacles = runner.horizon.obstacles;
    let dist = 1, width = 0, height = 0, type = 0, obsY = 0;
    if (obstacles && obstacles.length > 0) {
      const ahead = obstacles.filter(o => o.xPos + (o.width || 0) > DINO_X);
      if (ahead.length > 0) {
        const obs = ahead.reduce((a, b) => (b.xPos < a.xPos ? b : a));
        dist   = Math.max(0, obs.xPos - DINO_X) / CANVAS_WIDTH;
        width  = (obs.width || 0) / CANVAS_WIDTH;
        height = (obs.typeConfig.height || 0) / CANVAS_HEIGHT;
        type   = (obs.typeConfig.type === 'PTERODACTYL' || (obs.typeConfig.numFrames || 1) > 1) ? 1 : 0;
        obsY   = obs.yPos / CANVAS_HEIGHT;
      }
    }
    // Vertical velocity: maps [-MAX_VEL, +MAX_VEL] → [0, 1]
    const normVelY = Math.min(1, Math.max(0, (dinoVelY + MAX_VEL) / (2 * MAX_VEL)));
    return [dist, width, height, type, obsY, runner.currentSpeed / MAX_SPEED, runner.tRex.yPos / CANVAS_HEIGHT, normVelY];
  }

  // ── Active network (accounts for replay / duel) ───────────────────────────

  function getActiveNetwork() {
    if (replayMode && replayNetwork) return replayNetwork;
    if (duelMode) {
      if (duelPhase === 1 && duelNetworks[0]) return duelNetworks[0];
      if (duelPhase === 2 && duelNetworks[1]) return duelNetworks[1];
    }
    return ga.getCurrentNetwork();
  }

  // ── Game actions ───────────────────────────────────────────────────────────

  function gameJump(runner) {
    if (!runner.tRex.jumping) { runner.tRex.setDuck(false); runner.tRex.startJump(runner.currentSpeed); }
  }
  function gameDuck(runner, enable) { runner.tRex.setDuck(enable); }
  function gameRestart(runner) {
    if (typeof runner.restart === 'function') runner.restart();
    else document.dispatchEvent(new KeyboardEvent('keydown', { keyCode: 32, which: 32, bubbles: true }));
  }

  let lastStartAttempt = 0;
  function gameStart() {
    const now = Date.now();
    if (now - lastStartAttempt < 1000) return;
    lastStartAttempt = now;
    document.dispatchEvent(new KeyboardEvent('keydown', { keyCode: 32, which: 32, bubbles: true }));
  }

  // ── Main loop ──────────────────────────────────────────────────────────────

  function tick() {
    // Duel complete — stay on results screen until user clicks End Duel
    if (duelMode && duelPhase === 3) {
      updateGenPanel();
      return;
    }

    const runner = getRunner();

    if (!runner) {
      if (!overlay) return;
      const hasClass = typeof Runner !== 'undefined';
      const keys     = hasClass ? Object.keys(Runner).slice(0, 6).join(', ') : 'N/A';
      overlay.innerHTML = [
        '<b style="color:#fff">DINO BOT</b>', '',
        '<span style="color:#ff4">&#9654; Press SPACE once to start</span>',
        '  then bot takes over', '',
        '<span style="color:#555">Runner class : ' + (hasClass ? '<span style="color:#0f0">ok</span>' : '<span style="color:#f44">missing</span>') + '</span>',
        '<span style="color:#555">static keys  : ' + keys + '</span>', '',
        'If stuck, run in console:',
        '<span style="color:#39ff14">DinoBot.forceRunner(Runner.getInstance())</span>',
      ].join('<br>');
      return;
    }

    if (!runnerLogged) {
      runnerLogged = true;
      console.log('[DinoBot] Runner found:', runner);
    }

    const isPlaying = runner.playing || runner.activated;
    if (!isPlaying && !runner.crashed) {
      lastAction = 'starting...';
      updateOverlay(); updateGenPanel();
      gameStart();
      return;
    }

    if (runner.crashed || runner.crashed_) {
      if (!crashedAt) {
        const score = getDisplayScore(runner);
        lastAction  = 'DEAD';
        crashedAt   = Date.now();

        if (replayMode) {
          replayScore = score;
          replayMode  = false;
          replayNetwork = null;
          lastActivations = null;
          // Don't call ga.recordFitness() — just exit replay and continue training
        } else if (duelMode) {
          if (duelPhase === 1) { duelResults[0] = score; duelPhase = 2; }
          else if (duelPhase === 2) { duelResults[1] = score; duelPhase = 3; }
        } else {
          displayScores[ga.currentIndex] = score;
          if (score > rawAllTimeBest) rawAllTimeBest = score;
          const fitness = Math.max(0, score + speedIntegral * 0.5 + dodgeBonus);
          ga.recordFitness(fitness, score);
        }
        currentScore = 0; jumpOut = 0; duckOut = 0; isDucking = false;
        speedIntegral = 0; dodgeBonus = 0; lastObstacleXPos = Infinity;
        lastDinoY = 0; dinoVelY = 0;
      }
      updateOverlay(); updateGenPanel(); drawBottomLeftCanvas();
      if (Date.now() - crashedAt >= CRASH_PAUSE_MS) {
        if (pendingStop) {
          pendingStop = false;
          clearInterval(intervalId); intervalId = null;
          const r = getRunner(); if (r) gameDuck(r, false);
          updateGenPanel();
          return;  // leave crashedAt non-null so Resume skips re-recording fitness
        }
        if (!(duelMode && duelPhase === 3)) gameRestart(runner);
      }
      return;
    }

    // Game is running normally — safe to clear crash state now
    crashedAt = null;

    currentScore  = getDisplayScore(runner);

    // Vertical velocity (position delta between ticks)
    const rawDinoY = runner.tRex.yPos;
    dinoVelY = rawDinoY - lastDinoY;
    lastDinoY = rawDinoY;

    lastInputs    = buildInputs(runner);
    const fwd     = getActiveNetwork().forwardWithActivations(lastInputs);
    lastActivations = fwd;
    const [jOut, dOut] = fwd.output;
    jumpOut = jOut; duckOut = dOut;

    if (runner.tRex.jumping) {
      // Already airborne — trigger speed drop for a fast fall if the network wants it
      if (duckOut > jumpOut && duckOut > 0.3 && !runner.tRex.speedDrop) {
        runner.tRex.setSpeedDrop(); lastAction = 'FALL';
      } else {
        lastAction = 'JUMP';
      }
      if (isDucking) { gameDuck(runner, false); isDucking = false; }
    } 
    else if (jumpOut > duckOut && jumpOut > 0.3) {
      if (isDucking) { gameDuck(runner, false); isDucking = false; }
      gameJump(runner); lastAction = 'JUMP';
    } else if (duckOut > jumpOut && duckOut > 0.3) {
      if (!isDucking) { gameDuck(runner, true); isDucking = true; }
      lastAction = 'DUCK';
    } else {
      if (isDucking) { gameDuck(runner, false); isDucking = false; }
      lastAction = 'run';
    }

    // ── Fitness shaping accumulators ─────────────────────────────────────────
    if (!replayMode && !duelMode) {
      // Speed-weighted survival bonus
      speedIntegral += (runner.currentSpeed / MAX_SPEED);

      // Action-based reward: reward the correct action that got the dino past each obstacle
      const obstacles = runner.horizon.obstacles;
      const aheadObs = obstacles && obstacles.length > 0
        ? obstacles.filter(o => o.xPos + (o.width || 0) > DINO_X)
        : [];
      const nearestObs = aheadObs.length > 0
        ? aheadObs.reduce((a, b) => (b.xPos < a.xPos ? b : a))
        : null;
      if (nearestObs) {
        if (lastObstacleXPos > DINO_X && nearestObs.xPos <= DINO_X) {
          // Obstacle just cleared — reward only the action that caused it
          const isPtero = (nearestObs.typeConfig.type === 'PTERODACTYL' || (nearestObs.typeConfig.numFrames || 1) > 1);
          if (!isPtero) {
            if (runner.tRex.jumping) dodgeBonus += 100;           // jumped over cactus
          } else if (nearestObs.yPos > CANVAS_HEIGHT * 0.5) {
            if (isDucking) dodgeBonus += 100;                     // ducked under low bird
          } else {
            if (!runner.tRex.jumping && !isDucking) dodgeBonus += 75; // ran under high bird
          }
        }
        lastObstacleXPos = nearestObs.xPos;
      } else {
        lastObstacleXPos = Infinity;
      }

    }

    updateOverlay(); tickGenPanel(); drawBottomLeftCanvas();
  }

  // Lightweight per-tick genPanel update: only rebuilds the full panel every 400 ms
  // so header buttons stay stable long enough to be clicked.
  function tickGenPanel() {
    const now = Date.now();
    if (now - _lastGenPanelFull >= 400) {
      _lastGenPanelFull = now;
      updateGenPanel();
    } else {
      // just patch the live score bar of the current individual in-place
      const bar = genPanel && genPanel.querySelector('[data-live-bar]');
      if (bar) {
        const scores  = displayScores.slice(0, ga.currentIndex);
        const maxScore = Math.max(...scores, currentScore, 1);
        const pct = Math.min(100, Math.round((currentScore / maxScore) * 100));
        bar.style.width = pct + '%';
      }
      const scoreEl = genPanel && genPanel.querySelector('[data-live-score]');
      if (scoreEl) scoreEl.textContent = String(currentScore).padStart(4, '\u00a0');
    }
  }

  // ── Main HUD (top-right) ──────────────────────────────────────────────────

  function bar(v, len = 10) {
    const f = Math.round(Math.max(0, Math.min(1, v)) * len);
    return '[' + '|'.repeat(f) + '.'.repeat(len - f) + '] ' + Math.round(v * 100) + '%';
  }

  function actionColor(a) {
    return a === 'JUMP' ? '#ffff00' : a === 'DUCK' ? '#00aaff' : a === 'FALL' ? '#ff9900' : a === 'DEAD' ? '#ff4444' : '#39ff14';
  }

  function createOverlay() {
    overlay = document.createElement('div');
    Object.assign(overlay.style, {
      position: 'fixed', top: '10px', right: '10px',
      background: '#111', color: '#39ff14',
      fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.9',
      padding: '10px 14px', borderRadius: '6px', border: '1px solid #333',
      zIndex: '99999', pointerEvents: 'none', userSelect: 'none', minWidth: '240px',
    });
    document.body.appendChild(overlay);
    updateOverlay();
  }

  function updateOverlay() {
    if (!overlay) return;
    const s       = ga.getStats();
    const obsType = lastInputs[3] > 0.5 ? 'pterodactyl' : 'cactus';

    let modeBadge = '';
    if (replayMode) {
      modeBadge = ' <span style="color:#ff8c00;font-weight:bold">[REPLAY]</span>';
    } else if (duelMode) {
      const label = duelPhase === 1 ? 'DUEL-A' : duelPhase === 2 ? 'DUEL-B' : 'DUEL';
      modeBadge = ' <span style="color:#00ffff;font-weight:bold">[' + label + ']</span>';
    }

    const mutStr = (s.mutationRate * 100).toFixed(1) + '%' + (s.stagnantGens > 0 ? ' (+' + s.stagnantGens + ' stg)' : '');

    overlay.innerHTML = [
      '<b style="color:#fff;font-size:13px">DINO BOT' + modeBadge + '</b>',
      '<span style="color:#666;font-size:11px">mut: ' + mutStr + '</span>',
      '',
      '<span style="color:#aaa">── INPUTS ──────────────────</span>',
      ...INPUT_LABELS.map((l, i) => l + ' ' + bar(lastInputs[i]) + (i === 3 ? ' ' + obsType : '')),
      '',
      '<span style="color:#aaa">── OUTPUTS ─────────────────</span>',
      'jump  ' + bar(jumpOut),
      'duck  ' + bar(duckOut),
      '',
      'Action: <b style="color:' + actionColor(lastAction) + '">' + lastAction + '</b>',
    ].join('<br>');
  }

  // ── Competition mode ──────────────────────────────────────────────────────

  function toggleCompetition() {
    if (competitionActive) { exitCompetition(); return; }
    competitionActive  = true;
    competitionDone    = false;
    competitionResults = null;
    ga.competitionMode = true;
    ga.currentIndex    = 0;
    ga.fitnesses       = new Array(ga.populationSize).fill(0);
    updateGenPanel();
    if (!intervalId) start();
  }

  function exitCompetition() {
    competitionActive  = false;
    competitionDone    = false;
    competitionResults = null;
    ga.competitionMode = false;
    ga.currentIndex    = 0;
    ga.fitnesses       = new Array(ga.populationSize).fill(0);
    updateGenPanel();
    if (!intervalId) start();
  }

  // ── Replay mode ────────────────────────────────────────────────────────────

  function enterReplayMode(nn) {
    replayMode      = true;
    replayNetwork   = nn;
    replayScore     = 0;
    lastActivations = null;
    updateOverlay(); updateGenPanel();
    if (!intervalId) start();
  }

  // ── Duel mode ──────────────────────────────────────────────────────────────

  function _loadGenomeFromFile(callback) {
    const input = Object.assign(document.createElement('input'), { type: 'file', accept: '.json' });
    input.onchange = (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        try {
          const data = JSON.parse(ev.target.result);
          if (JSON.stringify(data.topology) !== JSON.stringify(TOPOLOGY)) {
            console.error('[DinoBot] Duel: topology mismatch.'); return;
          }
          const genome = data.type === 'population' ? data.genomes[0] : data.genome;
          callback(new NeuralNetwork(data.topology).fromGenome(genome), data);
        } catch (err) { console.error('[DinoBot] Duel: failed to load file:', err); }
      };
      reader.readAsText(file);
    };
    input.click();
  }

  function startDuel() {
    console.log('[DinoBot] Duel: select Genome A...');
    _loadGenomeFromFile((nnA, dataA) => {
      console.log('[DinoBot] Duel: Genome A loaded (gen ' + dataA.generation + ', score ' + Math.round(dataA.fitness || 0) + '). Select Genome B...');
      _loadGenomeFromFile((nnB, dataB) => {
        console.log('[DinoBot] Duel: Genome B loaded (gen ' + dataB.generation + ', score ' + Math.round(dataB.fitness || 0) + '). Starting duel!');
        duelNetworks = [nnA, nnB];
        duelResults  = [null, null];
        duelPhase    = 1;
        duelMode     = true;
        updateGenPanel(); updateOverlay();
        if (!intervalId) start();
      });
    });
  }

  function exitDuelMode() {
    const runner = getRunner();
    if (runner) gameRestart(runner);
    duelMode     = false;
    duelNetworks = [null, null];
    duelResults  = [null, null];
    duelPhase    = 0;
    updateGenPanel();
    updateOverlay();
  }

  // ── Generation performance panel (top-left) ───────────────────────────────

  const BTN = (color) =>
    'background:' + color + ';color:#fff;border:none;border-radius:3px;' +
    'padding:2px 7px;cursor:pointer;font-family:monospace;font-size:11px;margin-left:4px';

  function createGenPanel() {
    genPanel = document.createElement('div');
    Object.assign(genPanel.style, {
      position: 'fixed', top: '10px', left: '10px',
      background: '#111', color: '#39ff14',
      fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.7',
      borderRadius: '6px', border: '1px solid #333',
      zIndex: '99999', userSelect: 'none', minWidth: '260px',
    });

    genPanel.addEventListener('click', (e) => {
      const btn = e.target.closest('button');
      if (btn) {
        if (btn.dataset.action === 'savegen')    ga.exportPopulationToFile();
        if (btn.dataset.action === 'newgen') {
          if (window.confirm('Reset all training and start a completely new generation?')) {
            competitionActive = false; competitionDone = false; competitionResults = null;
            duelMode = false; duelPhase = 0;
            genBestHistory = []; allTimeBestHist = [];
            ga.competitionMode = false;
            ga.reset();
          }
        }
        if (btn.dataset.action === 'competition') toggleCompetition();
        if (btn.dataset.action === 'resumetrain') exitCompetition();
        if (btn.dataset.action === 'endduel')     exitDuelMode();
        if (btn.dataset.action === 'stopbot') {
          pendingStop = true;
          updateGenPanel();
        }
        if (btn.dataset.action === 'cancelstop') {
          pendingStop = false;
          updateGenPanel();
        }
        if (btn.dataset.action === 'startbot') {
          pendingStop = false;
          if (!intervalId) { intervalId = setInterval(tick, POLL_MS); }
          updateGenPanel();
        }
        return;
      }
      if (e.target.closest('[data-toggle]')) { genExpanded = !genExpanded; updateGenPanel(); }
    });

    ga.onEvolve = () => {
      // Capture raw scores before reset — these are the real game scores for the chart
      const rawGenBest = Math.round(Math.max(...displayScores, 0));
      displayScores = new Array(ga.populationSize).fill(0);
      genBestHistory.push(rawGenBest);
      allTimeBestHist.push(rawAllTimeBest);
      // Stamp the just-created checkpoint with the true session high score up to this gen
      if (ga.autoSaveEvery > 0 && ga.generation % ga.autoSaveEvery === 0 && ga.checkpoints.length > 0) {
        ga.checkpoints[0].rawScore = rawAllTimeBest;
      }
      updateGenPanel();
      updateBottomLeftPanel();
    };

    ga.onGenerationComplete = () => {
      competitionDone    = true;
      competitionResults = displayScores
        .map((score, i) => ({ individual: i + 1, score }))
        .sort((a, b) => b.score - a.score);
      clearInterval(intervalId);
      intervalId = null;
      updateGenPanel();
    };

    document.body.appendChild(genPanel);
    updateGenPanel();
  }

  function updateGenPanel() {
    _lastGenPanelFull = Date.now();
    if (!genPanel) return;
    const s     = ga.getStats();
    const arrow = genExpanded ? '▲' : '▼';

    const scores   = displayScores.slice(0, ga.currentIndex);
    const maxScore = Math.max(...scores, currentScore, 1);

    // ── Header ──────────────────────────────────────────────────────────────
    let modeLabel = '';
    if (competitionActive) modeLabel = '<span style="color:#ff8c00"> ⚔ COMPETITION</span>';
    else if (duelMode)     modeLabel = '<span style="color:#00ffff"> ⚔ DUEL</span>';
    else if (replayMode)   modeLabel = '<span style="color:#ff8c00"> ▶ REPLAY</span>';

    let stopBtnHtml = '';
    if (!intervalId) {
      stopBtnHtml = '<button data-action="startbot" style="' + BTN('#1a4a2a') + '">Resume</button>';
    } else if (pendingStop) {
      stopBtnHtml =
        '<button data-action="cancelstop" style="' + BTN('#4a4a00') + '">Stopping…</button>';
    } else {
      stopBtnHtml = '<button data-action="stopbot" style="' + BTN('#4a2a2a') + '">Stop</button>';
    }

    let rightBtns = '';
    if (duelMode) {
      rightBtns =
        stopBtnHtml +
        '<button data-action="endduel" style="' + BTN('#4a0000') + '">✕ End Duel</button>';
    } else {
      const compBtnTxt = competitionActive ? '✕ Exit' : '⚔ Compete';
      const compBtnClr = competitionActive ? '#6b2a1a' : '#4a3a00';
      rightBtns =
        stopBtnHtml +
        '<button data-action="newgen"      style="' + BTN('#3a1a1a') + '">+ New</button>' +
        '<button data-action="savegen"     style="' + BTN('#1a3a6b') + '">&#8595; Save</button>' +
        '<button data-action="competition" style="' + BTN(compBtnClr) + '">' + compBtnTxt + '</button>';
    }

    let html =
      '<div style="padding:8px 14px;border-bottom:1px solid #222;display:flex;align-items:center;gap:6px">' +
      '<span data-toggle style="flex:1;cursor:pointer"><b style="color:#fff">GEN ' + s.generation + modeLabel + '</b></span>' +
      rightBtns +
      '<span data-toggle style="color:#666;cursor:pointer">' + arrow + '</span>' +
      '</div>';

    if (!genExpanded) { genPanel.innerHTML = html; return; }

    // ── Duel results ────────────────────────────────────────────────────────
    if (duelMode && duelPhase === 3 && duelResults[0] !== null && duelResults[1] !== null) {
      const [sA, sB]  = duelResults;
      const aWins     = sA >= sB;
      const wColor    = '#ffd700';
      const lColor    = '#555';
      html += '<div style="padding:8px 14px">';
      html += '<div style="color:#00ffff;font-size:11px;border-bottom:1px solid #222;padding-bottom:4px;margin-bottom:6px">DUEL RESULTS</div>';
      html +=
        '<div style="display:flex;align-items:center;gap:8px;padding:4px 0">' +
        '<span style="color:#aaa;width:70px">Genome A</span>' +
        '<b style="color:' + (aWins ? wColor : lColor) + ';width:50px;text-align:right">' + sA + '</b>' +
        '<span style="color:' + (aWins ? wColor : lColor) + ';font-size:16px">' + (aWins ? ' 🏆' : '') + '</span>' +
        '</div>';
      html +=
        '<div style="display:flex;align-items:center;gap:8px;padding:4px 0">' +
        '<span style="color:#aaa;width:70px">Genome B</span>' +
        '<b style="color:' + (!aWins ? wColor : lColor) + ';width:50px;text-align:right">' + sB + '</b>' +
        '<span style="color:' + (!aWins ? wColor : lColor) + ';font-size:16px">' + (!aWins ? ' 🏆' : '') + '</span>' +
        '</div>';
      html += '</div>';
      genPanel.innerHTML = html;
      return;
    }

    // ── Competition results ─────────────────────────────────────────────────
    if (competitionDone && competitionResults) {
      const scrs  = competitionResults.map(r => r.score);
      const best  = scrs[0];
      const avg   = Math.round(scrs.reduce((a, b) => a + b, 0) / scrs.length);
      const worst = scrs[scrs.length - 1];

      html += '<div style="padding:4px 0">';
      html += '<div style="padding:4px 14px;color:#ff8c00;font-size:11px;border-bottom:1px solid #222">' +
        'Best: <b style="color:#fff">' + best + '</b> &nbsp; ' +
        'Avg: <b style="color:#aaa">' + avg + '</b> &nbsp; ' +
        'Worst: <b style="color:#555">' + worst + '</b>' +
        '</div>';
      competitionResults.forEach((r, rank) => {
        const pct       = Math.min(100, Math.round((r.score / (best || 1)) * 100));
        const rankColor = rank === 0 ? '#ffd700' : rank === 1 ? '#aaa' : rank === 2 ? '#cd7f32' : '#555';
        html +=
          '<div style="padding:3px 14px;display:flex;align-items:center;gap:6px">' +
          '<span style="color:' + rankColor + ';width:20px;text-align:right">#' + (rank + 1) + '</span>' +
          '<span style="color:#555;width:28px;text-align:right;font-size:10px">i' + r.individual + '</span>' +
          '<span style="color:#39ff14;width:34px;text-align:right">' + r.score + '</span>' +
          '<div style="flex:1;background:#1a1a1a;height:8px;border-radius:3px;overflow:hidden">' +
            '<div style="width:' + pct + '%;height:100%;background:#39ff14;border-radius:3px"></div>' +
          '</div>' +
          '</div>';
      });
      html += '<div style="padding:6px 14px;border-top:1px solid #222">' +
        '<button data-action="resumetrain" style="' + BTN('#1a6b2a') + ';width:100%;margin:0">▶ Resume Training</button>' +
        '</div>';
      html += '</div>';
      genPanel.innerHTML = html;
      return;
    }

    // ── Normal generation view ───────────────────────────────────────────────
    html += '<div style="padding:4px 0">';
    for (let i = 0; i < ga.populationSize; i++) {
      const isDone    = i < ga.currentIndex;
      const isCurrent = i === ga.currentIndex;
      const score     = isDone ? displayScores[i] : (isCurrent ? currentScore : 0);
      const pct       = Math.min(100, Math.round((score / maxScore) * 100));

      const numColor  = isDone ? '#39ff14' : isCurrent ? '#ffff00' : '#444';
      const barColor  = isDone ? '#1a6b2a' : isCurrent ? '#6b6b00' : '#1a1a1a';
      const fillColor = isDone ? '#39ff14' : isCurrent ? '#ffff00' : '#222';
      const label     = isCurrent ? '&#9654;' : isDone ? '&#10003;' : '&nbsp;';
      const scoreText = (isDone || isCurrent) ? String(score).padStart(4, '\u00a0') : '\u00a0---';

      const origin  = ga.evolutionLog && ga.evolutionLog.origins[i];
      const isElite = origin && origin.type === 'elite';
      const isCross = origin && origin.type === 'crossover';
      const badge   = origin
        ? (isElite
            ? '<span style="color:#39ff14;font-size:10px">[E]</span>'
            : isCross
              ? '<span style="color:#00ffff;font-size:10px">[X]</span>'
              : '<span style="color:#ff8c00;font-size:10px">[M]</span>')
        : '<span style="width:22px;display:inline-block"></span>';
      const from = origin
        ? (isCross
            ? '<span style="color:#444;font-size:10px">\u2190' + Math.round(origin.prevScoreA) + '\u00d7' + Math.round(origin.prevScoreB) + '</span>'
            : '<span style="color:#444;font-size:10px">\u2190' + Math.round(origin.prevScore) + '</span>')
        : '';

      html +=
        '<div style="padding:3px 14px;display:flex;align-items:center;gap:6px' + (isCurrent ? ';background:#1a1a00' : '') + '">' +
        '<span style="color:#555;width:20px;text-align:right">#' + (i + 1) + '</span>' +
        badge +
        '<span style="color:' + numColor + ';width:34px;text-align:right"' + (isCurrent ? ' data-live-score' : '') + '>' + scoreText + '</span>' +
        '<div style="flex:1;background:#1a1a1a;height:8px;border-radius:3px;overflow:hidden;border:1px solid ' + barColor + '">' +
          '<div ' + (isCurrent ? 'data-live-bar ' : '') + 'style="width:' + pct + '%;height:100%;background:' + fillColor + ';border-radius:3px;transition:width 0.1s"></div>' +
        '</div>' +
        '<span style="color:' + numColor + ';width:14px;text-align:center">' + label + '</span>' +
        from +
        '</div>';
    }
    html += '<div style="padding:4px 14px;color:#444;font-size:10px;border-top:1px solid #1a1a1a">' +
      '<span style="color:#39ff14">[E]</span> elite &nbsp; ' +
      '<span style="color:#ff8c00">[M]</span> mutated &nbsp; ' +
      '<span style="color:#00ffff">[X]</span> crossover' +
      '</div>';
    html += '</div>';

    genPanel.innerHTML = html;
  }

  // ── Checkpoint panel (bottom-right) ──────────────────────────────────────

  function createCheckpointPanel() {
    cpPanel = document.createElement('div');
    Object.assign(cpPanel.style, {
      position: 'fixed', bottom: '10px', right: '640px',
      background: '#111', color: '#39ff14',
      fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.7',
      borderRadius: '6px', border: '1px solid #333',
      zIndex: '99999', userSelect: 'none', minWidth: '300px',
    });

    cpPanel.addEventListener('click', (e) => {
      const btn = e.target.closest('button');
      if (btn) {
        const idx = parseInt(btn.dataset.idx, 10);
        if (btn.dataset.action === 'dl')     ga.downloadCheckpoint(idx);
        if (btn.dataset.action === 'load')   { ga.loadCheckpoint(idx); updateCheckpointPanel(); }
        if (btn.dataset.action === 'import') ga.importFromFile();
        return;
      }
      if (e.target.closest('[data-toggle]')) { cpExpanded = !cpExpanded; updateCheckpointPanel(); }
    });

    document.body.appendChild(cpPanel);
    updateCheckpointPanel();
  }

  function updateCheckpointPanel() {
    if (!cpPanel) return;
    const cpCount = ga.checkpoints.length;
    const arrow   = cpExpanded ? '▲' : '▼';

    let html =
      '<div style="padding:8px 14px;border-bottom:1px solid #222;display:flex;align-items:center;gap:8px">' +
      '<span data-toggle style="flex:1;cursor:pointer"><b style="color:#fff">CHECKPOINTS</b> <span style="color:#555">(auto every 5 gens)</span></span>' +
      '<button data-action="import" style="' + BTN('#555') + '">&#8593; Import</button>' +
      '<span data-toggle style="color:#666;cursor:pointer">' + cpCount + ' ' + arrow + '</span>' +
      '</div>';

    if (cpExpanded) {
      if (cpCount === 0) {
        html += '<div style="padding:8px 14px;color:#555">No checkpoints yet...</div>';
      } else {
        ga.checkpoints.forEach((cp, i) => {
          html +=
            '<div style="padding:5px 14px;border-bottom:1px solid #1a1a1a;display:flex;align-items:center">' +
            '<span style="flex:1;color:#aaa">' +
              'Gen <b style="color:#fff">' + String(cp.generation).padStart(3, '\u00a0') + '</b>' +
              ' &nbsp;|&nbsp; <b style="color:#39ff14">' + (cp.rawScore || Math.round(cp.fitness)) + '</b> pts' +
              ' &nbsp;|&nbsp; <span style="color:#555">' + cp.timestamp + '</span>' +
            '</span>' +
            '<button data-action="dl"   data-idx="' + i + '" style="' + BTN('#1a6b2a') + '">&#8595; Save</button>' +
            '<button data-action="load" data-idx="' + i + '" style="' + BTN('#1a3a6b') + '">&#9654; Load</button>' +
            '</div>';
        });
      }
    }

    cpPanel.innerHTML = html;
  }

  // ── Hall of Fame panel (bottom: right of training stats) ──────────────────

  function createHofPanel() {
    hofPanel = document.createElement('div');
    Object.assign(hofPanel.style, {
      position: 'fixed', bottom: '10px', right: '330px',
      background: '#111', color: '#39ff14',
      fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.7',
      borderRadius: '6px', border: '1px solid #333',
      zIndex: '99999', userSelect: 'none', minWidth: '300px',
    });

    hofPanel.addEventListener('click', (e) => {
      const btn = e.target.closest('button');
      if (btn) {
        const idx = parseInt(btn.dataset.idx, 10);
        if (btn.dataset.action === 'hofDl') ga.downloadHallOfFame(idx);
        if (btn.dataset.action === 'hofReplay') {
          const entry = ga.hallOfFame[idx];
          if (entry) enterReplayMode(new NeuralNetwork(entry.topology).fromGenome(entry.genome));
        }
        return;
      }
      if (e.target.closest('[data-toggle]')) { hofExpanded = !hofExpanded; updateHofPanel(); }
    });

    ga.onCheckpoint = () => { updateCheckpointPanel(); updateHofPanel(); };

    document.body.appendChild(hofPanel);
    updateHofPanel();
  }

  function updateHofPanel() {
    if (!hofPanel) return;
    const hofCount = ga.hallOfFame.length;
    const arrow    = hofExpanded ? '▲' : '▼';
    const medals   = ['🥇', '🥈', '🥉', '#4', '#5'];

    let html =
      '<div style="padding:8px 14px;border-bottom:1px solid #222;display:flex;align-items:center;gap:8px">' +
      '<span data-toggle style="flex:1;cursor:pointer"><b style="color:#fff">HALL OF FAME</b> <span style="color:#555">(top 5 all-time)</span></span>' +
      '<span data-toggle style="color:#666;cursor:pointer">' + hofCount + ' ' + arrow + '</span>' +
      '</div>';

    if (hofExpanded) {
      if (hofCount === 0) {
        html += '<div style="padding:8px 14px;color:#555">No entries yet — keep training!</div>';
      } else {
        ga.hallOfFame.forEach((entry, i) => {
          const rankColor = i === 0 ? '#ffd700' : i === 1 ? '#c0c0c0' : i === 2 ? '#cd7f32' : '#555';
          html +=
            '<div style="padding:5px 14px;border-bottom:1px solid #1a1a1a;display:flex;align-items:center;gap:6px">' +
            '<span style="color:' + rankColor + ';width:18px">' + medals[i] + '</span>' +
            '<span style="flex:1;color:#aaa">' +
              '<b style="color:#39ff14">' + entry.rawScore + '</b> pts' +
              ' &nbsp;·&nbsp; <span style="color:#555">gen ' + entry.generation + '</span>' +
              ' &nbsp;·&nbsp; <span style="color:#333">' + entry.timestamp + '</span>' +
            '</span>' +
            '<button data-action="hofDl"     data-idx="' + i + '" style="' + BTN('#1a6b2a') + '">&#8595;</button>' +
            '</div>';
        });
      }
    }

    hofPanel.innerHTML = html;
  }

  // ── Bottom-left panel (fitness chart / network diagram) ───────────────────

  function createBottomLeftPanel() {
    blPanel = document.createElement('div');
    Object.assign(blPanel.style, {
      position: 'fixed', bottom: '10px', right: '10px',
      background: '#111', color: '#39ff14',
      fontFamily: 'monospace', fontSize: '12px',
      borderRadius: '6px', border: '1px solid #333',
      zIndex: '99999', userSelect: 'none', width: '310px',
    });

    blHeader = document.createElement('div');
    blHeader.style.cssText =
      'padding:8px 14px;border-bottom:1px solid #222;display:flex;align-items:center;gap:6px';

    const blBody = document.createElement('div');
    blCanvas = document.createElement('canvas');
    blCanvas.width  = 276;
    blCanvas.height = 170;
    blCanvas.style.cssText = 'display:block;padding:6px 8px';
    blBody.appendChild(blCanvas);

    blPanel.addEventListener('click', (e) => {
      const tabBtn = e.target.closest('button[data-tab]');
      if (tabBtn) {
        blTab = tabBtn.dataset.tab;
        if (!blExpanded) { blExpanded = true; blBody.style.display = 'block'; }
        updateBottomLeftPanel();
        return;
      }
      if (e.target.closest('[data-collapse]')) {
        blExpanded = !blExpanded;
        blBody.style.display = blExpanded ? 'block' : 'none';
        updateBottomLeftPanel();
      }
    });

    blPanel.appendChild(blHeader);
    blPanel.appendChild(blBody);
    document.body.appendChild(blPanel);
    updateBottomLeftPanel();
  }

  function updateBottomLeftPanel() {
    if (!blPanel || !blHeader || !blCanvas) return;
    const arrow      = blExpanded ? '▲' : '▼';
    const fitActive  = blTab === 'fitness';
    const netActive  = blTab === 'network';
    const origActive = blTab === 'origins';
    blHeader.innerHTML =
      '<span style="flex:1"><b style="color:#fff">TRAINING STATS</b></span>' +
      '<button data-tab="fitness" style="' + BTN(fitActive  ? '#2a5a2a' : '#1a2a1a') + '">Fitness</button>' +
      '<button data-tab="network" style="' + BTN(netActive  ? '#2a5a2a' : '#1a2a1a') + '">Network</button>' +
      '<button data-tab="origins" style="' + BTN(origActive ? '#2a5a2a' : '#1a2a1a') + '">Origins</button>' +
      '<span data-collapse style="color:#666;margin-left:6px;cursor:pointer">' + arrow + '</span>';
    drawBottomLeftCanvas();
  }

  function drawBottomLeftCanvas() {
    if (!blCanvas || !blExpanded) return;
    if (blTab === 'fitness')      drawFitnessChart(blCanvas);
    else if (blTab === 'network') drawNetworkDiagram(blCanvas);
    else                          drawOriginsChart(blCanvas);
  }

  function drawFitnessChart(canvas) {
    const ctx = canvas.getContext('2d');
    const W   = canvas.width;
    const H   = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, W, H);

    if (genBestHistory.length < 1) {
      ctx.fillStyle = '#444';
      ctx.font = '11px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for first generation...', W / 2, H / 2);
      return;
    }

    const pad    = { left: 38, right: 8, top: 18, bottom: 20 };
    const cW     = W - pad.left - pad.right;
    const cH     = H - pad.top - pad.bottom;
    const n      = genBestHistory.length;
    const maxVal = Math.max(...allTimeBestHist, 1);

    const yPos   = (v) => pad.top + cH - (v / maxVal) * cH;

    // Bar width with a small gap
    const barW   = Math.max(2, Math.floor(cW / n) - 1);
    const barX   = (i) => pad.left + (i / n) * cW;

    // Y-axis grid lines + labels
    ctx.font      = '9px monospace';
    ctx.textAlign = 'right';
    for (let t = 0; t <= 4; t++) {
      const val = Math.round((maxVal / 4) * (4 - t));
      const yy  = pad.top + (t / 4) * cH;
      ctx.strokeStyle = '#1a1a1a'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.left, yy); ctx.lineTo(W - pad.right, yy); ctx.stroke();
      ctx.fillStyle = '#444';
      ctx.fillText(val, pad.left - 3, yy + 3);
    }

    // Bars — one per generation + inline score labels
    // Decide label frequency so text never overlaps (min ~22 px per label)
    const labelEvery = Math.max(1, Math.ceil(22 / Math.max(1, barW)));

    genBestHistory.forEach((v, i) => {
      const x      = barX(i);
      const isLast = i === n - 1;
      const bTop   = yPos(v);
      const barH   = Math.max(1, cH - (bTop - pad.top));

      // Bar fill
      ctx.fillStyle = isLast ? '#484848' : '#252525';
      ctx.fillRect(x, bTop, barW, barH);

      // Score label
      if (v === 0) return;
      const showLabel = isLast || (i % labelEvery === 0);
      if (!showLabel) return;

      ctx.save();
      ctx.font = isLast ? 'bold 9px monospace' : '8px monospace';
      ctx.textAlign = 'center';
      const cx = x + barW / 2;

      if (barW >= 22 && barH >= 14) {
        // Inside the bar — centered vertically
        ctx.fillStyle = isLast ? 'rgba(255,255,255,0.85)' : 'rgba(255,255,255,0.3)';
        ctx.fillText(v, cx, bTop + barH / 2 + 3);
      } else if (barW >= 10) {
        // Above the bar
        ctx.fillStyle = isLast ? '#fff' : '#555';
        ctx.fillText(v, cx, bTop - 3);
      } else {
        // Very narrow bars — rotate label, only for last bar
        if (!isLast) { ctx.restore(); return; }
        ctx.translate(cx, bTop - 4);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'left';
        ctx.fillStyle = '#fff';
        ctx.fillText(v, 2, 0);
      }
      ctx.restore();
    });

    // All-time best as a step function (green) — only ever goes up
    ctx.strokeStyle = '#39ff14';
    ctx.lineWidth   = 2;
    ctx.beginPath();
    allTimeBestHist.forEach((v, i) => {
      const x = barX(i);
      if (i === 0) {
        ctx.moveTo(x, yPos(v));
      } else {
        ctx.lineTo(x, yPos(v));   // horizontal: carry previous value to this gen's x
      }
      ctx.lineTo(x + barW, yPos(v)); // extend to right edge of this bar
    });
    ctx.stroke();

    // X-axis: generation numbers
    ctx.font = '9px monospace'; ctx.fillStyle = '#444'; ctx.textAlign = 'center';
    ctx.fillText('1', barX(0) + barW / 2, H - 5);
    if (n > 1) ctx.fillText(n, barX(n - 1) + barW / 2, H - 5);
    if (n > 4) {
      const mid = Math.floor(n / 2);
      ctx.fillText(mid + 1, barX(mid) + barW / 2, H - 5);
    }

    // Legend with current values — scores shown here, not floating on the chart
    const atb     = allTimeBestHist[n - 1];
    ctx.font = '9px monospace'; ctx.textAlign = 'left';
    ctx.fillStyle = '#39ff14';
    ctx.fillText('── record: ' + atb, pad.left, pad.top - 5);
  }

  function drawNetworkDiagram(canvas) {
    const ctx = canvas.getContext('2d');
    const W   = canvas.width;
    const H   = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, W, H);

    const nn  = getActiveNetwork();
    const act = lastActivations ? lastActivations.layerActivations : null;
    const top = nn.topology;  // [7, 8, 2]

    const layerX   = [42, W / 2, W - 42];
    const yTop     = 12;
    const yRange   = H - 24;

    // Node positions per layer
    const nodes = top.map((count, l) =>
      Array.from({ length: count }, (_, j) => ({
        x: layerX[l],
        y: count === 1
          ? yTop + yRange / 2
          : yTop + (j / (count - 1)) * yRange,
      }))
    );

    // Draw connections
    for (let l = 0; l < nn.weights.length; l++) {
      for (let j = 0; j < nn.weights[l].length; j++) {
        for (let k = 0; k < nn.weights[l][j].length; k++) {
          const w     = nn.weights[l][j][k];
          const alpha = Math.min(1.0, Math.max(0.05, Math.abs(w)));
          const color = w > 0
            ? 'rgba(57,255,20,'  + alpha.toFixed(2) + ')'
            : 'rgba(255,68,68,'  + alpha.toFixed(2) + ')';
          ctx.strokeStyle = color;
          ctx.lineWidth   = 0.7;
          ctx.beginPath();
          ctx.moveTo(nodes[l][k].x,     nodes[l][k].y);
          ctx.lineTo(nodes[l + 1][j].x, nodes[l + 1][j].y);
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    const R = 6;
    top.forEach((count, l) => {
      for (let j = 0; j < count; j++) {
        const { x, y } = nodes[l][j];
        const a = act ? act[l][j] : 0;
        // Lerp from #0a0a0a → #39ff14
        const r = Math.round(10 + 47  * a);
        const g = Math.round(10 + 245 * a);
        const b = Math.round(10 + 10  * a);

        ctx.fillStyle = '#0a0a0a';
        ctx.beginPath(); ctx.arc(x, y, R + 1, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.beginPath(); ctx.arc(x, y, R,     0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = '#333'; ctx.lineWidth = 0.5; ctx.stroke();
      }
    });

    // Input labels
    const inLabels = ['dist', 'wid', 'hgt', 'typ', 'oby', 'spd', 'dy', 'vly'];
    ctx.fillStyle  = '#555'; ctx.font = '8px monospace'; ctx.textAlign = 'right';
    nodes[0].forEach(({ x, y }, j) => ctx.fillText(inLabels[j], x - R - 3, y + 3));

    // Output labels
    const outLabels = ['jump', 'duck'];
    ctx.textAlign = 'left';
    nodes[top.length - 1].forEach(({ x, y }, j) => ctx.fillText(outLabels[j], x + R + 3, y + 3));
  }

  function drawOriginsChart(canvas) {
    const ctx = canvas.getContext('2d');
    const W   = canvas.width;
    const H   = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, W, H);

    const log = ga.evolutionLog;
    if (!log) {
      ctx.fillStyle = '#444';
      ctx.font = '11px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for first evolution...', W / 2, H / 2);
      return;
    }

    const { prevFitnesses, origins } = log;
    const n = origins.length;

    const pad  = { left: 52, right: 52, top: 18, bottom: 18 };
    const cH   = H - pad.top - pad.bottom;
    const step = cH / Math.max(n - 1, 1);

    const xPrev = pad.left;
    const xCur  = W - pad.right;

    // Prev gen: sorted by fitness descending → ranked y positions
    const prevRanked = prevFitnesses
      .map((f, i) => ({ f, i }))
      .sort((a, b) => b.f - a.f);
    const prevNodeY = new Array(n);
    prevRanked.forEach((e, rank) => { prevNodeY[e.i] = pad.top + rank * step; });

    // Cur gen: y by current index order
    const curNodeY = Array.from({ length: n }, (_, i) => pad.top + i * step);

    // Find previous individual index by fitness score (nearest match)
    function findPrevIdx(score) {
      let best = 0, bestD = Infinity;
      prevFitnesses.forEach((f, i) => {
        const d = Math.abs(f - score);
        if (d < bestD) { bestD = d; best = i; }
      });
      return best;
    }

    // Draw connection lines
    origins.forEach((origin, i) => {
      const cy = curNodeY[i];
      if (origin.type === 'elite') {
        const pi = findPrevIdx(origin.prevScore);
        ctx.strokeStyle = 'rgba(57,255,20,0.65)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(xPrev + 6, prevNodeY[pi]); ctx.lineTo(xCur - 6, cy); ctx.stroke();
      } else if (origin.type === 'offspring') {
        const pi = findPrevIdx(origin.prevScore);
        ctx.strokeStyle = 'rgba(255,140,0,0.45)';
        ctx.lineWidth = 0.8;
        ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(xPrev + 6, prevNodeY[pi]); ctx.lineTo(xCur - 6, cy); ctx.stroke();
      } else { // crossover
        const piA = findPrevIdx(origin.prevScoreA);
        const piB = findPrevIdx(origin.prevScoreB);
        ctx.strokeStyle = 'rgba(0,170,255,0.45)';
        ctx.lineWidth = 0.9;
        ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(xPrev + 6, prevNodeY[piA]); ctx.lineTo(xCur - 6, cy); ctx.stroke();
        ctx.strokeStyle = 'rgba(0,170,255,0.25)';
        ctx.lineWidth = 0.8;
        ctx.setLineDash([2, 3]);
        ctx.beginPath(); ctx.moveTo(xPrev + 6, prevNodeY[piB]); ctx.lineTo(xCur - 6, cy); ctx.stroke();
        ctx.setLineDash([]);
      }
    });

    // Draw prev gen dots + fitness labels
    const R = 4;
    prevRanked.forEach((e) => {
      const y = prevNodeY[e.i];
      ctx.fillStyle = '#1a3a1a';
      ctx.beginPath(); ctx.arc(xPrev, y, R, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#39ff14'; ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.arc(xPrev, y, R, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = '#444'; ctx.font = '8px monospace'; ctx.textAlign = 'right';
      ctx.fillText(Math.round(e.f), xPrev - R - 2, y + 3);
    });

    // Draw cur gen dots + type badges + scores
    origins.forEach((origin, i) => {
      const y         = curNodeY[i];
      const isDone    = i < ga.currentIndex;
      const isCurrent = i === ga.currentIndex;
      const typeColor = origin.type === 'elite'     ? '#39ff14'
                      : origin.type === 'crossover' ? '#00aaff'
                      :                               '#ff8c00';

      ctx.fillStyle = isDone || isCurrent ? typeColor : '#2a2a2a';
      ctx.beginPath(); ctx.arc(xCur, y, R, 0, Math.PI * 2); ctx.fill();
      if (isCurrent) {
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(xCur, y, R + 2, 0, Math.PI * 2); ctx.stroke();
      }

      // Type badge (left of dot)
      const badge = origin.type === 'elite' ? 'E' : origin.type === 'crossover' ? 'X' : 'M';
      ctx.fillStyle = isDone || isCurrent ? typeColor : '#333';
      ctx.font = 'bold 7px monospace'; ctx.textAlign = 'right';
      ctx.fillText(badge, xCur - R - 2, y + 3);

      // Score label (right of dot)
      const scoreVal = isDone ? displayScores[i] : (isCurrent ? currentScore : null);
      if (scoreVal !== null) {
        ctx.fillStyle = isDone ? '#aaa' : '#ffff00';
        ctx.font = '8px monospace'; ctx.textAlign = 'left';
        ctx.fillText(Math.round(scoreVal), xCur + R + 2, y + 3);
      }
    });

    // Column headers
    ctx.fillStyle = '#555'; ctx.font = '9px monospace'; ctx.textAlign = 'center';
    ctx.fillText('gen ' + (ga.generation - 1), xPrev, pad.top - 5);
    ctx.fillText('gen ' + ga.generation,       xCur,  pad.top - 5);

    // Footer legend
    ctx.font = '8px monospace';
    const ly = H - 4;
    ctx.fillStyle = '#39ff14'; ctx.textAlign = 'left';  ctx.fillText('E=elite',  pad.left + 8,      ly);
    ctx.fillStyle = '#ff8c00'; ctx.textAlign = 'center'; ctx.fillText('M=mut',   W / 2,             ly);
    ctx.fillStyle = '#00aaff'; ctx.textAlign = 'right';  ctx.fillText('X=cross', W - pad.right - 8, ly);
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  function start() {
    if (intervalId) { console.log('[DinoBot] Already running.'); return; }
    if (!overlay)   createOverlay();
    if (!genPanel)  createGenPanel();
    if (!cpPanel)   createCheckpointPanel();
    if (!hofPanel)  createHofPanel();
    if (!blPanel)   createBottomLeftPanel();
    intervalId = setInterval(tick, POLL_MS);
    console.log(
      '[DinoBot] Started.\n' +
      '  DinoBot.stop()                            — pause\n' +
      '  DinoBot.save()                            — download best genome\n' +
      '  DinoBot.load()                            — import saved file\n' +
      '  DinoBot.replay(DinoBot.ga.allTimeBest)    — watch best play without training\n' +
      '  DinoBot.duel()                            — load 2 genomes and compare them\n' +
      '  DinoBot.ga.reset()                        — wipe training\n' +
      '  DinoBot.forceRunner(Runner.getInstance()) — set runner manually'
    );
  }

  function stop() {
    clearInterval(intervalId); intervalId = null;
    const runner = getRunner();
    if (runner) gameDuck(runner, false);
    if (overlay)  { overlay.remove();  overlay   = null; }
    if (genPanel) { genPanel.remove(); genPanel  = null; }
    if (cpPanel)  { cpPanel.remove();  cpPanel   = null; }
    if (hofPanel) { hofPanel.remove(); hofPanel  = null; }
    if (blPanel)  { blPanel.remove();  blPanel   = null; blHeader = null; blCanvas = null; }
    console.log('[DinoBot] Stopped.');
  }

  start();

  return {
    start,
    stop,
    ga,
    save()         { ga.exportToFile(); },
    load()         { ga.importFromFile(); },
    replay(nn)     { enterReplayMode(nn instanceof NeuralNetwork ? nn : new NeuralNetwork(TOPOLOGY).fromGenome(nn)); },
    duel()         { startDuel(); },
    forceRunner(r) { forcedRunner = r; console.log('[DinoBot] Runner set to:', r); },
  };
})();
