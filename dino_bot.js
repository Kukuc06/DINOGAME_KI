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
      activation = next;
    }
    return activation;
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
 *   4. evolve() is called automatically — selects elites, produces offspring
 *      via tournament selection + Gaussian mutation.
 *   5. A checkpoint is stored in memory every autoSaveEvery generations.
 */
class GeneticAlgorithm {
  /**
   * @param {object}   opts
   * @param {number}   opts.populationSize   Individuals per generation (default 15)
   * @param {number}   opts.mutationRate     Probability of mutating each gene (default 0.15)
   * @param {number}   opts.mutationScale    Std-dev of Gaussian noise on mutation (default 0.4)
   * @param {number}   opts.eliteCount       Top individuals that survive unchanged (default 2)
   * @param {number[]} opts.topology         NN layer sizes, e.g. [7, 8, 2]
   * @param {number}   opts.autoSaveEvery    Store a checkpoint every N generations (default 5)
   * @param {number}   opts.maxCheckpoints   Max checkpoints kept in memory (default 10)
   */
  constructor({
    populationSize  = 15,
    mutationRate    = 0.15,
    mutationScale   = 0.4,
    eliteCount      = 2,
    topology        = [7, 8, 2],
    autoSaveEvery   = 5,
    maxCheckpoints  = 10,
  } = {}) {
    this.populationSize  = populationSize;
    this.mutationRate    = mutationRate;
    this.mutationScale   = mutationScale;
    this.eliteCount      = eliteCount;
    this.topology        = topology;
    this.autoSaveEvery   = autoSaveEvery;
    this.maxCheckpoints  = maxCheckpoints;

    this.generation         = 1;
    this.currentIndex       = 0;
    this.fitnesses          = new Array(populationSize).fill(0);
    this.allTimeBest        = null;
    this.allTimeBestFitness = -Infinity;
    this.checkpoints           = []; // in-memory list, newest first
    this.evolutionLog          = null; // set after each _evolve(): origins of current gen
    this.competitionMode       = false; // when true, generation completes without evolving
    this.onCheckpoint          = null; // called when checkpoint list changes
    this.onEvolve              = null; // called when a new generation starts
    this.onGenerationComplete  = null; // called when all individuals have played (competition mode)

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
  recordFitness(fitness) {
    this.fitnesses[this.currentIndex] = fitness;

    if (fitness > this.allTimeBestFitness) {
      this.allTimeBestFitness = fitness;
      this.allTimeBest = this.population[this.currentIndex].clone();
      this._saveToStorage();
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

    // Fill rest with mutated tournament winners
    while (next.length < this.populationSize) {
      const winner = this._tournamentSelect(ranked);
      const child  = winner.nn.clone();
      this._mutate(child);
      next.push(child);
      origins.push({ type: 'offspring', prevScore: winner.fitness });
    }

    this.population    = next;
    this.fitnesses     = new Array(this.populationSize).fill(0);
    this.currentIndex  = 0;
    this.evolutionLog  = { prevFitnesses, origins };

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

  // ── Checkpoints ────────────────────────────────────────────────────────────

  _addCheckpoint() {
    if (!this.allTimeBest) return;
    this.checkpoints.unshift({
      genome:     this.allTimeBest.toGenome(),
      topology:   this.topology,
      fitness:    Math.max(0, this.allTimeBestFitness),
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

  loadCheckpoint(index) {
    const cp = this.checkpoints[index];
    if (!cp) return;
    this._seedFromBest(new NeuralNetwork(cp.topology).fromGenome(cp.genome), cp.fitness, cp.generation);
    console.log('[DinoBot] Loaded checkpoint — gen ' + cp.generation + ', score ' + Math.round(cp.fitness));
    if (this.onCheckpoint) this.onCheckpoint();
    if (this.onEvolve)     this.onEvolve();
  }

  // ── File export ────────────────────────────────────────────────────────────

  /** Download the all-time best genome as a single-best JSON file. */
  exportToFile() {
    if (!this.allTimeBest) { console.warn('[DinoBot] No best genome yet.'); return; }
    this._download(
      { type: 'best', genome: this.allTimeBest.toGenome(), topology: this.topology,
        fitness: this.allTimeBestFitness, generation: this.generation },
      'dino_genome.json'
    );
    console.log('[DinoBot] Saved dino_genome.json (gen ' + this.generation + ', score ' + Math.round(this.allTimeBestFitness) + ')');
  }

  /** Download the entire current population so training can resume exactly. */
  exportPopulationToFile() {
    this._download(
      { type: 'population',
        genomes:    this.population.map(nn => nn.toGenome()),
        fitnesses:  this.fitnesses.slice(),
        topology:   this.topology,
        generation: this.generation,
        currentIndex: this.currentIndex,
        allTimeBestGenome:   this.allTimeBest ? this.allTimeBest.toGenome() : null,
        allTimeBestFitness:  this.allTimeBestFitness,
      },
      `dino_gen${this.generation}_population.json`
    );
    console.log('[DinoBot] Saved full population (gen ' + this.generation + ')');
  }

  /**
   * Open a file picker and load either a single-best or full-population file.
   * Detects the file type automatically via the `type` field.
   */
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
            // Restore full population
            this.population   = data.genomes.map(g => new NeuralNetwork(data.topology).fromGenome(g));
            this.fitnesses    = data.fitnesses.slice();
            this.generation   = data.generation;
            this.currentIndex = data.currentIndex || 0;
            if (data.allTimeBestGenome) {
              this.allTimeBest        = new NeuralNetwork(data.topology).fromGenome(data.allTimeBestGenome);
              this.allTimeBestFitness = data.allTimeBestFitness;
            }
            console.log('[DinoBot] Loaded full population — gen ' + data.generation);
          } else {
            // Single best genome — seed population from it
            const best = new NeuralNetwork(data.topology).fromGenome(data.genome);
            this._seedFromBest(best, data.fitness, data.generation);
            console.log('[DinoBot] Loaded best genome — gen ' + data.generation + ', score ' + Math.round(data.fitness));
          }
          this._addCheckpoint(); // imported session appears in checkpoints panel
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
    this.allTimeBest        = null;
    this.allTimeBestFitness = -Infinity;
    this.checkpoints        = [];
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
    };
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  _seedFromBest(nn, fitness, generation) {
    this.allTimeBest        = nn;
    this.allTimeBestFitness = fitness;
    this.generation         = generation;
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
        topology: this.topology, fitness: this.allTimeBestFitness, generation: this.generation,
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
      this._seedFromBest(new NeuralNetwork(data.topology).fromGenome(data.genome), data.fitness, data.generation);
      console.log('[DinoBot] Loaded saved genome — gen ' + data.generation + ', score ' + Math.round(data.fitness));
      return true;
    } catch (_) { return false; }
  }
}
/**
 * DinoBot — main controller
 *
 * Reads game state from Runner.instance_, feeds it into the current neural
 * network, and sends actions directly via the game's own API.
 *
 * Depends on: NeuralNetwork, GeneticAlgorithm (load them first).
 */
window.DinoBot = (() => {
  // ── Constants ──────────────────────────────────────────────────────────────

  const DINO_X         = 75;
  const CANVAS_WIDTH   = 600;
  const CANVAS_HEIGHT  = 150;
  const MAX_SPEED      = 13;
  const TOPOLOGY       = [7, 8, 2];
  const POLL_MS        = 50;
  const CRASH_PAUSE_MS = 600;

  const INPUT_LABELS = ['dist  ', 'width ', 'height', 'type  ', 'obs-y ', 'speed ', 'dino-y'];

  // ── GA setup ───────────────────────────────────────────────────────────────

  const ga = new GeneticAlgorithm({
    populationSize: 15,
    mutationRate:   0.15,
    mutationScale:  0.4,
    eliteCount:     2,
    topology:       TOPOLOGY,
    autoSaveEvery:  5,
    maxCheckpoints: 10,
  });

  ga.loadFromStorage();

  // ── State ──────────────────────────────────────────────────────────────────

  let intervalId        = null;
  let overlay           = null;   // top-right: main HUD
  let cpPanel           = null;   // bottom-right: checkpoints
  let genPanel          = null;   // top-left: generation performance
  let cpExpanded        = true;
  let genExpanded       = true;
  let competitionActive = false;
  let competitionDone   = false;
  let competitionResults = null;  // sorted scores after competition completes
  let currentScore = 0;
  let lastAction   = 'waiting...';
  let jumpOut      = 0;
  let duckOut      = 0;
  let lastInputs   = [0, 0, 0, 0, 0, 0, 0];
  let crashedAt    = null;
  let forcedRunner = null;
  let runnerLogged = false;

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
      const obs = obstacles.reduce((a, b) => (b.xPos < a.xPos ? b : a));
      dist   = Math.max(0, obs.xPos - DINO_X) / CANVAS_WIDTH;
      width  = (obs.width || 0) / CANVAS_WIDTH;
      height = (obs.typeConfig.height || 0) / CANVAS_HEIGHT;
      type   = obs.typeConfig.type === 'PTERODACTYL' ? 1 : 0;
      obsY   = obs.yPos / CANVAS_HEIGHT;
    }
    return [dist, width, height, type, obsY, runner.currentSpeed / MAX_SPEED, runner.tRex.yPos / CANVAS_HEIGHT];
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
      console.log('[DinoBot] playing:', runner.playing, '| crashed:', runner.crashed);
      console.log('[DinoBot] tRex keys:', runner.tRex ? Object.keys(runner.tRex) : 'no tRex');
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
        ga.recordFitness(getDisplayScore(runner));
        currentScore = 0; jumpOut = 0; duckOut = 0;
        lastAction   = 'DEAD';
        crashedAt    = Date.now();
      }
      updateOverlay(); updateGenPanel();
      if (Date.now() - crashedAt >= CRASH_PAUSE_MS) { crashedAt = null; gameRestart(runner); }
      return;
    }

    currentScore = getDisplayScore(runner);
    lastInputs   = buildInputs(runner);
    const [jOut, dOut] = ga.getCurrentNetwork().forward(lastInputs);
    jumpOut = jOut; duckOut = dOut;

    if (jumpOut > duckOut && jumpOut > 0.5) {
      gameJump(runner); gameDuck(runner, false); lastAction = 'JUMP';
    } else if (duckOut > jumpOut && duckOut > 0.5) {
      gameDuck(runner, true); lastAction = 'DUCK';
    } else {
      gameDuck(runner, false); lastAction = 'run';
    }

    updateOverlay(); updateGenPanel();
  }

  // ── Main HUD (top-right) ──────────────────────────────────────────────────

  function bar(v, len = 10) {
    const f = Math.round(Math.max(0, Math.min(1, v)) * len);
    return '[' + '|'.repeat(f) + '.'.repeat(len - f) + '] ' + Math.round(v * 100) + '%';
  }

  function actionColor(a) {
    return a === 'JUMP' ? '#ffff00' : a === 'DUCK' ? '#00aaff' : a === 'DEAD' ? '#ff4444' : '#39ff14';
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
    overlay.innerHTML = [
      '<b style="color:#fff;font-size:13px">DINO BOT</b>',
      '<span style="color:#666">Gen ' + s.generation + '  #' + s.individual + ' / ' + s.populationSize + '</span>',
      '',
      'Score    : <b>' + Math.round(currentScore) + '</b>',
      'Gen best : ' + s.generationBest,
      'All-time : ' + s.allTimeBest,
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
    // Reset current gen so all individuals play from the start
    ga.currentIndex = 0;
    ga.fitnesses    = new Array(ga.populationSize).fill(0);
    updateGenPanel();
    // Make sure the bot is running
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

  // ── Generation performance panel (top-left) ───────────────────────────────

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
            ga.competitionMode = false;
            ga.reset();
          }
        }
        if (btn.dataset.action === 'competition') toggleCompetition();
        if (btn.dataset.action === 'resumetrain') exitCompetition();
        return;
      }
      if (e.target.closest('[data-toggle]')) { genExpanded = !genExpanded; updateGenPanel(); }
    });

    ga.onEvolve              = () => updateGenPanel();
    ga.onGenerationComplete  = () => {
      // Competition finished — collect sorted results and stop the bot
      competitionDone    = true;
      competitionResults = ga.fitnesses
        .map((score, i) => ({ individual: i + 1, score }))
        .sort((a, b) => b.score - a.score);
      updateGenPanel();
      stop();
    };

    document.body.appendChild(genPanel);
    updateGenPanel();
  }

  function updateGenPanel() {
    if (!genPanel) return;
    const s     = ga.getStats();
    const arrow = genExpanded ? '▲' : '▼';

    // Max score in this generation so far (for normalising bar widths)
    const scores   = ga.fitnesses.slice(0, ga.currentIndex);
    const maxScore = Math.max(...scores, currentScore, 1);

    const modeLabel  = competitionActive ? '<span style="color:#ff8c00"> ⚔ COMPETITION</span>' : '';
    const compBtnTxt = competitionActive ? '✕ Exit' : '⚔ Compete';
    const compBtnClr = competitionActive ? '#6b2a1a' : '#4a3a00';

    let html =
      '<div style="padding:8px 14px;border-bottom:1px solid #222;display:flex;align-items:center;gap:6px">' +
      '<span data-toggle style="flex:1;cursor:pointer"><b style="color:#fff">GEN ' + s.generation + modeLabel + '</b></span>' +
      '<button data-action="newgen"      style="' + BTN('#3a1a1a') + '">+ New</button>' +
      '<button data-action="savegen"     style="' + BTN('#1a3a6b') + '">&#8595; Save</button>' +
      '<button data-action="competition" style="' + BTN(compBtnClr) + '">' + compBtnTxt + '</button>' +
      '<span data-toggle style="color:#666;cursor:pointer">' + arrow + '</span>' +
      '</div>';

    if (genExpanded && competitionDone && competitionResults) {
      // ── Competition results view ─────────────────────────────────────────
      const scores  = competitionResults.map(r => r.score);
      const best    = scores[0];
      const avg     = Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);
      const worst   = scores[scores.length - 1];

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
    } else if (genExpanded) {
      html += '<div style="padding:4px 0">';
      for (let i = 0; i < ga.populationSize; i++) {
        const isDone    = i < ga.currentIndex;
        const isCurrent = i === ga.currentIndex;
        const score     = isDone ? ga.fitnesses[i] : (isCurrent ? currentScore : 0);
        const pct       = Math.min(100, Math.round((score / maxScore) * 100));

        const numColor  = isDone ? '#39ff14' : isCurrent ? '#ffff00' : '#444';
        const barColor  = isDone ? '#1a6b2a' : isCurrent ? '#6b6b00' : '#1a1a1a';
        const fillColor = isDone ? '#39ff14' : isCurrent ? '#ffff00' : '#222';
        const label     = isCurrent ? '&#9654;' : isDone ? '&#10003;' : '&nbsp;';
        const scoreText = (isDone || isCurrent) ? String(score).padStart(4, '\u00a0') : '\u00a0---';

        // Origin badge from evolutionLog (only available from gen 2 onward)
        const origin    = ga.evolutionLog && ga.evolutionLog.origins[i];
        const isElite   = origin && origin.type === 'elite';
        const badge     = origin
          ? (isElite
              ? '<span style="color:#39ff14;font-size:10px">[E]</span>'
              : '<span style="color:#ff8c00;font-size:10px">[M]</span>')
          : '<span style="width:22px;display:inline-block"></span>';
        const from      = origin
          ? '<span style="color:#444;font-size:10px">\u2190' + Math.round(origin.prevScore) + '</span>'
          : '';

        html +=
          '<div style="padding:3px 14px;display:flex;align-items:center;gap:6px' + (isCurrent ? ';background:#1a1a00' : '') + '">' +
          '<span style="color:#555;width:20px;text-align:right">#' + (i + 1) + '</span>' +
          badge +
          '<span style="color:' + numColor + ';width:34px;text-align:right">' + scoreText + '</span>' +
          '<div style="flex:1;background:#1a1a1a;height:8px;border-radius:3px;overflow:hidden;border:1px solid ' + barColor + '">' +
            '<div style="width:' + pct + '%;height:100%;background:' + fillColor + ';border-radius:3px;transition:width 0.1s"></div>' +
          '</div>' +
          '<span style="color:' + numColor + ';width:14px;text-align:center">' + label + '</span>' +
          from +
          '</div>';
      }
      html += '<div style="padding:4px 14px;color:#444;font-size:10px;border-top:1px solid #1a1a1a">' +
        '<span style="color:#39ff14">[E]</span> elite carried over &nbsp; ' +
        '<span style="color:#ff8c00">[M]</span> mutated offspring' +
        '</div>';
      html += '</div>';
    } // end else-if genExpanded

    genPanel.innerHTML = html;
  }

  // ── Checkpoint panel (bottom-right) ───────────────────────────────────────

  const BTN = (color) =>
    'background:' + color + ';color:#fff;border:none;border-radius:3px;' +
    'padding:2px 7px;cursor:pointer;font-family:monospace;font-size:11px;margin-left:4px';

  function createCheckpointPanel() {
    cpPanel = document.createElement('div');
    Object.assign(cpPanel.style, {
      position: 'fixed', bottom: '10px', right: '10px',
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

    ga.onCheckpoint = () => updateCheckpointPanel();

    document.body.appendChild(cpPanel);
    updateCheckpointPanel();
  }

  function updateCheckpointPanel() {
    if (!cpPanel) return;
    const count = ga.checkpoints.length;
    const arrow = cpExpanded ? '▲' : '▼';

    let html =
      '<div style="padding:8px 14px;border-bottom:1px solid #222;display:flex;align-items:center;gap:8px">' +
      '<span data-toggle style="flex:1;cursor:pointer"><b style="color:#fff">CHECKPOINTS</b> <span style="color:#555">(auto every 5 gens)</span></span>' +
      '<button data-action="import" style="' + BTN('#555') + '">&#8593; Import</button>' +
      '<span data-toggle style="color:#666;cursor:pointer">' + count + ' ' + arrow + '</span>' +
      '</div>';

    if (cpExpanded) {
      if (count === 0) {
        html += '<div style="padding:8px 14px;color:#555">No checkpoints yet — keeps training...</div>';
      } else {
        ga.checkpoints.forEach((cp, i) => {
          html +=
            '<div style="padding:5px 14px;border-bottom:1px solid #1a1a1a;display:flex;align-items:center">' +
            '<span style="flex:1;color:#aaa">' +
              'Gen <b style="color:#fff">' + String(cp.generation).padStart(3, '\u00a0') + '</b>' +
              ' &nbsp;|&nbsp; <b style="color:#39ff14">' + Math.round(cp.fitness) + '</b> pts' +
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

  // ── Public API ─────────────────────────────────────────────────────────────

  function start() {
    if (intervalId) { console.log('[DinoBot] Already running.'); return; }
    createOverlay();
    createGenPanel();
    createCheckpointPanel();
    intervalId = setInterval(tick, POLL_MS);
    console.log(
      '[DinoBot] Started.\n' +
      '  DinoBot.stop()                           — pause\n' +
      '  DinoBot.save()                           — download best genome as dino_genome.json\n' +
      '  DinoBot.load()                           — import a saved file (best or population)\n' +
      '  DinoBot.ga.reset()                       — wipe training and start over\n' +
      '  DinoBot.forceRunner(Runner.getInstance()) — set runner manually if needed'
    );
  }

  function stop() {
    clearInterval(intervalId); intervalId = null;
    const runner = getRunner();
    if (runner) gameDuck(runner, false);
    if (overlay)  { overlay.remove();  overlay  = null; }
    if (genPanel) { genPanel.remove(); genPanel = null; }
    if (cpPanel)  { cpPanel.remove();  cpPanel  = null; }
    console.log('[DinoBot] Stopped.');
  }

  start();

  return {
    start, stop, ga,
    save() { ga.exportToFile(); },
    load() { ga.importFromFile(); },
    forceRunner(r) { forcedRunner = r; console.log('[DinoBot] Runner set to:', r); },
  };
})();
