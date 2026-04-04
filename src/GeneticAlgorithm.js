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
