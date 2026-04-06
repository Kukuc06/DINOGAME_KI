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
    topology              = [13, 8, 2],
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
          // If mutation is already high, do a partial reset of the weakest individuals
          // instead of just cranking mutation further — avoids destroying refined elites
          if (this.mutationRate >= this.adaptiveMutationMax * 0.75) {
            const resetCount = Math.floor(this.populationSize * 0.25);
            for (let i = this.population.length - resetCount; i < this.population.length; i++) {
              this.population[i] = new NeuralNetwork(this.topology);
            }
            this.stagnantGens = 0;
          } else {
            this.mutationRate = Math.min(this.adaptiveMutationMax, this.mutationRate * 1.3);
          }
        }
      }
    }

    // Diversity maintenance: if genomes have converged too closely, inject fresh individuals
    this._maintainDiversity();

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

  _maintainDiversity() {
    // Compute average pairwise L1 genome distance among non-elite individuals.
    // If the population has collapsed (all genomes nearly identical), replace the
    // weakest quarter with fresh random networks to restore exploration.
    const genomes = this.population.slice(this.eliteCount).map(nn => nn.toGenome());
    if (genomes.length < 2) return;

    let totalDist = 0;
    let pairs = 0;
    for (let i = 0; i < genomes.length; i++) {
      for (let j = i + 1; j < genomes.length; j++) {
        let d = 0;
        for (let k = 0; k < genomes[i].length; k++) d += Math.abs(genomes[i][k] - genomes[j][k]);
        totalDist += d / genomes[i].length; // normalize by genome length
        pairs++;
      }
    }
    const avgDist = totalDist / pairs;

    // Threshold: if average per-weight difference is below 0.05, genomes are too similar
    if (avgDist < 0.05) {
      const resetCount = Math.floor(this.populationSize * 0.25);
      for (let i = this.population.length - resetCount; i < this.population.length; i++) {
        this.population[i] = new NeuralNetwork(this.topology);
      }
    }
  }

  _updateHallOfFame(fitness, nn, rawScore) {
    const score = rawScore !== undefined ? rawScore : Math.round(fitness);
    this.hallOfFame.push({
      genome:     nn.toGenome(),
      topology:   this.topology,
      fitness,
      rawScore:   score,
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
              this._updateHallOfFame(this.allTimeBestFitness, this.allTimeBest, data.allTimeBestRawScore);
            }
            console.log('[DinoBot] Loaded full population — gen ' + data.generation);
          } else {
            const best = new NeuralNetwork(data.topology).fromGenome(data.genome);
            this._seedFromBest(best, data.fitness, data.generation, data.rawScore);
            this._updateHallOfFame(data.fitness, best, data.rawScore);
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
    this.mutationRate       = this._baseMutationRate;
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
