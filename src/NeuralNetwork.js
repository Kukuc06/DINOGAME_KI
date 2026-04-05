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
