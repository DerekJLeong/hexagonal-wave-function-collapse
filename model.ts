"use strict";

import { randomIndice } from "./randomIndice";

/**
 * Base class for Wave Function Collapse models adapted for hexagonal tiles.
 */
export class Model {
  FMX: number; // Field width
  FMY: number; // Field height
  FMXxFMY: number; // Total number of cells in the field
  T: number; // Total number of tile types
  N: number; // Not used in this context
  initiliazedField: boolean; // Flag to indicate if the field is initialized
  generationComplete: boolean; // Flag to indicate if the generation is complete
  wave: boolean[][]; // Wave state representing the possibility space
  compatible: number[][][]; // Compatibility matrix for each tile
  weightLogWeights: number[]; // Weight log weights for each tile type
  sumOfWeights: number; // Sum of weights for all tile types
  sumOfWeightLogWeights: number; // Sum of weight log weights for all tile types
  startingEntropy: number; // Starting entropy value
  sumsOfOnes: number[]; // Sum of ones for each cell in the wave
  sumsOfWeights: number[]; // Sum of weights for each cell in the wave
  sumsOfWeightLogWeights: number[]; // Sum of weight log weights for each cell in the wave
  entropies: number[]; // Entropy values for each cell in the wave
  propagator: number[][][]; // Propagation rules for each direction and tile
  observed: number[]; // Observed state of each cell in the wave
  distribution: number[]; // Probability distribution used in observation
  stack: [number, number][]; // Stack used in propagation to store contradictions
  stackSize: number; // Current size of the stack
  DX: number[]; // Delta X for neighbor directions in a hex grid
  DY: number[]; // Delta Y for neighbor directions in a hex grid
  opposite: number[]; // Opposite direction mapping for each direction
  weights: number[]; // Weights for each tile type
  onBoundary: (x: number, y: number) => boolean; // Function to check boundary conditions


  constructor() {
    this.FMX = 0;
    this.FMY = 0;
    this.FMXxFMY = 0;
    this.T = 0;
    this.N = 0;

    this.initiliazedField = false;
    this.generationComplete = false;

    this.wave = [];
    this.weights = [];
    this.compatible = [];
    this.weightLogWeights = [];
    this.sumOfWeights = 0;
    this.sumOfWeightLogWeights = 0;

    this.startingEntropy = 0;

    this.sumsOfOnes = [];
    this.sumsOfWeights = [];
    this.sumsOfWeightLogWeights = [];
    this.entropies = [];

    this.propagator = [];
    this.observed = [];
    this.distribution = [];

    this.stack = [];
    this.stackSize = 0;

    this.DX = [-1, -1, 0, 1, 1, 0];
    this.DY = [0, 1, 1, 1, 0, -1];
    this.opposite = [3, 4, 5, 0, 1, 2];

    this.onBoundary = (x: number, y: number) => false;

    this.initialize();
  }

  /**
   * Initializes the model's variables and arrays for a new generation.
   */
  private initialize() {
    this.distribution = new Array(this.T);

    this.wave = new Array(this.FMXxFMY);
    this.compatible = new Array(this.FMXxFMY);

    for (let i = 0; i < this.FMXxFMY; i++) {
      this.wave[i] = new Array(this.T);
      this.compatible[i] = new Array(this.T);

      for (let t = 0; t < this.T; t++) {
        this.compatible[i][t] = [0, 0, 0, 0];
      }
    }

    this.weightLogWeights = new Array(this.T);
    this.sumOfWeights = 0;
    this.sumOfWeightLogWeights = 0;

    for (let t = 0; t < this.T; t++) {
      this.weightLogWeights[t] = this.weights[t] * Math.log(this.weights[t]);
      this.sumOfWeights += this.weights[t];
      this.sumOfWeightLogWeights += this.weightLogWeights[t];
    }

    this.startingEntropy =
      Math.log(this.sumOfWeights) -
      this.sumOfWeightLogWeights / this.sumOfWeights;

    this.sumsOfOnes = new Array(this.FMXxFMY);
    this.sumsOfWeights = new Array(this.FMXxFMY);
    this.sumsOfWeightLogWeights = new Array(this.FMXxFMY);
    this.entropies = new Array(this.FMXxFMY);

    this.stack = new Array(this.FMXxFMY * this.T);
    this.stackSize = 0;
  }

  /**
   * Observes the wave to collapse a cell based on minimum entropy.
   */
  private observe(rng: () => number): boolean | null {
    let min = 1000;
    let argmin = -1;

    for (let i = 0; i < this.FMXxFMY; i++) {
      if (this.onBoundary(i % this.FMX, (i / this.FMX) | 0)) continue;

      const amount = this.sumsOfOnes[i];

      if (amount === 0) return false;

      const entropy = this.entropies[i];

      if (amount > 1 && entropy <= min) {
        const noise = 0.000001 * rng();

        if (entropy + noise < min) {
          min = entropy + noise;
          argmin = i;
        }
      }
    }

    if (argmin === -1) {
      this.observed = new Array(this.FMXxFMY);

      for (let i = 0; i < this.FMXxFMY; i++) {
        for (let t = 0; t < this.T; t++) {
          if (this.wave[i][t]) {
            this.observed[i] = t;
            break;
          }
        }
      }

      return true;
    }

    for (let t = 0; t < this.T; t++) {
      this.distribution[t] = this.wave[argmin][t] ? this.weights[t] : 0;
    }
    const r = randomIndice(this.distribution, rng());

    const w = this.wave[argmin];
    for (let t = 0; t < this.T; t++) {
      if (w[t] !== (t === r)) this.ban(argmin, t);
    }

    return null;
  }

  //   onBoundary(arg0: number, arg1: number) {
  //     throw new Error("Method not implemented.");
  //   }

  /**
   * Propagates the consequences of a wave collapse through the field.
   */
  private propagate() {
    while (this.stackSize > 0) {
      const e1 = this.stack[this.stackSize - 1];
      this.stackSize--;

      const i1 = e1[0];
      const x1 = i1 % this.FMX;
      const y1 = Math.floor(i1 / this.FMX);

      for (let d = 0; d < 6; d++) {
        let dx = this.DX[d];
        let dy = this.DY[d];

        // Adjust for even/odd rows in a flat-topped hex grid
        if (y1 % 2 === 0) {
          // Even row
          if (d === 1) {
            // Northeast
            dx = 1;
            dy = 0;
          } else if (d === 4) {
            // Southeast
            dx = 1;
            dy = 1;
          }
        } else {
          // Odd row
          if (d === 1) {
            // Northeast
            dx = 0;
            dy = -1;
          } else if (d === 4) {
            // Southeast
            dx = 0;
            dy = 1;
          }
        }

        let x2 = x1 + dx;
        let y2 = y1 + dy;

        if (this.onBoundary(x2, y2)) continue;

        // Wrap around logic, if applicable
        if (x2 < 0) x2 += this.FMX;
        else if (x2 >= this.FMX) x2 -= this.FMX;
        if (y2 < 0) y2 += this.FMY;
        else if (y2 >= this.FMY) y2 -= this.FMY;

        const i2 = x2 + y2 * this.FMX;
        const p = this.propagator[d][e1[1]];
        const compat = this.compatible[i2];

        for (let l = 0; l < p.length; l++) {
          const t2 = p[l];
          const comp = compat[t2];
          comp[d]--;
          if (comp[d] === 0) this.ban(i2, t2);
        }
      }
    }
  }

  /**
   * Execute a single iteration
   */
  private singleIteration(rng: () => number): boolean | null {
    const result = this.observe(rng);

    if (result !== null) {
      this.generationComplete = result;

      return !!result;
    }

    this.propagate();

    return null;
  }

  /**
   * Execute a fixed number of iterations. Stop when the generation is successful or reaches a contradiction.
   */
  iterate(iterations = 0, rng = Math.random): boolean {
    if (!this.wave.length) this.initialize();

    if (!this.initiliazedField) {
      this.clear();
    }

    for (let i = 0; i < iterations || iterations === 0; i++) {
      const result = this.singleIteration(rng);

      if (result !== null) {
        return !!result;
      }
    }

    return true;
  }

  /**
   * Execute a complete new generation
   */
  generate(rng = Math.random): boolean {
    if (!this.wave.length) this.initialize();

    this.clear();

    while (true) {
      const result = this.singleIteration(rng);

      if (result !== null) {
        return !!result;
      }
    }
  }

  /**
   * Checks if the generation process is complete.
   */
  isGenerationComplete(): boolean {
    return this.generationComplete;
  }

  /**
   * Bans a specific tile at a specific position in the wave.
   */
  private ban(i: number, t: number) {
    const comp = this.compatible[i][t];

    for (let d = 0; d < 6; d++) {
      comp[d] = 0;
    }

    this.wave[i][t] = false;

    this.stack[this.stackSize] = [i, t];
    this.stackSize++;

    this.sumsOfOnes[i] -= 1;
    this.sumsOfWeights[i] -= this.weights[t];
    this.sumsOfWeightLogWeights[i] -= this.weightLogWeights[t];

    const sum = this.sumsOfWeights[i];
    this.entropies[i] = Math.log(sum) - this.sumsOfWeightLogWeights[i] / sum;
  }

  /**
   * Clears the wave and other relevant variables to start a new generation.
   */
  private clear() {
    for (let i = 0; i < this.FMXxFMY; i++) {
      for (let t = 0; t < this.T; t++) {
        this.wave[i][t] = true;

        for (let d = 0; d < 6; d++) {
          this.compatible[i][t][d] =
            this.propagator[this.opposite[d]][t].length;
        }
      }

      this.sumsOfOnes[i] = this.weights.length;
      this.sumsOfWeights[i] = this.sumOfWeights;
      this.sumsOfWeightLogWeights[i] = this.sumOfWeightLogWeights;
      this.entropies[i] = this.startingEntropy;
    }

    this.initiliazedField = true;
    this.generationComplete = false;
  }
}
