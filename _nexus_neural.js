/**
 * _nexus_neural.js  —  AKIE Neural Core
 *
 * Rede neural real baseada em LSTM para geração de linguagem.
 * Tecnologia: TensorFlow.js (tfjs-node) — treina localmente em CPU/GPU.
 *
 * Arquitetura:
 *   Embedding(vocabSize, embDim) → LSTM(hiddenSize) → Dense(vocabSize) → Softmax
 *
 * Paradigma: next-word prediction (language modeling padrão).
 * Geração: amostragem com temperatura (não é template, é inferência real).
 */

const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const { Vocabulary, makeTrainingPairs, SPECIAL } = require('./_akie_vocab');

// ---------------------------------------------------------------------------
// Hiperparâmetros
// ---------------------------------------------------------------------------

const HPARAMS = {
  embDim:     64,    // dimensão dos embeddings
  hiddenSize: 128,   // tamanho do estado LSTM
  maxSeqLen:  32,    // comprimento máximo de contexto
  batchSize:  16,    // tamanho de batch para treino
  learningRate: 0.001,
  temperature: 0.8,  // controla aleatoriedade da geração (0=greedy, 1=criativo)
  maxGenTokens: 40,  // máximo de tokens gerados por resposta
  minNewData:  3,    // mínimo de exemplos novos para disparar treino
};

// ---------------------------------------------------------------------------
// Classe principal: AKIEModel
// ---------------------------------------------------------------------------

class AKIEModel {
  constructor(vocab, hparams = HPARAMS) {
    this.vocab = vocab;
    this.hparams = { ...HPARAMS, ...hparams };
    this.model = null;
    this.optimizer = null;
    this.trainSteps = 0;
    this.ready = false;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Construção do modelo
  // ─────────────────────────────────────────────────────────────────────────

  build() {
    const { embDim, hiddenSize, maxSeqLen } = this.hparams;
    const vocabSize = this.vocab.size;

    const input = tf.input({ shape: [maxSeqLen], dtype: 'int32', name: 'context' });

    // Embedding: converte token IDs em vetores densos
    const embedding = tf.layers.embedding({
      inputDim: vocabSize,
      outputDim: embDim,
      maskZero: true,   // ignora padding no LSTM
      name: 'embedding',
    }).apply(input);

    // LSTM: captura dependências sequenciais
    const lstm = tf.layers.lstm({
      units: hiddenSize,
      returnSequences: false,
      dropout: 0.1,
      recurrentDropout: 0.1,
      name: 'lstm',
    }).apply(embedding);

    // Camada intermediária
    const dense1 = tf.layers.dense({
      units: hiddenSize,
      activation: 'relu',
      name: 'dense_hidden',
    }).apply(lstm);

    // Saída: distribuição sobre o vocabulário
    const output = tf.layers.dense({
      units: vocabSize,
      activation: 'softmax',
      name: 'output',
    }).apply(dense1);

    this.model = tf.model({ inputs: input, outputs: output, name: 'AKIE' });

    this.optimizer = tf.train.adam(this.hparams.learningRate);

    this.model.compile({
      optimizer: this.optimizer,
      loss: 'sparseCategoricalCrossentropy',
      metrics: ['accuracy'],
    });

    this.ready = true;
    console.log(`[AKIE] Modelo construído. Parâmetros: ${this.model.countParams().toLocaleString()}`);
    console.log(`[AKIE] Vocabulário: ${vocabSize} tokens`);
    return this;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Treinamento incremental
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Treina o modelo com um batch de pares (contexto → próximo token).
   * 
   * @param {Array<{x: number[], y: number}>} pairs  - Pares de treino
   * @param {number} epochs  - Épocas de treino neste batch
   * @returns {object} { loss, accuracy, steps }
   */
  async trainBatch(pairs, epochs = 3) {
    if (!this.ready) throw new Error('Modelo não construído. Chame build() primeiro.');
    if (!pairs || pairs.length === 0) return { loss: null, accuracy: null, steps: 0 };

    // Construir tensores
    const xs = tf.tensor2d(
      pairs.map(p => p.x),
      [pairs.length, this.hparams.maxSeqLen],
      'int32'
    );
    const ys = tf.tensor1d(pairs.map(p => p.y), 'int32');

    let lastLoss = null;
    let lastAcc = null;

    try {
      const history = await this.model.fit(xs, ys, {
        epochs,
        batchSize: this.hparams.batchSize,
        shuffle: true,
        verbose: 0,  // silencioso — log manual abaixo
      });

      const losses = history.history.loss;
      const accs   = history.history.acc || history.history.accuracy;

      lastLoss = losses[losses.length - 1];
      lastAcc  = accs ? accs[accs.length - 1] : null;
      this.trainSteps += pairs.length * epochs;

    } finally {
      xs.dispose();
      ys.dispose();
    }

    return {
      loss: lastLoss,
      accuracy: lastAcc,
      steps: this.trainSteps,
      samples: pairs.length,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Geração de linguagem
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Gera texto a partir de um prompt usando amostragem com temperatura.
   * NÃO é template — é inferência real da rede neural.
   *
   * @param {string} prompt        - Texto de entrada (contexto)
   * @param {number} maxTokens     - Máximo de tokens a gerar
   * @param {number} temperature   - 0.0 = determinístico, 1.0+ = criativo
   * @returns {string}             - Texto gerado
   */
  generate(prompt = '', maxTokens = null, temperature = null) {
    if (!this.ready) return null;

    const maxT = maxTokens || this.hparams.maxGenTokens;
    const temp = temperature !== null ? temperature : this.hparams.temperature;

    // Tokenizar prompt
    const promptIds = this.vocab.tokenize(prompt);
    // Remove EOS do fim para continuar gerando
    const contextIds = promptIds.filter(id => id !== SPECIAL.EOS);

    const generated = [];
    let currentContext = [...contextIds];

    for (let step = 0; step < maxT; step++) {
      // Preparar contexto com padding
      const padLen = this.hparams.maxSeqLen;
      const padded = [
        ...Array(Math.max(0, padLen - currentContext.length)).fill(SPECIAL.PAD),
        ...currentContext.slice(-padLen),
      ];

      // Inferência
      const inputTensor = tf.tensor2d([padded], [1, padLen], 'int32');
      const rawLogits = this.model.predict(inputTensor);

      // Cast explícito para float32 — tfjs-node pode retornar int32 dependendo
      // do dtype interno das camadas de embedding quando o input é int32
      const logits = rawLogits.dtype === 'float32'
        ? rawLogits
        : rawLogits.cast('float32');

      // Amostragem com temperatura
      const nextId = this._sampleWithTemperature(logits, temp);
      inputTensor.dispose();
      if (logits !== rawLogits) rawLogits.dispose();
      logits.dispose();

      // Parar em EOS
      if (nextId === SPECIAL.EOS) break;
      // Pular PAD e BOS na saída
      if (nextId === SPECIAL.PAD || nextId === SPECIAL.BOS) continue;

      generated.push(nextId);
      currentContext.push(nextId);
    }

    return this.vocab.detokenize(generated);
  }

  /**
   * Amostragem com temperatura a partir de logits.
   * Espera tensor float32 — o cast deve ter sido feito antes de chamar este método.
   * @private
   */
  _sampleWithTemperature(logitsTensor, temperature) {
    // dataSync() em float32 retorna Float32Array diretamente
    // Se por alguma razão ainda for int32, Math.log ainda funciona — mas o cast
    // no método generate() deve ter resolvido antes de chegar aqui
    const probs = Array.from(logitsTensor.dataSync());

    if (temperature <= 0.01) {
      // Greedy: retorna o token com maior probabilidade
      let maxIdx = 0;
      let maxVal = probs[0];
      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > maxVal) { maxVal = probs[i]; maxIdx = i; }
      }
      return maxIdx;
    }

    // Aplicar temperatura: dividir log-probs por temperature e re-normalizar
    const logProbs = probs.map(p => Math.log(Math.max(p, 1e-10)) / temperature);
    const maxLog = Math.max(...logProbs);
    const exps = logProbs.map(lp => Math.exp(lp - maxLog));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const scaled = exps.map(e => e / sumExps);

    // Amostragem multinomial
    const rand = Math.random();
    let cumSum = 0;
    for (let i = 0; i < scaled.length; i++) {
      cumSum += scaled[i];
      if (rand <= cumSum) return i;
    }
    return scaled.length - 1;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Persistência
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Salva o modelo em disco (Railway Volume ou path local).
   * @param {string} dir - Diretório de destino
   */
  async save(dir) {
    if (!this.ready) return;
    await fs.promises.mkdir(dir, { recursive: true });
    await this.model.save(`file://${dir}`);

    // Salvar vocabulário e metadados junto
    const meta = {
      trainSteps: this.trainSteps,
      vocabSize: this.vocab.size,
      hparams: this.hparams,
      savedAt: new Date().toISOString(),
    };
    await fs.promises.writeFile(
      path.join(dir, 'akie_meta.json'),
      JSON.stringify(meta, null, 2)
    );
    await fs.promises.writeFile(
      path.join(dir, 'akie_vocab.json'),
      JSON.stringify(this.vocab.toJSON(), null, 2)
    );

    console.log(`[AKIE] Modelo salvo em ${dir} (steps: ${this.trainSteps})`);
  }

  /**
   * Carrega modelo do disco.
   * @param {string} dir - Diretório de origem
   * @returns {boolean} - true se carregou com sucesso
   */
  async load(dir) {
    try {
      const modelPath = path.join(dir, 'model.json');
      if (!fs.existsSync(modelPath)) return false;

      this.model = await tf.loadLayersModel(`file://${dir}/model.json`);

      // Recompilar após load
      this.optimizer = tf.train.adam(this.hparams.learningRate);
      this.model.compile({
        optimizer: this.optimizer,
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy'],
      });

      // Carregar metadados
      const metaPath = path.join(dir, 'akie_meta.json');
      if (fs.existsSync(metaPath)) {
        const meta = JSON.parse(await fs.promises.readFile(metaPath, 'utf8'));
        this.trainSteps = meta.trainSteps || 0;
      }

      this.ready = true;
      console.log(`[AKIE] Modelo carregado de ${dir} (steps: ${this.trainSteps})`);
      return true;

    } catch (err) {
      console.error('[AKIE] Falha ao carregar modelo:', err.message);
      return false;
    }
  }

  /**
   * Expande a camada de embedding e output quando o vocabulário cresce.
   * Necessário quando novos tokens são adicionados após o build inicial.
   * 
   * Estratégia: cria novo modelo com pesos copiados para tokens existentes.
   */
  async expandVocabulary(newVocabSize) {
    if (!this.ready) return;
    if (newVocabSize <= this.vocab.size) return;

    console.log(`[AKIE] Expandindo vocabulário: ${this.vocab.size} → ${newVocabSize}`);

    // Salvar pesos existentes
    const oldWeights = this.model.getWeights();
    const oldVocabSize = this.vocab.size;

    // Rebuild com novo tamanho
    const oldVocab = this.vocab;
    this.vocab = { ...oldVocab, size: newVocabSize }; // temporário para build
    this.build();

    // Copiar pesos das camadas que não mudaram
    const newWeights = this.model.getWeights();
    const updatedWeights = newWeights.map((w, i) => {
      const oldW = oldWeights[i];
      if (!oldW) return w;

      const oldShape = oldW.shape;
      const newShape = w.shape;

      // Se shapes são iguais, copiar diretamente
      if (JSON.stringify(oldShape) === JSON.stringify(newShape)) {
        return oldW;
      }

      // Se é a camada de embedding ou output (primeira dimensão expandiu)
      if (oldShape.length === 2 && newShape[0] > oldShape[0]) {
        const oldData = oldW.dataSync();
        const newData = w.dataSync(); // inicializado aleatoriamente
        // Copiar pesos dos tokens antigos
        const merged = new Float32Array(newData);
        for (let r = 0; r < oldShape[0]; r++) {
          for (let c = 0; c < oldShape[1]; c++) {
            merged[r * newShape[1] + c] = oldData[r * oldShape[1] + c];
          }
        }
        return tf.tensor(merged, newShape);
      }

      return w;
    });

    this.model.setWeights(updatedWeights);
    oldWeights.forEach(w => w.dispose());
    console.log(`[AKIE] Vocabulário expandido com sucesso.`);
  }

  /**
   * Retorna métricas do modelo para monitoramento.
   */
  getStats() {
    return {
      ready: this.ready,
      vocabSize: this.vocab.size,
      trainSteps: this.trainSteps,
      parameters: this.ready ? this.model.countParams() : 0,
      hparams: this.hparams,
    };
  }
}

module.exports = { AKIEModel, HPARAMS };
