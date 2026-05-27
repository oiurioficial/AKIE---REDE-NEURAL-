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
  temperature: 0.45, // reduzido: modelo em estágio inicial precisa de menos aleatoriedade
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
    // Tamanho real da camada de embedding — diferente de vocab.size quando
    // o vocabulário cresce em runtime mas o modelo ainda não foi expandido
    this.embeddingVocabSize = 0;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Construção do modelo
  // ─────────────────────────────────────────────────────────────────────────

  build() {
    const { embDim, hiddenSize, maxSeqLen } = this.hparams;
    const vocabSize = this.vocab.size;

    const input = tf.input({ shape: [maxSeqLen], dtype: 'int32', name: 'context' });

    // Embedding: converte token IDs em vetores densos
    // maskZero removido — tfjs-node propaga a máscara como int32 e quebra
    // a operação floor() internamente no LSTM com input dtype int32
    const embedding = tf.layers.embedding({
      inputDim: vocabSize,
      outputDim: embDim,
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
    this.embeddingVocabSize = vocabSize; // registrar tamanho real da camada de embedding
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
    // float32 obrigatório — sparseCategoricalCrossentropy chama floor() internamente
    // nos labels e tfjs-node não aceita int32 nessa operação
    const ys = tf.tensor1d(pairs.map(p => p.y), 'float32');

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
   * Gera texto usando beam search com repetition penalty.
   *
   * Beam search mantém as `beamWidth` melhores sequências candidatas
   * a cada passo e escolhe a de maior score ao final.
   * Repetition penalty reduz a probabilidade de tokens já gerados,
   * evitando loops de palavras repetidas.
   *
   * @param {string} prompt        - Texto de entrada (contexto)
   * @param {number} maxTokens     - Máximo de tokens a gerar
   * @param {number} temperature   - 0.0 = determinístico, 1.0+ = criativo
   * @returns {string}             - Texto gerado
   */
  generate(prompt = '', maxTokens = null, temperature = null) {
    if (!this.ready) return null;

    const maxT      = maxTokens || this.hparams.maxGenTokens;
    const temp      = temperature !== null ? temperature : this.hparams.temperature;
    const beamWidth = 3;
    const repPenalty = 1.3; // fator de penalidade para tokens já gerados

    // Tokenizar prompt — remover EOS para continuar gerando
    const promptIds  = this.vocab.tokenize(prompt).filter(id => id !== SPECIAL.EOS);

    // Cada beam: { context: number[], tokens: number[], score: number }
    let beams = [{ context: [...promptIds], tokens: [], score: 0.0 }];

    for (let step = 0; step < maxT; step++) {
      const candidates = [];

      for (const beam of beams) {
        // Preparar contexto com padding à esquerda
        const padLen = this.hparams.maxSeqLen;
        const padded = [
          ...Array(Math.max(0, padLen - beam.context.length)).fill(SPECIAL.PAD),
          ...beam.context.slice(-padLen),
        ];

        // Inferência
        const inputTensor = tf.tensor2d([padded], [1, padLen], 'int32');
        const rawLogits   = this.model.predict(inputTensor);
        const logits      = rawLogits.dtype === 'float32'
          ? rawLogits : rawLogits.cast('float32');

        let probs = Array.from(logits.dataSync());
        inputTensor.dispose();
        if (logits !== rawLogits) rawLogits.dispose();
        logits.dispose();

        // Repetition penalty — dividir prob de tokens já gerados
        const generatedSet = new Set(beam.tokens);
        probs = probs.map((p, id) =>
          generatedSet.has(id) ? p / repPenalty : p
        );

        // Aplicar temperatura e re-normalizar
        const logP  = probs.map(p => Math.log(Math.max(p, 1e-10)) / Math.max(temp, 0.01));
        const maxLP = Math.max(...logP);
        const exps  = logP.map(lp => Math.exp(lp - maxLP));
        const sum   = exps.reduce((a, b) => a + b, 0);
        const scaled = exps.map(e => e / sum);

        // Top-K: manter os beamWidth tokens mais prováveis como candidatos
        const topK = scaled
          .map((p, id) => ({ id, p }))
          .filter(t => t.id !== SPECIAL.PAD && t.id !== SPECIAL.BOS)
          .sort((a, b) => b.p - a.p)
          .slice(0, beamWidth);

        for (const { id: nextId, p } of topK) {
          candidates.push({
            context: [...beam.context, nextId],
            tokens:  [...beam.tokens, nextId],
            score:   beam.score + Math.log(Math.max(p, 1e-10)),
            done:    nextId === SPECIAL.EOS,
          });
        }
      }

      // Selecionar os beamWidth melhores candidatos
      candidates.sort((a, b) => b.score - a.score);
      beams = candidates.slice(0, beamWidth);

      // Se todos terminaram com EOS, parar
      if (beams.every(b => b.done)) break;

      // Remover beams que terminaram da lista ativa (mas guardar o melhor)
      const done   = beams.filter(b => b.done);
      const active = beams.filter(b => !b.done);
      if (active.length === 0) { beams = done; break; }
      beams = active;
    }

    // Retornar o beam com maior score, ignorando EOS no output
    const best = beams.sort((a, b) => b.score - a.score)[0];
    const outputIds = best.tokens.filter(
      id => id !== SPECIAL.EOS && id !== SPECIAL.PAD && id !== SPECIAL.BOS
    );

    return this.vocab.detokenize(outputIds);
  }

  /**
   * Amostragem simples com temperatura — mantida para uso interno / self-play.
   * @private
   */
  _sampleWithTemperature(logitsTensor, temperature) {
    const probs = Array.from(logitsTensor.dataSync());

    if (temperature <= 0.01) {
      let maxIdx = 0;
      for (let i = 1; i < probs.length; i++) {
        if (probs[i] > probs[maxIdx]) maxIdx = i;
      }
      return maxIdx;
    }

    const logProbs = probs.map(p => Math.log(Math.max(p, 1e-10)) / temperature);
    const maxLog   = Math.max(...logProbs);
    const exps     = logProbs.map(lp => Math.exp(lp - maxLog));
    const sumExps  = exps.reduce((a, b) => a + b, 0);
    const scaled   = exps.map(e => e / sumExps);

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

      // Inferir embeddingVocabSize a partir da camada de embedding carregada
      try {
        const embLayer = this.model.getLayer('embedding');
        this.embeddingVocabSize = embLayer.getWeights()[0].shape[0];
      } catch(e) {
        this.embeddingVocabSize = this.vocab.size;
      }
      this.ready = true;
      console.log(`[AKIE] Modelo carregado de ${dir} (steps: ${this.trainSteps}, embVocab: ${this.embeddingVocabSize})`);
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
    // Comparar contra embeddingVocabSize (tamanho REAL da camada),
    // não vocab.size — que já cresceu antes desta chamada ser feita
    if (newVocabSize <= this.embeddingVocabSize) return;

    console.log(`[AKIE] Expandindo vocabulário: ${this.embeddingVocabSize} → ${newVocabSize}`);

    // Salvar pesos existentes
    const oldWeights = this.model.getWeights();
    const oldEmbSize = this.embeddingVocabSize;

    // Patch temporário do vocab.size para o build()
    // Vocabulary.size é um getter sobre id2token.length — patchear diretamente
    const realId2token = this.vocab.id2token;
    const paddingNeeded = newVocabSize - realId2token.length;
    const tempTokens = paddingNeeded > 0
      ? Array.from({ length: paddingNeeded }, (_, i) => `<EXPAND_${oldEmbSize + i}>`)
      : [];
    if (tempTokens.length > 0) {
      this.vocab.id2token = [...realId2token, ...tempTokens];
      for (const t of tempTokens) {
        if (!(t in this.vocab.token2id)) this.vocab.token2id[t] = this.vocab.id2token.indexOf(t);
      }
    }

    this.build();

    // Restaurar — os tokens temporários de padding serão substituídos
    // por tokens reais quando o vocab crescer organicamente
    // (mantemos no vocab para não quebrar IDs já emitidos)

    // Copiar pesos das camadas que não mudaram
    const newWeights = this.model.getWeights();
    const updatedWeights = newWeights.map((w, i) => {
      const oldW = oldWeights[i];
      if (!oldW) return w;

      const oldShape = oldW.shape;
      const newShape = w.shape;

      // Shapes iguais → copiar diretamente (LSTM, Dense intermediário)
      if (JSON.stringify(oldShape) === JSON.stringify(newShape)) {
        return oldW;
      }

      // Embedding [vocabSize, embDim] ou Output [hiddenSize, vocabSize] —
      // primeira ou segunda dimensão expandiu
      if (oldShape.length === 2 && (newShape[0] > oldShape[0] || newShape[1] > oldShape[1])) {
        const oldData = oldW.dataSync();
        const newData = w.dataSync();
        const merged  = new Float32Array(newData); // inicializado aleatoriamente
        // Copiar apenas a região dos tokens antigos
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
    this.embeddingVocabSize = newVocabSize;
    console.log(`[AKIE] Vocabulário expandido com sucesso.`);
  }

  /**
   * Retorna métricas do modelo para monitoramento.
   */
  getStats() {
    return {
      ready: this.ready,
      vocabSize: this.vocab.size,
      embeddingVocabSize: this.embeddingVocabSize, // tamanho REAL da camada
      trainSteps: this.trainSteps,
      parameters: this.ready ? this.model.countParams() : 0,
      hparams: this.hparams,
    };
  }
}

module.exports = { AKIEModel, HPARAMS };
