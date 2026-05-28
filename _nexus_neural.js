/**
 * _nexus_neural.js — AKIE Neural Core (Transformer Decoder)
 * CORREÇÕES v5: load() limpa modelo corrompido automaticamente
 */

const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const { SPECIAL } = require('./_akie_vocab');

// ── NLP Layer ──────────────────────────────────────────────────────────────
let NaturalNLP = null;
try {
  const natural = require('natural');
  const stemmer = natural.PorterStemmerPt;
  const tfidf = new natural.TfIdf();
  NaturalNLP = {
    isReady: true,
    stem: t => { try { return stemmer.stem(t); } catch { return t; } },
    addDocument: (text, key) => { try { tfidf.addDocument(text, key); } catch {} },
    generationScore: (prompt, generated) => {
      if (!prompt || !generated) return 0;
      const ps = stemmer.tokenizeAndStem(prompt, false);
      const gs = stemmer.tokenizeAndStem(generated, false);
      if (!ps.length || !gs.length) return 0;
      return ps.filter(s => gs.includes(s)).length / ps.length;
    }
  };
  console.log('[NLP] natural carregado.');
} catch {
  NaturalNLP = { isReady: false, stem: t => t, addDocument: () => {}, generationScore: () => 0 };
  console.log('[NLP] natural ausente.');
}
module.exports.NaturalNLP = NaturalNLP;

// ── Hiperparâmetros ────────────────────────────────────────────────────────
const HPARAMS = {
  embDim: 64, hiddenSize: 128, maxSeqLen: 32, batchSize: 16,
  learningRate: 0.001, temperature: 0.8, maxGenTokens: 40,
  beamSize: 3, repetitionPenalty: 1.2,
};

// ── Camadas Customizadas ───────────────────────────────────────────────────
class MultiHeadCausalAttention extends tf.layers.Layer {
  constructor(config) {
    super(config || {});
    this.numHeads = config.numHeads || 4;
    this.headDim = config.headDim || 16;
    this.embDim = this.numHeads * this.headDim;
  }
  build(inputShape) {
    this.qDense = this.addWeight('q_proj', [this.embDim, this.embDim], 'float32', tf.initializers.glorotUniform());
    this.kDense = this.addWeight('k_proj', [this.embDim, this.embDim], 'float32', tf.initializers.glorotUniform());
    this.vDense = this.addWeight('v_proj', [this.embDim, this.embDim], 'float32', tf.initializers.glorotUniform());
    this.outDense = this.addWeight('out_proj', [this.embDim, this.embDim], 'float32', tf.initializers.glorotUniform());
    this.built = true;
  }
  computeOutputShape(inputShape) { return inputShape; }
  call(inputs) {
    return tf.tidy(() => {
      const x = inputs[0] || inputs;
      const [batch, seqLen] = x.shape;
      const xFlat = x.reshape([-1, this.embDim]);
      const q = xFlat.matMul(this.qDense.read()).reshape([batch, seqLen, this.numHeads, this.headDim]).transpose([0, 2, 1, 3]);
      const k = xFlat.matMul(this.kDense.read()).reshape([batch, seqLen, this.numHeads, this.headDim]).transpose([0, 2, 1, 3]);
      const v = xFlat.matMul(this.vDense.read()).reshape([batch, seqLen, this.numHeads, this.headDim]).transpose([0, 2, 1, 3]);
      let scores = q.matMul(k.transpose([0, 1, 3, 2])).div(tf.scalar(Math.sqrt(this.headDim)));
      const mask = tf.tidy(() => {
        const ones = tf.ones([seqLen, seqLen]);
        const tril = tf.linalg.bandPart(ones, -1, 0);
        return tf.where(tril.equal(1), tf.zeros([seqLen, seqLen]), tf.scalar(-1e9));
      });
      scores = scores.add(mask);
      const attnProbs = tf.softmax(scores, -1);
      const context = attnProbs.matMul(v).transpose([0, 2, 1, 3]).reshape([batch, seqLen, this.embDim]);
      return context.reshape([-1, this.embDim]).matMul(this.outDense.read()).reshape([batch, seqLen, this.embDim]);
    });
  }
  static get className() { return 'MultiHeadCausalAttention'; }
}
tf.serialization.registerClass(MultiHeadCausalAttention);

class AddPositionalEmbedding extends tf.layers.Layer {
  constructor(config) { super(config || {}); this.maxSeqLen = config.maxSeqLen; this.embDim = config.embDim; }
  build(inputShape) { this.posEmbedding = this.addWeight('pos_embedding', [this.maxSeqLen, this.embDim], 'float32', tf.initializers.glorotUniform()); this.built = true; }
  computeOutputShape(inputShape) { return inputShape; }
  call(inputs) {
    return tf.tidy(() => {
      const tokenEmbeds = inputs[0] || inputs;
      const seqLen = tokenEmbeds.shape[1];
      return tokenEmbeds.add(this.posEmbedding.read().slice([0, 0], [seqLen, this.embDim]));
    });
  }
  static get className() { return 'AddPositionalEmbedding'; }
}
tf.serialization.registerClass(AddPositionalEmbedding);

class ExtractLastToken extends tf.layers.Layer {
  constructor(config) { super(config || {}); }
  computeOutputShape(inputShape) { return [inputShape[0], inputShape[2]]; }
  call(inputs) {
    return tf.tidy(() => {
      const x = inputs[0] || inputs;
      return x.slice([0, x.shape[1] - 1, 0], [-1, 1, -1]).squeeze([1]);
    });
  }
  static get className() { return 'ExtractLastToken'; }
}
tf.serialization.registerClass(ExtractLastToken);

// ── AKIEModel ──────────────────────────────────────────────────────────────
class AKIEModel {
  constructor(vocab, hparams = HPARAMS) {
    this.vocab = vocab;
    this.hparams = { ...HPARAMS, ...hparams };
    this.model = null;
    this.optimizer = null;
    this.trainSteps = 0;
    this.ready = false;
    this.embeddingVocabSize = 0;
  }

  build() {
    const { embDim, maxSeqLen } = this.hparams;
    const vocabSize = this.vocab.size;
    const numHeads = 4, headDim = Math.floor(embDim / numHeads);

    const input = tf.input({ shape: [maxSeqLen], dtype: 'int32', name: 'context' });
    const tokenEmbeds = tf.layers.embedding({ inputDim: vocabSize, outputDim: embDim, name: 'embedding' }).apply(input);
    const x = new AddPositionalEmbedding({ maxSeqLen, embDim, name: 'pos_embed' }).apply(tokenEmbeds);

    const norm1_1 = tf.layers.layerNormalization({ axis: -1, name: 'norm1_1' }).apply(x);
    const attn1 = new MultiHeadCausalAttention({ numHeads, headDim, name: 'attn_1' }).apply(norm1_1);
    const res1 = tf.layers.add({ name: 'add1_1' }).apply([x, attn1]);
    const norm1_2 = tf.layers.layerNormalization({ axis: -1, name: 'norm1_2' }).apply(res1);
    const ffn1_1 = tf.layers.dense({ units: embDim * 4, activation: 'relu', name: 'ffn1_1' }).apply(norm1_2);
    const ffn1_2 = tf.layers.dense({ units: embDim, name: 'ffn1_2' }).apply(ffn1_1);
    const block1 = tf.layers.add({ name: 'add1_2' }).apply([res1, ffn1_2]);

    const norm2_1 = tf.layers.layerNormalization({ axis: -1, name: 'norm2_1' }).apply(block1);
    const attn2 = new MultiHeadCausalAttention({ numHeads, headDim, name: 'attn_2' }).apply(norm2_1);
    const res2 = tf.layers.add({ name: 'add2_1' }).apply([block1, attn2]);
    const norm2_2 = tf.layers.layerNormalization({ axis: -1, name: 'norm2_2' }).apply(res2);
    const ffn2_1 = tf.layers.dense({ units: embDim * 4, activation: 'relu', name: 'ffn2_1' }).apply(norm2_2);
    const ffn2_2 = tf.layers.dense({ units: embDim, name: 'ffn2_2' }).apply(ffn2_1);
    const block2 = tf.layers.add({ name: 'add2_2' }).apply([res2, ffn2_2]);

    const finalNorm = tf.layers.layerNormalization({ axis: -1, name: 'final_norm' }).apply(block2);
    const lastToken = new ExtractLastToken({ name: 'extract_last' }).apply(finalNorm);
    const output = tf.layers.dense({ units: vocabSize, activation: 'softmax', name: 'output' }).apply(lastToken);

    this.model = tf.model({ inputs: input, outputs: output, name: 'AKIE_Transformer' });
    this.optimizer = tf.train.adam(this.hparams.learningRate);
    this.model.compile({ optimizer: this.optimizer, loss: 'sparseCategoricalCrossentropy', metrics: ['accuracy'] });
    this.ready = true;
    this.embeddingVocabSize = vocabSize;
    console.log(`[AKIE] Transformer Decoder construído. Parâmetros: ${this.model.countParams().toLocaleString()}`);
    console.log(`[AKIE] Vocabulário ativo: ${vocabSize} tokens`);
    return this;
  }

  async trainBatch(pairs, epochs = 3) {
    if (!this.ready) throw new Error('Modelo não construído.');
    if (!pairs || pairs.length === 0) return { loss: null, accuracy: null, steps: 0 };

    const validPairs = pairs.filter(p => {
      if (!Array.isArray(p.x)) return false;
      return p.x.every(id => id < this.embeddingVocabSize) && p.y < this.embeddingVocabSize;
    });
    if (validPairs.length === 0) {
      console.warn('[AKIE] Nenhum par válido. Pulando treino.');
      return { loss: null, accuracy: null, steps: 0 };
    }
    if (validPairs.length < pairs.length) {
      console.warn(`[AKIE] ${pairs.length - validPairs.length} pares descartados.`);
    }

    if (NaturalNLP.isReady && validPairs.length > 10) {
      const sample = validPairs.slice(0, Math.min(50, validPairs.length));
      NaturalNLP.addDocument(sample.map(p => p.x.filter(id => id > 3).join(' ')).join(' '), `batch_${this.trainSteps}`);
    }

    const xs = tf.tensor2d(validPairs.map(p => p.x), [validPairs.length, this.hparams.maxSeqLen], 'int32');
    const ys = tf.tensor1d(validPairs.map(p => p.y), 'float32');
    let lastLoss = null, lastAcc = null;
    try {
      const history = await this.model.fit(xs, ys, { epochs, batchSize: this.hparams.batchSize, shuffle: true, verbose: 0 });
      lastLoss = history.history.loss[history.history.loss.length - 1];
      lastAcc = (history.history.acc || history.history.accuracy)[history.history.loss.length - 1];
      this.trainSteps += validPairs.length * epochs;
    } finally { xs.dispose(); ys.dispose(); }
    return { loss: lastLoss, accuracy: lastAcc, steps: this.trainSteps, samples: validPairs.length };
  }

  generate(prompt = '', maxTokens = null, options = {}) {
    if (!this.ready) return null;
    const maxT = maxTokens || this.hparams.maxGenTokens;
    const beamSize = options.beamSize || this.hparams.beamSize;
    const repPenalty = options.repetitionPenalty || this.hparams.repetitionPenalty;
    const temp = options.temperature != null ? options.temperature : this.hparams.temperature;

    const contextIds = this.vocab.tokenize(prompt).filter(id => id !== SPECIAL.EOS);
    let beams = [{ score: 0, ids: [...contextIds], generatedIds: [] }];
    const completedBeams = [];
    const padLen = this.hparams.maxSeqLen;

    for (let step = 0; step < maxT; step++) {
      if (!beams.length) break;
      const paddedBatch = beams.map(b => [...Array(Math.max(0, padLen - b.ids.length)).fill(SPECIAL.PAD), ...b.ids.slice(-padLen)]);
      const inputTensor = tf.tensor2d(paddedBatch, [beams.length, padLen], 'int32');
      const rawLogits = this.model.predict(inputTensor);
      const logitsTensor = rawLogits.dtype === 'float32' ? rawLogits : rawLogits.cast('float32');
      const logitsArray = logitsTensor.arraySync();
      inputTensor.dispose();
      if (logitsTensor !== rawLogits) rawLogits.dispose();
      logitsTensor.dispose();

      const candidates = [];
      for (let b = 0; b < beams.length; b++) {
        const beam = beams[b];
        let logits = logitsArray[b];
        if (repPenalty !== 1.0) {
          const seen = new Set(beam.ids);
          logits = logits.map((l, i) => seen.has(i) ? (l > 0 ? l / repPenalty : l * repPenalty) : l);
        }
        if (temp > 0 && temp !== 1.0) logits = logits.map(l => l / temp);
        const maxLogit = Math.max(...logits);
        const exps = logits.map(l => Math.exp(l - maxLogit));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        const logSoftmax = logits.map(l => (l - maxLogit) - Math.log(sumExps));
        logSoftmax.map((lp, idx) => ({ lp, idx })).sort((a, b) => b.lp - a.lp).slice(0, beamSize).forEach(c => {
          candidates.push({ score: beam.score + c.lp, ids: [...beam.ids, c.idx], generatedIds: [...beam.generatedIds, c.idx] });
        });
      }
      candidates.sort((a, b) => b.score - a.score);
      beams = [];
      for (const cand of candidates) {
        if (beams.length >= beamSize) break;
        cand.generatedIds[cand.generatedIds.length - 1] === SPECIAL.EOS ? completedBeams.push(cand) : beams.push(cand);
      }
    }

    const finalSelection = [...completedBeams, ...beams];
    if (!finalSelection.length || !finalSelection[0]) return '';
    if (NaturalNLP.isReady && finalSelection.length > 1) {
      finalSelection.forEach(c => {
        const gen = this.vocab.detokenize(c.generatedIds.filter(id => ![SPECIAL.EOS, SPECIAL.PAD, SPECIAL.BOS].includes(id)));
        c.combinedScore = c.score * 0.85 + NaturalNLP.generationScore(prompt, gen) * 0.15;
      });
      finalSelection.sort((a, b) => b.combinedScore - a.combinedScore);
    } else {
      finalSelection.sort((a, b) => b.score - a.score);
    }
    const best = finalSelection[0];
    if (!best.generatedIds?.length) return '';
    const tokens = best.generatedIds.filter(id => ![SPECIAL.EOS, SPECIAL.PAD, SPECIAL.BOS].includes(id));
    return tokens.length ? this.vocab.detokenize(tokens) : '';
  }

  async save(dir) {
    if (!this.ready) return;
    await fs.promises.mkdir(dir, { recursive: true });
    await this.model.save(`file://${dir}`);
    const meta = { trainSteps: this.trainSteps, vocabSize: this.vocab.size, embeddingVocabSize: this.embeddingVocabSize, hparams: this.hparams, savedAt: new Date().toISOString() };
    await fs.promises.writeFile(path.join(dir, 'akie_meta.json'), JSON.stringify(meta, null, 2));
    await fs.promises.writeFile(path.join(dir, 'akie_vocab.json'), JSON.stringify(this.vocab.toJSON(), null, 2));
    console.log(`[AKIE] Transformer salvo em ${dir} (steps: ${this.trainSteps})`);
  }

  async load(dir) {
    try {
      const modelPath = path.join(dir, 'model.json');
      console.log(`[AKIE] Procurando modelo em: ${modelPath}`);
      console.log(`[AKIE] Diretório ${dir} existe? ${fs.existsSync(dir)}`);
      if (fs.existsSync(dir)) console.log(`[AKIE] Arquivos no diretório: ${fs.readdirSync(dir).join(', ') || 'nenhum'}`);
      if (!fs.existsSync(modelPath)) { console.log(`[AKIE] Nenhum modelo encontrado.`); return false; }

      const metaPath = path.join(dir, 'akie_meta.json');
      let savedVocabSize = this.vocab.size;
      if (fs.existsSync(metaPath)) {
        const meta = JSON.parse(await fs.promises.readFile(metaPath, 'utf8'));
        this.trainSteps = meta.trainSteps || 0;
        savedVocabSize = meta.vocabSize || this.vocab.size;
      }

      if (this.vocab.size > savedVocabSize) {
        console.log(`[AKIE] Vocab cresceu. Carregando e expandindo...`);
        this.model = await tf.loadLayersModel(`file://${dir}/model.json`);
        this.optimizer = tf.train.adam(this.hparams.learningRate);
        this.model.compile({ optimizer: this.optimizer, loss: 'sparseCategoricalCrossentropy', metrics: ['accuracy'] });
        try { this.embeddingVocabSize = this.model.getLayer('embedding').getWeights()[0].shape[0]; } catch { this.embeddingVocabSize = savedVocabSize; }
        this.ready = true;
        if (this.vocab.size > this.embeddingVocabSize) await this.expandVocabulary(this.vocab.size);
        console.log(`[AKIE] Transformer carregado e expandido.`);
        return true;
      }

      this.model = await tf.loadLayersModel(`file://${dir}/model.json`);
      this.optimizer = tf.train.adam(this.hparams.learningRate);
      this.model.compile({ optimizer: this.optimizer, loss: 'sparseCategoricalCrossentropy', metrics: ['accuracy'] });
      try { this.embeddingVocabSize = this.model.getLayer('embedding').getWeights()[0].shape[0]; } catch { this.embeddingVocabSize = this.vocab.size; }
      this.ready = true;
      console.log(`[AKIE] Transformer carregado. (steps: ${this.trainSteps})`);
      return true;
    } catch (err) {
      console.error(`[AKIE] Falha ao carregar modelo: ${err.message}`);
      console.log('[AKIE] Modelo corrompido. Limpando diretório...');
      try {
        for (const file of fs.readdirSync(dir)) fs.unlinkSync(path.join(dir, file));
        console.log('[AKIE] Diretório limpo.');
      } catch (cleanErr) { console.error('[AKIE] Erro ao limpar:', cleanErr.message); }
      return false;
    }
  }

  async expandVocabulary(newVocabSize) {
    if (!this.ready || newVocabSize <= this.embeddingVocabSize) return;
    console.log(`[AKIE] Expandindo embedding: ${this.embeddingVocabSize} → ${newVocabSize}`);
    const oldWeights = this.model.getWeights();
    this.embeddingVocabSize = newVocabSize;
    this.build();
    const newWeights = this.model.getWeights();
    const updatedWeights = newWeights.map((w, i) => {
      const oldW = oldWeights[i];
      if (!oldW) return w;
      const oldShape = oldW.shape, newShape = w.shape;
      if (oldShape.length === newShape.length && oldShape.every((d, j) => d === newShape[j])) return oldW;
      if (oldShape.length === 2 && newShape.length === 2 && newShape[0] > oldShape[0] && oldShape[1] === newShape[1]) {
        const merged = new Float32Array(w.dataSync()); merged.set(oldW.dataSync()); return tf.tensor(merged, newShape, 'float32');
      }
      if (oldShape.length === 2 && newShape.length === 2 && newShape[1] > oldShape[1] && oldShape[0] === newShape[0]) {
        const oldData = oldW.dataSync(), merged = new Float32Array(w.dataSync());
        const [rows, oldCols] = oldShape, newCols = newShape[1];
        for (let r = 0; r < rows; r++) for (let c = 0; c < oldCols; c++) merged[r * newCols + c] = oldData[r * oldCols + c];
        return tf.tensor(merged, newShape, 'float32');
      }
      if (oldShape.length === 1 && newShape.length === 1 && newShape[0] > oldShape[0]) {
        const merged = new Float32Array(w.dataSync()); merged.set(oldW.dataSync()); return tf.tensor(merged, newShape, 'float32');
      }
      return w;
    });
    this.model.setWeights(updatedWeights);
    oldWeights.forEach(w => { try { w.dispose(); } catch {} });
    console.log('[AKIE] Expansão concluída.');
  }

  getStats() {
    return {
      architecture: 'Causal Transformer Decoder', ready: this.ready,
      vocabSize: this.vocab.size, embeddingVocabSize: this.embeddingVocabSize,
      trainSteps: this.trainSteps, parameters: this.ready ? this.model.countParams() : 0,
      hparams: this.hparams, nlp: { natural: NaturalNLP.isReady },
    };
  }

  static stemTokens(tokens) { return tokens.map(t => NaturalNLP.stem(t)); }
}

module.exports = { AKIEModel, HPARAMS, NaturalNLP };
