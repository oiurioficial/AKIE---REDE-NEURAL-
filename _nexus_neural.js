/**
 * _nexus_neural.js  —  AKIE Neural Core v2.0 (Upgrade Moderado)
 *
 * MUDANÇAS ESTRUTURAIS:
 *   embDim:    64 → 128     (2x)
 *   hiddenSize: 128 → 256   (FFN interno dos blocos Transformer)
 *   maxSeqLen:  32 → 64     (2x contexto)
 *
 * Parâmetros: ~260k → ~1.95M
 *
 * CORREÇÃO v2.0.1:
 *   - Removido ff_expand mal posicionado antes da atenção (causava reshape inválido)
 *   - Arquitetura Transformer Decoder correta: Embed → PosEmbed → [Norm → Attn → Res → Norm → FFN → Res] x2 → Norm → ExtractLast → Output
 *   - MultiHeadCausalAttention agora recebe embDim correto em todos os caminhos
 */

const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const { Vocabulary, makeTrainingPairs, SPECIAL } = require('./_akie_vocab');

// ---------------------------------------------------------------------------
// NLP Layer
// ---------------------------------------------------------------------------

let NaturalNLP = null;

try {
  const natural = require('natural');
  const stemmer = natural.PorterStemmerPt;
  const tfidf = new natural.TfIdf();
  const JaroWinklerDistance = natural.JaroWinklerDistance;

  NaturalNLP = {
    stem(token) { try { return stemmer.stem(token); } catch { return token; } },
    stemSentence(sentence) {
      try { return stemmer.tokenizeAndStem(sentence, false); }
      catch { return sentence.toLowerCase().split(/\s+/).filter(Boolean); }
    },
    addDocument(text, key) { try { tfidf.addDocument(text, key); } catch { /* ignora */ } },
    topTerms(docIndex, n = 5) {
      try { const terms = []; tfidf.listTerms(docIndex).slice(0, n).forEach(t => terms.push(t.term)); return terms; }
      catch { return []; }
    },
    similarity(a, b) { try { return JaroWinklerDistance(a, b); } catch { return 0; } },
    generationScore(prompt, generated) {
      if (!prompt || !generated) return 0;
      const promptStems = stemmer.tokenizeAndStem(prompt, false);
      const genStems = stemmer.tokenizeAndStem(generated, false);
      if (!promptStems.length || !genStems.length) return 0;
      const overlap = promptStems.filter(s => genStems.includes(s)).length;
      return overlap / promptStems.length;
    },
    isReady: true,
  };
  console.log('[NLP] Biblioteca `natural` carregada.');
} catch (e) {
  NaturalNLP = {
    isReady: false,
    stem: t => t,
    stemSentence: s => s.split(/\s+/),
    similarity: () => 0,
    generationScore: () => 0,
    addDocument: () => {},
    topTerms: () => [],
  };
  console.log('[NLP] `natural` não encontrada.');
}

module.exports.NaturalNLP = NaturalNLP;

// ---------------------------------------------------------------------------
// Hiperparâmetros — v2.0
// ---------------------------------------------------------------------------

const HPARAMS = {
  embDim:            128,   // dimensão dos embeddings (numHeads * headDim = 8 * 16)
  hiddenSize:        256,   // FFN interno dos blocos (embDim * 2)
  maxSeqLen:          64,   // comprimento máximo de contexto
  batchSize:           4,   // batches pequenos para RAM limitada no Railway
  learningRate:    0.0005,
  temperature:       0.8,
  maxGenTokens:       40,
  minNewData:          3,
  beamSize:            3,
  repetitionPenalty: 1.2,
  modelVersion:     '2.0',
};

// ---------------------------------------------------------------------------
// Camadas Customizadas
// ---------------------------------------------------------------------------

/**
 * Multi-Head Causal Attention.
 * IMPORTANTE: espera entrada com última dimensão == embDim (não hiddenSize).
 * numHeads=8, headDim=16 → embDim=128. Sempre consistente com HPARAMS.embDim.
 */
class MultiHeadCausalAttention extends tf.layers.Layer {
  constructor(config) {
    super(config || {});
    this.numHeads = config.numHeads || 8;
    this.headDim  = config.headDim  || 16;
    this.embDim   = this.numHeads * this.headDim; // 8 * 16 = 128
  }

  build(inputShape) {
    this.qDense   = this.addWeight('q_proj',   [this.embDim, this.embDim], 'float32', tf.initializers.glorotUniform());
    this.kDense   = this.addWeight('k_proj',   [this.embDim, this.embDim], 'float32', tf.initializers.glorotUniform());
    this.vDense   = this.addWeight('v_proj',   [this.embDim, this.embDim], 'float32', tf.initializers.glorotUniform());
    this.outDense = this.addWeight('out_proj', [this.embDim, this.embDim], 'float32', tf.initializers.glorotUniform());
    this.built = true;
  }

  computeOutputShape(inputShape) { return inputShape; }

  call(inputs) {
    return tf.tidy(() => {
      const x = inputs[0] || inputs;
      const [batch, seqLen] = x.shape;

      // x tem shape [batch, seqLen, embDim=128] — reshape seguro
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

      const attentionProbs = tf.softmax(scores, -1);
      const context = attentionProbs.matMul(v).transpose([0, 2, 1, 3]).reshape([batch, seqLen, this.embDim]);
      const outFlat = context.reshape([-1, this.embDim]);
      return outFlat.matMul(this.outDense.read()).reshape([batch, seqLen, this.embDim]);
    });
  }

  static get className() { return 'MultiHeadCausalAttention'; }
}
tf.serialization.registerClass(MultiHeadCausalAttention);

/**
 * Positional Embedding treinável.
 */
class AddPositionalEmbedding extends tf.layers.Layer {
  constructor(config) {
    super(config || {});
    this.maxSeqLen = config.maxSeqLen;
    this.embDim    = config.embDim;
  }

  build(inputShape) {
    this.posEmbedding = this.addWeight('pos_embedding', [this.maxSeqLen, this.embDim], 'float32', tf.initializers.glorotUniform());
    this.built = true;
  }

  computeOutputShape(inputShape) { return inputShape; }

  call(inputs) {
    return tf.tidy(() => {
      const tokenEmbeds = inputs[0] || inputs;
      const seqLen = tokenEmbeds.shape[1];
      const posEmbeds = this.posEmbedding.read().slice([0, 0], [seqLen, this.embDim]);
      return tokenEmbeds.add(posEmbeds);
    });
  }

  static get className() { return 'AddPositionalEmbedding'; }
}
tf.serialization.registerClass(AddPositionalEmbedding);

/**
 * TokenEmbedding customizado (compatível com tfjs 4.x).
 * Usa tf.gather em vez da camada Embedding nativa para evitar issues de dtype.
 */
class TokenEmbedding extends tf.layers.Layer {
  constructor(config) {
    super(config || {});
    this.inputDim    = config.inputDim;
    this.outputDim   = config.outputDim;
    this.inputLength = config.inputLength;
  }

  build(inputShape) {
    this.embeddings = this.addWeight(
      'embeddings',
      [this.inputDim, this.outputDim],
      'float32',
      tf.initializers.glorotUniform()
    );
    this.built = true;
  }

  computeOutputShape(inputShape) {
    return [inputShape[0], this.inputLength, this.outputDim];
  }

  call(inputs) {
    return tf.tidy(() => {
      const input = inputs[0] || inputs;
      const [batch, seqLen] = input.shape;
      const flatInput = input.reshape([-1]);
      const embedded = tf.gather(this.embeddings.read(), flatInput);
      return embedded.reshape([batch, seqLen, this.outputDim]);
    });
  }

  static get className() { return 'TokenEmbedding'; }
}
tf.serialization.registerClass(TokenEmbedding);

/**
 * Extrai o último token da sequência para predição autoregressiva.
 */
class ExtractLastToken extends tf.layers.Layer {
  constructor(config) { super(config || {}); }
  computeOutputShape(inputShape) { return [inputShape[0], inputShape[2]]; }
  call(inputs) {
    return tf.tidy(() => {
      const x = inputs[0] || inputs;
      const seqLen = x.shape[1];
      return x.slice([0, seqLen - 1, 0], [-1, 1, -1]).squeeze([1]);
    });
  }
  static get className() { return 'ExtractLastToken'; }
}
tf.serialization.registerClass(ExtractLastToken);

// ---------------------------------------------------------------------------
// Helper: cópia recursiva de diretório
// ---------------------------------------------------------------------------

async function _copyDir(src, dst) {
  await fs.promises.mkdir(dst, { recursive: true });
  const entries = await fs.promises.readdir(src, { withFileTypes: true });
  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const dstPath = path.join(dst, entry.name);
    if (entry.isDirectory()) {
      await _copyDir(srcPath, dstPath);
    } else {
      await fs.promises.copyFile(srcPath, dstPath);
    }
  }
}

// ---------------------------------------------------------------------------
// AKIEModel v2.0
// ---------------------------------------------------------------------------

class AKIEModel {
  constructor(vocab, hparams = HPARAMS) {
    this.vocab   = vocab;
    this.hparams = { ...HPARAMS, ...hparams };
    this.model   = null;
    this.optimizer = null;
    this.trainSteps = 0;
    this.ready   = false;
    this.embeddingVocabSize = 0;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // build() — Transformer Decoder correto
  //
  // Fluxo:
  //   input [batch, maxSeqLen int32]
  //   → TokenEmbedding          → [batch, maxSeqLen, embDim=128]
  //   → AddPositionalEmbedding  → [batch, maxSeqLen, 128]
  //   ── Bloco 1 ──
  //   → LayerNorm               → [batch, maxSeqLen, 128]
  //   → MultiHeadCausalAttention → [batch, maxSeqLen, 128]
  //   → Add (residual)          → [batch, maxSeqLen, 128]
  //   → LayerNorm               → [batch, maxSeqLen, 128]
  //   → Dense(hiddenSize=256, relu) → [batch, maxSeqLen, 256]
  //   → Dense(embDim=128)       → [batch, maxSeqLen, 128]
  //   → Add (residual)          → [batch, maxSeqLen, 128]
  //   ── Bloco 2 (idêntico) ──
  //   → LayerNorm final         → [batch, maxSeqLen, 128]
  //   → ExtractLastToken        → [batch, 128]
  //   → Dense(vocabSize, softmax) → [batch, vocabSize]
  // ─────────────────────────────────────────────────────────────────────────

  build() {
    const { embDim, hiddenSize, maxSeqLen } = this.hparams;
    const vocabSize = this.embeddingVocabSize > 0 ? this.embeddingVocabSize : this.vocab.size;
    const numHeads  = 8;
    const headDim   = Math.floor(embDim / numHeads); // 128 / 8 = 16

    const input = tf.input({ shape: [maxSeqLen], dtype: 'int32', name: 'context' });

    // 1. Token Embedding: [batch, maxSeqLen] → [batch, maxSeqLen, 128]
    const tokenEmbedded = new TokenEmbedding({
      inputDim:    vocabSize,
      outputDim:   embDim,
      inputLength: maxSeqLen,
      name: 'token_embedding',
    }).apply(input);

    // 2. Positional Embedding: soma posições → [batch, maxSeqLen, 128]
    let x = new AddPositionalEmbedding({
      maxSeqLen,
      embDim,
      name: 'pos_embedding',
    }).apply(tokenEmbedded);

    // ── Bloco Transformer Decoder 1 ──────────────────────────────────────
    // Pre-norm → Atenção → Residual
    const norm1_1 = tf.layers.layerNormalization({ epsilon: 1e-5, name: 'norm1_1' }).apply(x);
    const attn1   = new MultiHeadCausalAttention({ numHeads, headDim, name: 'attn_1' }).apply(norm1_1);
    const res1    = tf.layers.add({ name: 'add1_1' }).apply([x, attn1]);

    // Pre-norm → FFN (embDim→hiddenSize→embDim) → Residual
    const norm1_2 = tf.layers.layerNormalization({ epsilon: 1e-5, name: 'norm1_2' }).apply(res1);
    const ffn1_up = tf.layers.dense({ units: hiddenSize, activation: 'relu', name: 'ffn1_up' }).apply(norm1_2);
    const ffn1_dn = tf.layers.dense({ units: embDim, name: 'ffn1_dn' }).apply(ffn1_up);
    const block1  = tf.layers.add({ name: 'add1_2' }).apply([res1, ffn1_dn]);

    // ── Bloco Transformer Decoder 2 ──────────────────────────────────────
    const norm2_1 = tf.layers.layerNormalization({ epsilon: 1e-5, name: 'norm2_1' }).apply(block1);
    const attn2   = new MultiHeadCausalAttention({ numHeads, headDim, name: 'attn_2' }).apply(norm2_1);
    const res2    = tf.layers.add({ name: 'add2_1' }).apply([block1, attn2]);

    const norm2_2 = tf.layers.layerNormalization({ epsilon: 1e-5, name: 'norm2_2' }).apply(res2);
    const ffn2_up = tf.layers.dense({ units: hiddenSize, activation: 'relu', name: 'ffn2_up' }).apply(norm2_2);
    const ffn2_dn = tf.layers.dense({ units: embDim, name: 'ffn2_dn' }).apply(ffn2_up);
    const block2  = tf.layers.add({ name: 'add2_2' }).apply([res2, ffn2_dn]);

    // 3. Normalização final + extração do último token
    const finalNorm = tf.layers.layerNormalization({ epsilon: 1e-5, name: 'final_norm' }).apply(block2);
    const lastToken = new ExtractLastToken({ name: 'extract_last' }).apply(finalNorm);

    // 4. Projeção para vocabulário com softmax
    const output = tf.layers.dense({
      units: vocabSize,
      activation: 'softmax',
      name: 'output',
    }).apply(lastToken);

    this.model = tf.model({ inputs: input, outputs: output });
    this.optimizer = tf.train.adam(this.hparams.learningRate);
    this.model.compile({
      optimizer: this.optimizer,
      loss: 'sparseCategoricalCrossentropy',
      metrics: ['accuracy'],
    });

    this.ready = true;
    if (this.embeddingVocabSize === 0) this.embeddingVocabSize = vocabSize;

    console.log(`[AKIE] Transformer v2.0 construído: embDim=${embDim}, hiddenSize=${hiddenSize}, maxSeqLen=${maxSeqLen}, vocab=${vocabSize}, params=${this.model.countParams().toLocaleString()}`);
    return this;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // trainBatch — loop manual por mini-batch (RAM-safe com batchSize=4)
  // ─────────────────────────────────────────────────────────────────────────

  async trainBatch(pairs, epochs = 3) {
    if (!this.ready || !pairs.length) return { loss: null, accuracy: null };

    const { batchSize } = this.hparams;
    let totalLoss = 0;
    let totalAcc  = 0;
    let steps     = 0;

    for (let ep = 0; ep < epochs; ep++) {
      for (let i = 0; i < pairs.length; i += batchSize) {
        const batch = pairs.slice(i, i + batchSize);
        if (!batch.length) continue;

        const seqLen = batch[0].x.length;
        const xs = tf.tensor2d(batch.map(p => p.x), [batch.length, seqLen], 'int32');
        const ys = tf.tensor1d(batch.map(p => p.y), 'int32');

        let result;
        try {
          result = this.model.trainOnBatch(xs, ys);
          if (result && typeof result.then === 'function') result = await result;
        } finally {
          xs.dispose();
          ys.dispose();
        }

        const _toNum = (v) => {
          if (v == null) return null;
          if (typeof v === 'number') return isFinite(v) ? v : null;
          if (Array.isArray(v)) return _toNum(v[0]);
          if (typeof v.dataSync === 'function') {
            try { const s = v.dataSync()[0]; v.dispose?.(); return isFinite(s) ? s : null; }
            catch { return null; }
          }
          return null;
        };

        const batchLoss = Array.isArray(result) ? _toNum(result[0]) : _toNum(result);
        const batchAcc  = Array.isArray(result) && result.length > 1 ? _toNum(result[1]) : null;

        if (batchLoss !== null) { totalLoss += batchLoss; steps++; }
        if (batchAcc  !== null) totalAcc += batchAcc;
      }
    }

    this.trainSteps += pairs.length * epochs;

    const loss     = steps > 0 ? totalLoss / steps : null;
    const accuracy = steps > 0 && totalAcc > 0 ? totalAcc / steps : null;
    return { loss, accuracy };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // generate — Beam Search com penalização de repetição
  // ─────────────────────────────────────────────────────────────────────────

  async generate(prompt, maxTokens = null, temperature = null) {
    if (!this.ready) return '';

    maxTokens   = maxTokens   || this.hparams.maxGenTokens;
    temperature = temperature || this.hparams.temperature;

    const ids   = this.vocab.tokenize(prompt);
    const beams = [{ ids: ids.slice(-this.hparams.maxSeqLen), logProb: 0 }];

    for (let step = 0; step < maxTokens; step++) {
      const nextBeams = [];

      for (const beam of beams) {
        const padded = [
          ...Array(Math.max(0, this.hparams.maxSeqLen - beam.ids.length)).fill(SPECIAL.PAD),
          ...beam.ids,
        ].slice(-this.hparams.maxSeqLen);

        const seqTensor = tf.tensor2d([padded], [1, this.hparams.maxSeqLen], 'int32');
        const logits    = this.model.predict(seqTensor);
        const logitsData = logits.dataSync();
        seqTensor.dispose();
        logits.dispose();

        // Penalização de repetição
        const seenSet = new Set(beam.ids);
        const probs = Array.from(logitsData).map((l, i) => {
          let adj = l;
          if (seenSet.has(i)) adj = adj > 0 ? adj / this.hparams.repetitionPenalty : adj * this.hparams.repetitionPenalty;
          return { id: i, logprob: Math.log(Math.max(1e-10, adj / temperature)) };
        });

        probs.sort((a, b) => b.logprob - a.logprob);

        for (let k = 0; k < this.hparams.beamSize && k < probs.length; k++) {
          const nextId = probs[k].id;
          nextBeams.push({
            ids:     [...beam.ids, nextId],
            logProb: beam.logProb + probs[k].logprob,
          });
          if (nextId === SPECIAL.EOS) break;
        }
      }

      nextBeams.sort((a, b) => b.logProb - a.logProb);
      beams.splice(0, beams.length, ...nextBeams.slice(0, this.hparams.beamSize));
    }

    const best = beams.reduce((b, c) => c.logProb > b.logProb ? c : b, beams[0]);
    const out  = best.ids.filter(id => id !== SPECIAL.EOS && id !== SPECIAL.PAD && id !== SPECIAL.BOS);
    return out.length ? this.vocab.detokenize(out) : '';
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Persistência — salva pesos em binário customizado (mais rápido que tf.LayersModel save)
  // ─────────────────────────────────────────────────────────────────────────

  async save(dir) {
    if (!this.ready) return;

    const tmpDir = `${dir}_tmp`;
    const bakDir = `${dir}_bak`;

    await fs.promises.rm(tmpDir, { recursive: true, force: true });
    await fs.promises.mkdir(tmpDir, { recursive: true });

    const weights   = this.model.getWeights();
    const manifest  = weights.map(w => ({ name: w.name, shape: w.shape }));
    const allData   = weights.map(w => w.dataSync());
    const totalFloats = allData.reduce((sum, arr) => sum + arr.length, 0);
    const buffer    = Buffer.allocUnsafe(totalFloats * 4);

    let offset = 0;
    for (const arr of allData) {
      for (const v of arr) { buffer.writeFloatLE(v, offset); offset += 4; }
    }

    await fs.promises.writeFile(path.join(tmpDir, 'weight_manifest.json'), JSON.stringify(manifest));
    await fs.promises.writeFile(path.join(tmpDir, 'weights.bin'), buffer);

    const meta = {
      trainSteps:         this.trainSteps,
      vocabSize:          this.vocab.size,
      embeddingVocabSize: this.embeddingVocabSize,
      hparams:            this.hparams,
      savedAt:            new Date().toISOString(),
      format:             'weights_bin_v2',
      modelVersion:       this.hparams.modelVersion,
    };
    await fs.promises.writeFile(path.join(tmpDir, 'akie_meta.json'), JSON.stringify(meta, null, 2));
    await fs.promises.writeFile(path.join(tmpDir, 'akie_vocab.json'), JSON.stringify(this.vocab.toJSON(), null, 2));

    if (fs.existsSync(path.join(dir, 'akie_meta.json'))) {
      await fs.promises.rm(bakDir, { recursive: true, force: true });
      await fs.promises.rename(dir, bakDir);
    } else {
      await fs.promises.rm(dir, { recursive: true, force: true });
    }
    await fs.promises.rename(tmpDir, dir);

    console.log(`[AKIE] ✓ Modelo v2.0 salvo em ${dir} (steps: ${this.trainSteps})`);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // load — carrega binário com validação de compatibilidade
  // ─────────────────────────────────────────────────────────────────────────

  async load(dir) {
    const bakDir = `${dir}_bak`;

    for (const tryDir of [dir, bakDir]) {
      const metaPath     = path.join(tryDir, 'akie_meta.json');
      const manifestPath = path.join(tryDir, 'weight_manifest.json');
      const weightsPath  = path.join(tryDir, 'weights.bin');

      if (!fs.existsSync(metaPath) || !fs.existsSync(manifestPath) || !fs.existsSync(weightsPath)) {
        const files = fs.existsSync(tryDir) ? fs.readdirSync(tryDir) : [];
        console.log(`[AKIE] Sem modelo em ${tryDir}. Arquivos: ${files.join(', ') || 'nenhum'}`);
        continue;
      }

      try {
        const meta     = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
        const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

        // Validação de compatibilidade de arquitetura
        const savedEmbDim    = meta.hparams?.embDim    || 64;
        const savedHidden    = meta.hparams?.hiddenSize || 128;
        const savedMaxSeqLen = meta.hparams?.maxSeqLen  || 32;

        if (
          savedEmbDim    !== this.hparams.embDim    ||
          savedHidden    !== this.hparams.hiddenSize ||
          savedMaxSeqLen !== this.hparams.maxSeqLen
        ) {
          console.warn(`[AKIE] ⚠️  Incompatibilidade: salvo=[emb=${savedEmbDim}|hid=${savedHidden}|seq=${savedMaxSeqLen}] vs atual=[emb=${this.hparams.embDim}|hid=${this.hparams.hiddenSize}|seq=${this.hparams.maxSeqLen}]`);
          console.warn('[AKIE] → Reinicializando com nova arquitetura...');
          this.build();
          return false;
        }

        this.trainSteps         = meta.trainSteps || 0;
        this.embeddingVocabSize = meta.embeddingVocabSize || meta.vocabSize || this.vocab.size;

        // Construir shell com vocab do tamanho salvo para carregar pesos corretamente
        const shellVocab = { ...this.vocab, size: this.embeddingVocabSize };
        const shell = new AKIEModel(shellVocab, this.hparams);
        shell.embeddingVocabSize = this.embeddingVocabSize;
        shell.build();

        const buffer  = fs.readFileSync(weightsPath);
        let bufOffset = 0;
        const tensors = manifest.map(w => {
          const numFloats = w.shape.reduce((a, b) => a * b, 1);
          const arr = new Float32Array(numFloats);
          for (let i = 0; i < numFloats; i++) {
            arr[i] = buffer.readFloatLE(bufOffset); bufOffset += 4;
          }
          return tf.tensor(arr, w.shape, 'float32');
        });

        shell.model.setWeights(tensors);
        tensors.forEach(t => t.dispose());

        this.model = shell.model;
        this.optimizer = tf.train.adam(this.hparams.learningRate);
        this.model.compile({
          optimizer: this.optimizer,
          loss: 'sparseCategoricalCrossentropy',
          metrics: ['accuracy'],
        });
        this.ready = true;

        if (tryDir === bakDir) {
          console.warn(`[AKIE] ✓ Recuperado do backup. (steps: ${this.trainSteps})`);
        } else {
          console.log(`[AKIE] ✓ Transformer v2.0 carregado. (steps: ${this.trainSteps}, embVocab: ${this.embeddingVocabSize})`);
        }

        // Expandir se vocab cresceu desde o último save
        if (this.vocab.size > this.embeddingVocabSize) {
          console.log(`[AKIE] Vocab cresceu (${this.embeddingVocabSize} → ${this.vocab.size}). Expandindo...`);
          await this.expandVocabulary(this.vocab.size);
        }

        return true;

      } catch (err) {
        console.error(`[AKIE] Falha ao carregar de ${tryDir}: ${err.message}`);
        if (tryDir === dir) {
          await fs.promises.rm(dir, { recursive: true, force: true }).catch(() => {});
        }
      }
    }

    console.log('[AKIE] ✗ Nenhum modelo encontrado. Inicializando novo...');
    this.build();
    return false;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // expandVocabulary — redimensiona embedding e output preservando pesos
  // ─────────────────────────────────────────────────────────────────────────

  async expandVocabulary(newVocabSize) {
    if (!this.ready) return;
    if (newVocabSize <= this.embeddingVocabSize) return;

    console.log(`[AKIE] Expandindo vocab: ${this.embeddingVocabSize} → ${newVocabSize}`);

    const oldWeights = this.model.getWeights();
    this.embeddingVocabSize = newVocabSize;
    this.build();

    const newWeights = this.model.getWeights();
    const updated = newWeights.map((w, i) => {
      const oldW = oldWeights[i];
      if (!oldW) return w;
      const os = oldW.shape, ns = w.shape;
      if (JSON.stringify(os) === JSON.stringify(ns)) return oldW;

      // Embedding: [vocabSize, embDim] — linha a linha
      if (os.length === 2 && ns.length === 2 && ns[0] > os[0] && os[1] === ns[1]) {
        const merged = new Float32Array(w.dataSync());
        merged.set(oldW.dataSync());
        return tf.tensor(merged, ns, 'float32');
      }
      // Output Dense kernel: [embDim, vocabSize]
      if (os.length === 2 && ns.length === 2 && ns[1] > os[1] && os[0] === ns[0]) {
        const oldData = oldW.dataSync(), merged = new Float32Array(w.dataSync());
        const [rows, oldCols] = os, newCols = ns[1];
        for (let r = 0; r < rows; r++)
          for (let c = 0; c < oldCols; c++)
            merged[r * newCols + c] = oldData[r * oldCols + c];
        return tf.tensor(merged, ns, 'float32');
      }
      // Output Dense bias: [vocabSize]
      if (os.length === 1 && ns.length === 1 && ns[0] > os[0]) {
        const merged = new Float32Array(w.dataSync());
        merged.set(oldW.dataSync());
        return tf.tensor(merged, ns, 'float32');
      }

      return w;
    });

    this.model.setWeights(updated);
    oldWeights.forEach(w => { try { w.dispose(); } catch {} });
    console.log('[AKIE] ✓ Expansão de vocab concluída.');
  }

  getStats() {
    return {
      architecture:       'Causal Transformer Decoder v2.0',
      ready:              this.ready,
      vocabSize:          this.vocab.size,
      embeddingVocabSize: this.embeddingVocabSize,
      trainSteps:         this.trainSteps,
      parameters:         this.ready ? this.model.countParams() : 0,
      hparams:            this.hparams,
      nlp:                { natural: NaturalNLP.isReady },
    };
  }

  static stemTokens(tokens) { return tokens.map(t => NaturalNLP.stem(t)); }
}

module.exports = { AKIEModel, HPARAMS, NaturalNLP };
