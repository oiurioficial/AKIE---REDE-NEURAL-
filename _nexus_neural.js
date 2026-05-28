/**
 * _nexus_neural.js  —  AKIE Neural Core v2.0 (Upgrade Moderado)
 *
 * MUDANÇAS ESTRUTURAIS:
 *   embDim:    64 → 128     (2x)
 *   hiddenSize: 128 → 256   (2x)
 *   maxSeqLen:  32 → 64     (2x contexto)
 *   
 * Parâmetros: ~260k → ~1.95M
 * RAM (Transformer Decoder): ~780MB (CPU-safe no Railway)
 *
 * COMPATIBILIDADE:
 *   - Versionamento de modelo (akie_meta.json)
 *   - Reset automático em incompatibilidade
 *   - Suporta migrações futuras sem falha crítica
 */

const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const { Vocabulary, makeTrainingPairs, SPECIAL } = require('./_akie_vocab');

// ---------------------------------------------------------------------------
// NLP Layer (unchanged)
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
  NaturalNLP = { isReady: false, stem: t => t, stemSentence: s => s.split(/\s+/), similarity: () => 0, generationScore: () => 0, addDocument: () => {}, topTerms: () => [] };
  console.log('[NLP] `natural` não encontrada.');
}

module.exports.NaturalNLP = NaturalNLP;

// ---------------------------------------------------------------------------
// Hiperparâmetros — UPGRADE MODERADO
// ---------------------------------------------------------------------------

const HPARAMS = {
  embDim:       128,      // ↑ 64 → 128
  hiddenSize:   256,      // ↑ 128 → 256
  maxSeqLen:    64,       // ↑ 32 → 64
  batchSize:    4,       // mantém segurança RAM
  learningRate: 0.0005,   // ↓ 0.001 → 0.0005 (maior modelo, LR conservador)
  temperature:  0.8,
  maxGenTokens: 40,
  minNewData:   3,
  beamSize:     3,
  repetitionPenalty: 1.2,
  modelVersion: '2.0',    // versão do modelo
};

// ---------------------------------------------------------------------------
// Camadas Customizadas (ajustadas para nova dimensionalidade)
// ---------------------------------------------------------------------------

class MultiHeadCausalAttention extends tf.layers.Layer {
  constructor(config) {
    super(config || {});
    this.numHeads = config.numHeads || 8;  // ↑ 4 → 8 heads (mais parallelismo)
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
      const attentionProbs = tf.softmax(scores, -1);
      const context = attentionProbs.matMul(v).transpose([0, 2, 1, 3]).reshape([batch, seqLen, this.embDim]);
      const outFlat = context.reshape([-1, this.embDim]);
      return outFlat.matMul(this.outDense.read()).reshape([batch, seqLen, this.embDim]);
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
      const posEmbeds = this.posEmbedding.read().slice([0, 0], [seqLen, this.embDim]);
      return tokenEmbeds.add(posEmbeds);
    });
  }
  static get className() { return 'AddPositionalEmbedding'; }
}
tf.serialization.registerClass(AddPositionalEmbedding);

// ---------------------------------------------------------------------------
// Embedding Customizado (compatível com tfjs 4.11.0+)
// ---------------------------------------------------------------------------

class TokenEmbedding extends tf.layers.Layer {
  constructor(config) {
    super(config || {});
    this.inputDim = config.inputDim;   // vocab size
    this.outputDim = config.outputDim; // embedding dimension
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
      
      // Gather: buscar embeddings para cada token
      const embedded = tf.gather(this.embeddings.read(), flatInput);
      
      // Reshape para [batch, seqLen, outputDim]
      return embedded.reshape([batch, seqLen, this.outputDim]);
    });
  }

  static get className() { return 'TokenEmbedding'; }
}
tf.serialization.registerClass(TokenEmbedding);

// ---------------------------------------------------------------------------

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
    this.vocab = vocab;
    this.hparams = { ...HPARAMS, ...hparams };
    this.model = null;
    this.optimizer = null;
    this.trainSteps = 0;
    this.ready = false;
    this.embeddingVocabSize = 0;
  }

  build() {
    const { embDim, maxSeqLen, hiddenSize } = this.hparams;
    const vocabSize = this.vocab.size;
    const numHeads = 8;  // ↑ aumentado de 4
    const headDim = Math.floor(embDim / numHeads);

    const input = tf.input({ shape: [maxSeqLen], dtype: 'int32', name: 'context' });
    
    // Use TokenEmbedding customizado (compatível com tfjs 4.11.0+)
    const embedding = new TokenEmbedding({
      inputDim: vocabSize,
      outputDim: embDim,
      inputLength: maxSeqLen,
      name: 'token_embedding'
    })(input);

    const posEmbedded = new AddPositionalEmbedding({
      maxSeqLen,
      embDim,
      name: 'pos_embedding'
    })(embedding);

    // Dense feed-forward expansion
    const ffExpanded = tf.layers.dense({
      units: hiddenSize,
      activation: 'relu',
      name: 'ff_expand'
    })(posEmbedded);

    // Attention + projection
    const attended = new MultiHeadCausalAttention({
      numHeads,
      headDim,
      name: 'attention'
    })(ffExpanded);

    // Layer norm + dense para vocab
    const normalized = tf.layers.layerNormalization({ epsilon: 1e-5, name: 'layer_norm' })(attended);
    const projected = tf.layers.dense({
      units: vocabSize,
      activation: null,
      name: 'vocab_projection'
    })(normalized);

    const lastToken = new ExtractLastToken({ name: 'extract_last' })(projected);
    const output = tf.layers.dense({
      units: vocabSize,
      activation: 'softmax',
      name: 'output'
    })(lastToken);

    this.model = tf.model({ inputs: input, outputs: output });
    this.optimizer = tf.train.adam(this.hparams.learningRate);
    this.model.compile({
      optimizer: this.optimizer,
      loss: 'sparseCategoricalCrossentropy',
      metrics: ['accuracy'],
    });

    console.log(`[AKIE] Modelo construído: embDim=${embDim}, hiddenSize=${hiddenSize}, maxSeqLen=${maxSeqLen}`);
  }

  async trainBatch(pairs, epochs = 3) {
    if (!this.ready || !pairs.length) return { loss: null, accuracy: null };

    const xs = tf.data.generator(function* () {
      for (const pair of pairs) {
        yield [tf.tensor1d(pair.x, 'int32'), tf.tensor1d([pair.y], 'int32')];
      }
    })
      .batch(this.hparams.batchSize)
      .repeat(epochs);

    const history = await this.model.fitDataset(xs, {
      epochs,
      verbose: 0,
    });

    const loss = history.history.loss[history.history.loss.length - 1];
    const accuracy = history.history.acc ? history.history.acc[history.history.acc.length - 1] : null;

    this.trainSteps += pairs.length * epochs;
    return { loss, accuracy };
  }

  async generate(prompt, maxTokens = null, temperature = null) {
    if (!this.ready) return '';

    maxTokens = maxTokens || this.hparams.maxGenTokens;
    temperature = temperature || this.hparams.temperature;

    const { tokenize, PAD, BOS, EOS } = this.vocab;
    const ids = this.vocab.tokenize(prompt);

    const beams = [{ ids: ids.slice(-this.hparams.maxSeqLen), logProb: 0 }];

    for (let step = 0; step < maxTokens; step++) {
      const nextBeams = [];

      for (const beam of beams) {
        const seqTensor = tf.tensor2d([
          [
            ...Array(Math.max(0, this.hparams.maxSeqLen - beam.ids.length)).fill(SPECIAL.PAD),
            ...beam.ids
          ].slice(-this.hparams.maxSeqLen)
        ], [1, this.hparams.maxSeqLen], 'int32');

        const logits = this.model.predict(seqTensor);
        const logitsData = logits.dataSync();
        seqTensor.dispose();
        logits.dispose();

        const probs = Array.from(logitsData).map((l, i) => ({
          id: i,
          logprob: Math.log(Math.max(1e-10, l / temperature))
        }));

        probs.sort((a, b) => b.logprob - a.logprob);

        for (let k = 0; k < this.hparams.beamSize && k < probs.length; k++) {
          const nextId = probs[k].id;
          const newIds = [...beam.ids, nextId];
          const newLogProb = beam.logProb + probs[k].logprob;

          nextBeams.push({ ids: newIds, logProb: newLogProb, generatedIds: newIds });

          if (nextId === SPECIAL.EOS) break;
        }
      }

      nextBeams.sort((a, b) => b.logProb - a.logProb);
      beams.splice(0, beams.length, ...nextBeams.slice(0, this.hparams.beamSize));
    }

    const bestBeam = beams.reduce((best, beam) => beam.logProb > best.logProb ? beam : best, beams[0]);
    const outputTokens = bestBeam.ids.filter(id => id !== SPECIAL.EOS && id !== SPECIAL.PAD && id !== SPECIAL.BOS);
    if (outputTokens.length === 0) return '';
    return this.vocab.detokenize(outputTokens);
  }

  async save(dir) {
    if (!this.ready) return;

    const tmpDir = `${dir}_tmp`;
    const bakDir = `${dir}_bak`;

    await fs.promises.rm(tmpDir, { recursive: true, force: true });
    await fs.promises.mkdir(tmpDir, { recursive: true });

    const weights = this.model.getWeights();
    const manifest = weights.map(w => ({ name: w.name, shape: w.shape }));
    const allData = weights.map(w => w.dataSync());

    const totalFloats = allData.reduce((sum, arr) => sum + arr.length, 0);
    const buffer = Buffer.allocUnsafe(totalFloats * 4);
    let offset = 0;
    for (const arr of allData) {
      for (const v of arr) { buffer.writeFloatLE(v, offset); offset += 4; }
    }

    await fs.promises.writeFile(path.join(tmpDir, 'weight_manifest.json'), JSON.stringify(manifest));
    await fs.promises.writeFile(path.join(tmpDir, 'weights.bin'), buffer);

    const meta = {
      trainSteps: this.trainSteps,
      vocabSize: this.vocab.size,
      embeddingVocabSize: this.embeddingVocabSize,
      hparams: this.hparams,
      savedAt: new Date().toISOString(),
      format: 'weights_bin_v2',  // ← versão 2.0
      modelVersion: this.hparams.modelVersion,
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

  async load(dir) {
    const bakDir = `${dir}_bak`;

    for (const tryDir of [dir, bakDir]) {
      const metaPath = path.join(tryDir, 'akie_meta.json');
      const manifestPath = path.join(tryDir, 'weight_manifest.json');
      const weightsPath = path.join(tryDir, 'weights.bin');

      if (!fs.existsSync(metaPath) || !fs.existsSync(manifestPath) || !fs.existsSync(weightsPath)) {
        const files = fs.existsSync(tryDir) ? fs.readdirSync(tryDir) : [];
        console.log(`[AKIE] Procurando em: ${path.join(tryDir, 'weights.bin')}`);
        console.log(`[AKIE] Arquivos: ${files.join(', ') || 'nenhum'}`);
        continue;
      }

      try {
        const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
        const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

        // ┌─ VALIDAÇÃO DE COMPATIBILIDADE ─┐
        const loadedVersion = meta.modelVersion || meta.hparams?.modelVersion || '1.0';
        const loadedEmbDim = meta.hparams?.embDim || 64;
        const loadedHiddenSize = meta.hparams?.hiddenSize || 128;
        const loadedMaxSeqLen = meta.hparams?.maxSeqLen || 32;

        const expectedEmbDim = this.hparams.embDim;
        const expectedHiddenSize = this.hparams.hiddenSize;
        const expectedMaxSeqLen = this.hparams.maxSeqLen;

        if (
          loadedEmbDim !== expectedEmbDim ||
          loadedHiddenSize !== expectedHiddenSize ||
          loadedMaxSeqLen !== expectedMaxSeqLen
        ) {
          console.warn('\n╔══════════════════════════════════════════════════════════════╗');
          console.warn('║  ⚠️  INCOMPATIBILIDADE DE MODELO DETECTADA                      ║');
          console.warn('╠══════════════════════════════════════════════════════════════╣');
          console.warn(`║  Carregado: v${loadedVersion} [emb=${loadedEmbDim} | hidden=${loadedHiddenSize} | seq=${loadedMaxSeqLen}]`);
          console.warn(`║  Esperado:  v${this.hparams.modelVersion} [emb=${expectedEmbDim} | hidden=${expectedHiddenSize} | seq=${expectedMaxSeqLen}]`);
          console.warn('║  → Reinicializando modelo com nova arquitetura...               ║');
          console.warn('╚══════════════════════════════════════════════════════════════╝\n');
          
          // Forçar reset
          this.build();
          this.ready = true;
          return false;
        }

        this.trainSteps = meta.trainSteps || 0;
        this.embeddingVocabSize = meta.embeddingVocabSize || meta.vocabSize || this.vocab.size;

        const buildVocab = { ...this.vocab, size: this.embeddingVocabSize };
        const shell = new AKIEModel(buildVocab, this.hparams);
        shell.build();

        const buffer = fs.readFileSync(weightsPath);
        let offset = 0;
        const tensors = manifest.map(w => {
          const numFloats = w.shape.reduce((a, b) => a * b, 1);
          const arr = new Float32Array(numFloats);
          for (let i = 0; i < numFloats; i++) {
            arr[i] = buffer.readFloatLE(offset); offset += 4;
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
          await fs.promises.rm(dir, { recursive: true, force: true }).catch(() => {});
          await _copyDir(bakDir, dir);
        } else {
          console.log(`[AKIE] ✓ Transformer v2.0 carregado. (steps: ${this.trainSteps}, embVocab: ${this.embeddingVocabSize})`);
        }

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

    console.log('[AKIE] ✗ Nenhum arquivo de modelo encontrado. Inicializando novo...');
    this.build();
    this.ready = true;
    return false;
  }

  async expandVocabulary(newVocabSize) {
    if (!this.ready) return;
    if (newVocabSize <= this.embeddingVocabSize) return;
    console.log(`[AKIE] Reconfigurando vocabulário: ${this.embeddingVocabSize} → ${newVocabSize}`);

    const oldWeights = this.model.getWeights();
    this.embeddingVocabSize = newVocabSize;
    this.build();

    const newWeights = this.model.getWeights();
    const updatedWeights = newWeights.map((w, i) => {
      const oldW = oldWeights[i];
      if (!oldW) { console.warn(`[AKIE] Peso ${i} não encontrado.`); return w; }
      const oldShape = oldW.shape, newShape = w.shape;
      if (oldShape.length === newShape.length && oldShape.every((d, idx) => d === newShape[idx])) return oldW;
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
      console.warn(`[AKIE] Peso ${i}: forma incompatível ${oldShape} → ${newShape}`);
      return w;
    });
    this.model.setWeights(updatedWeights);
    oldWeights.forEach(w => { try { w.dispose(); } catch {} });
    console.log('[AKIE] ✓ Expansão concluída.');
  }

  getStats() {
    return {
      architecture: 'Causal Transformer Decoder v2.0',
      ready: this.ready,
      vocabSize: this.vocab.size,
      embeddingVocabSize: this.embeddingVocabSize,
      trainSteps: this.trainSteps,
      parameters: this.ready ? this.model.countParams() : 0,
      hparams: this.hparams,
      nlp: { natural: NaturalNLP.isReady },
    };
  }

  static stemTokens(tokens) { return tokens.map(t => NaturalNLP.stem(t)); }
}

module.exports = { AKIEModel, HPARAMS, NaturalNLP };
