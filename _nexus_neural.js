/**
 * _nexus_neural.js — AKIE Neural Core (Transformer Decoder)
 */

const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const { SPECIAL } = require('./_akie_vocab');

let NaturalNLP = { isReady: false, stem: t => t, addDocument: () => {}, generationScore: () => 0 };
try {
  const natural = require('natural');
  const stemmer = natural.PorterStemmerPt;
  NaturalNLP = {
    isReady: true,
    stem: t => { try { return stemmer.stem(t); } catch { return t; } },
    addDocument: (text, key) => { try { new natural.TfIdf().addDocument(text, key); } catch {} },
    generationScore: (p, g) => {
      if (!p || !g) return 0;
      const ps = stemmer.tokenizeAndStem(p, false);
      const gs = stemmer.tokenizeAndStem(g, false);
      return ps.length && gs.length ? ps.filter(s => gs.includes(s)).length / ps.length : 0;
    }
  };
  console.log('[NLP] natural carregado.');
} catch { console.log('[NLP] natural ausente.'); }
module.exports.NaturalNLP = NaturalNLP;

const HPARAMS = { embDim: 64, maxSeqLen: 32, batchSize: 16, learningRate: 0.001, temperature: 0.8, maxGenTokens: 40, beamSize: 3, repetitionPenalty: 1.2 };

class MultiHeadCausalAttention extends tf.layers.Layer {
  constructor(config) { super(config || {}); this.numHeads = config.numHeads || 4; this.headDim = config.headDim || 16; this.embDim = this.numHeads * this.headDim; }
  build(inputShape) {
    this.qDense = this.addWeight('q', [this.embDim, this.embDim], 'float32', tf.initializers.glorotUniform());
    this.kDense = this.addWeight('k', [this.embDim, this.embDim], 'float32', tf.initializers.glorotUniform());
    this.vDense = this.addWeight('v', [this.embDim, this.embDim], 'float32', tf.initializers.glorotUniform());
    this.outDense = this.addWeight('o', [this.embDim, this.embDim], 'float32', tf.initializers.glorotUniform());
    this.built = true;
  }
  computeOutputShape(inputShape) { return inputShape; }
  call(inputs) {
    return tf.tidy(() => {
      const x = inputs[0] || inputs; const [batch, seqLen] = x.shape; const xFlat = x.reshape([-1, this.embDim]);
      const q = xFlat.matMul(this.qDense.read()).reshape([batch, seqLen, this.numHeads, this.headDim]).transpose([0, 2, 1, 3]);
      const k = xFlat.matMul(this.kDense.read()).reshape([batch, seqLen, this.numHeads, this.headDim]).transpose([0, 2, 1, 3]);
      const v = xFlat.matMul(this.vDense.read()).reshape([batch, seqLen, this.numHeads, this.headDim]).transpose([0, 2, 1, 3]);
      let scores = q.matMul(k.transpose([0, 1, 3, 2])).div(tf.scalar(Math.sqrt(this.headDim)));
      const mask = tf.tidy(() => { const o = tf.ones([seqLen, seqLen]); const t = tf.linalg.bandPart(o, -1, 0); return tf.where(t.equal(1), tf.zeros([seqLen, seqLen]), tf.scalar(-1e9)); });
      scores = scores.add(mask); const attn = tf.softmax(scores, -1);
      const ctx = attn.matMul(v).transpose([0, 2, 1, 3]).reshape([batch, seqLen, this.embDim]);
      return ctx.reshape([-1, this.embDim]).matMul(this.outDense.read()).reshape([batch, seqLen, this.embDim]);
    });
  }
  static get className() { return 'MultiHeadCausalAttention'; }
}
tf.serialization.registerClass(MultiHeadCausalAttention);

class AddPositionalEmbedding extends tf.layers.Layer {
  constructor(config) { super(config || {}); this.maxSeqLen = config.maxSeqLen; this.embDim = config.embDim; }
  build(inputShape) { this.posEmb = this.addWeight('pos', [this.maxSeqLen, this.embDim], 'float32', tf.initializers.glorotUniform()); this.built = true; }
  computeOutputShape(inputShape) { return inputShape; }
  call(inputs) { return tf.tidy(() => { const t = inputs[0] || inputs; return t.add(this.posEmb.read().slice([0, 0], [t.shape[1], this.embDim])); }); }
  static get className() { return 'AddPositionalEmbedding'; }
}
tf.serialization.registerClass(AddPositionalEmbedding);

class ExtractLastToken extends tf.layers.Layer {
  constructor(config) { super(config || {}); }
  computeOutputShape(inputShape) { return [inputShape[0], inputShape[2]]; }
  call(inputs) { return tf.tidy(() => { const x = inputs[0] || inputs; return x.slice([0, x.shape[1] - 1, 0], [-1, 1, -1]).squeeze([1]); }); }
  static get className() { return 'ExtractLastToken'; }
}
tf.serialization.registerClass(ExtractLastToken);

class AKIEModel {
  constructor(vocab, hparams = HPARAMS) { this.vocab = vocab; this.hparams = { ...HPARAMS, ...hparams }; this.model = null; this.optimizer = null; this.trainSteps = 0; this.ready = false; this.embeddingVocabSize = 0; }

  build() {
    const { embDim, maxSeqLen } = this.hparams; const vs = this.vocab.size; const nh = 4; const hd = Math.floor(embDim / nh);
    const inp = tf.input({ shape: [maxSeqLen], dtype: 'int32', name: 'context' });
    const te = tf.layers.embedding({ inputDim: vs, outputDim: embDim, name: 'embedding' }).apply(inp);
    const x = new AddPositionalEmbedding({ maxSeqLen, embDim, name: 'pos_embed' }).apply(te);
    const n11 = tf.layers.layerNormalization({ axis: -1, name: 'n11' }).apply(x);
    const a1 = new MultiHeadCausalAttention({ numHeads: nh, headDim: hd, name: 'a1' }).apply(n11);
    const r1 = tf.layers.add({ name: 'r1' }).apply([x, a1]);
    const n12 = tf.layers.layerNormalization({ axis: -1, name: 'n12' }).apply(r1);
    const f11 = tf.layers.dense({ units: embDim * 4, activation: 'relu', name: 'f11' }).apply(n12);
    const f12 = tf.layers.dense({ units: embDim, name: 'f12' }).apply(f11);
    const b1 = tf.layers.add({ name: 'b1' }).apply([r1, f12]);
    const n21 = tf.layers.layerNormalization({ axis: -1, name: 'n21' }).apply(b1);
    const a2 = new MultiHeadCausalAttention({ numHeads: nh, headDim: hd, name: 'a2' }).apply(n21);
    const r2 = tf.layers.add({ name: 'r2' }).apply([b1, a2]);
    const n22 = tf.layers.layerNormalization({ axis: -1, name: 'n22' }).apply(r2);
    const f21 = tf.layers.dense({ units: embDim * 4, activation: 'relu', name: 'f21' }).apply(n22);
    const f22 = tf.layers.dense({ units: embDim, name: 'f22' }).apply(f21);
    const b2 = tf.layers.add({ name: 'b2' }).apply([r2, f22]);
    const fn = tf.layers.layerNormalization({ axis: -1, name: 'fn' }).apply(b2);
    const lt = new ExtractLastToken({ name: 'lt' }).apply(fn);
    const out = tf.layers.dense({ units: vs, activation: 'softmax', name: 'out' }).apply(lt);
    this.model = tf.model({ inputs: inp, outputs: out, name: 'AKIE' });
    this.optimizer = tf.train.adam(this.hparams.learningRate);
    this.model.compile({ optimizer: this.optimizer, loss: 'sparseCategoricalCrossentropy', metrics: ['accuracy'] });
    this.ready = true; this.embeddingVocabSize = vs;
    console.log(`[AKIE] Transformer construído. Params: ${this.model.countParams().toLocaleString()}, Vocab: ${vs}`);
    return this;
  }

  async trainBatch(pairs, epochs = 3) {
    if (!this.ready) throw new Error('Modelo não construído.');
    if (!pairs?.length) return { loss: null, accuracy: null, steps: 0 };
    const vp = pairs.filter(p => Array.isArray(p.x) && p.x.every(id => id < this.embeddingVocabSize) && p.y < this.embeddingVocabSize);
    if (!vp.length) { console.warn('[AKIE] Nenhum par válido.'); return { loss: null, accuracy: null, steps: 0 }; }
    if (NaturalNLP.isReady && vp.length > 10) NaturalNLP.addDocument(vp.slice(0, 50).map(p => p.x.filter(id => id > 3).join(' ')).join(' '), `b${this.trainSteps}`);
    const xs = tf.tensor2d(vp.map(p => p.x), [vp.length, this.hparams.maxSeqLen], 'int32');
    const ys = tf.tensor1d(vp.map(p => p.y), 'float32');
    let ll = null, la = null;
    try {
      const h = await this.model.fit(xs, ys, { epochs, batchSize: this.hparams.batchSize, shuffle: true, verbose: 0 });
      ll = h.history.loss[h.history.loss.length - 1];
      la = (h.history.acc || h.history.accuracy)[h.history.loss.length - 1];
      this.trainSteps += vp.length * epochs;
    } finally { xs.dispose(); ys.dispose(); }
    return { loss: ll, accuracy: la, steps: this.trainSteps, samples: vp.length };
  }

  generate(prompt = '', maxTokens = null, options = {}) {
    if (!this.ready) return null;
    const mt = maxTokens || this.hparams.maxGenTokens, bs = options.beamSize || this.hparams.beamSize;
    const rp = options.repetitionPenalty || this.hparams.repetitionPenalty, tmp = options.temperature ?? this.hparams.temperature;
    const cids = this.vocab.tokenize(prompt).filter(id => id !== SPECIAL.EOS);
    let beams = [{ score: 0, ids: [...cids], gids: [] }]; const done = []; const pl = this.hparams.maxSeqLen;
    for (let s = 0; s < mt; s++) {
      if (!beams.length) break;
      const pb = beams.map(b => [...Array(Math.max(0, pl - b.ids.length)).fill(SPECIAL.PAD), ...b.ids.slice(-pl)]);
      const it = tf.tensor2d(pb, [beams.length, pl], 'int32');
      const rl = this.model.predict(it); const lt = rl.dtype === 'float32' ? rl : rl.cast('float32'); const la = lt.arraySync();
      it.dispose(); if (lt !== rl) rl.dispose(); lt.dispose();
      const cands = [];
      for (let b = 0; b < beams.length; b++) {
        let logs = la[b]; const bm = beams[b];
        if (rp !== 1.0) { const seen = new Set(bm.ids); logs = logs.map((l, i) => seen.has(i) ? (l > 0 ? l / rp : l * rp) : l); }
        if (tmp > 0 && tmp !== 1.0) logs = logs.map(l => l / tmp);
        const mx = Math.max(...logs), exps = logs.map(l => Math.exp(l - mx)), sum = exps.reduce((a, b) => a + b, 0);
        const ls = logs.map(l => (l - mx) - Math.log(sum));
        ls.map((lp, idx) => ({ lp, idx })).sort((a, b) => b.lp - a.lp).slice(0, bs).forEach(c => {
          cands.push({ score: bm.score + c.lp, ids: [...bm.ids, c.idx], gids: [...bm.gids, c.idx] });
        });
      }
      cands.sort((a, b) => b.score - a.score); beams = [];
      for (const c of cands) { if (beams.length >= bs) break; c.gids[c.gids.length - 1] === SPECIAL.EOS ? done.push(c) : beams.push(c); }
    }
    const fs = [...done, ...beams]; if (!fs.length || !fs[0]) return '';
    if (NaturalNLP.isReady && fs.length > 1) {
      fs.forEach(c => { const g = this.vocab.detokenize(c.gids.filter(id => ![SPECIAL.EOS, SPECIAL.PAD, SPECIAL.BOS].includes(id))); c.cs = c.score * 0.85 + NaturalNLP.generationScore(prompt, g) * 0.15; });
      fs.sort((a, b) => b.cs - a.cs);
    } else fs.sort((a, b) => b.score - a.score);
    const best = fs[0]; if (!best.gids?.length) return '';
    const toks = best.gids.filter(id => ![SPECIAL.EOS, SPECIAL.PAD, SPECIAL.BOS].includes(id));
    return toks.length ? this.vocab.detokenize(toks) : '';
  }

  async save(dir) {
    if (!this.ready) return; await fs.promises.mkdir(dir, { recursive: true }); await this.model.save(`file://${dir}`);
    const meta = { trainSteps: this.trainSteps, vocabSize: this.vocab.size, embeddingVocabSize: this.embeddingVocabSize, hparams: this.hparams, savedAt: new Date().toISOString() };
    await fs.promises.writeFile(path.join(dir, 'akie_meta.json'), JSON.stringify(meta, null, 2));
    await fs.promises.writeFile(path.join(dir, 'akie_vocab.json'), JSON.stringify(this.vocab.toJSON(), null, 2));
    console.log(`[AKIE] Salvo em ${dir} (steps: ${this.trainSteps})`);
  }

  async load(dir) {
    try {
      const mp = path.join(dir, 'model.json'); console.log(`[AKIE] Procurando em: ${mp}`);
      if (fs.existsSync(dir)) console.log(`[AKIE] Arquivos: ${fs.readdirSync(dir).join(', ') || 'nenhum'}`);
      if (!fs.existsSync(mp)) { console.log('[AKIE] Não encontrado.'); return false; }
      const metaP = path.join(dir, 'akie_meta.json'); let svs = this.vocab.size;
      if (fs.existsSync(metaP)) { const meta = JSON.parse(await fs.promises.readFile(metaP, 'utf8')); this.trainSteps = meta.trainSteps || 0; svs = meta.vocabSize || this.vocab.size; }
      if (this.vocab.size > svs) {
        console.log('[AKIE] Vocab cresceu. Expandindo...'); this.model = await tf.loadLayersModel(`file://${dir}/model.json`);
        this.optimizer = tf.train.adam(this.hparams.learningRate); this.model.compile({ optimizer: this.optimizer, loss: 'sparseCategoricalCrossentropy', metrics: ['accuracy'] });
        try { this.embeddingVocabSize = this.model.getLayer('embedding').getWeights()[0].shape[0]; } catch { this.embeddingVocabSize = svs; }
        this.ready = true; if (this.vocab.size > this.embeddingVocabSize) await this.expandVocabulary(this.vocab.size);
        return true;
      }
      this.model = await tf.loadLayersModel(`file://${dir}/model.json`); this.optimizer = tf.train.adam(this.hparams.learningRate);
      this.model.compile({ optimizer: this.optimizer, loss: 'sparseCategoricalCrossentropy', metrics: ['accuracy'] });
      try { this.embeddingVocabSize = this.model.getLayer('embedding').getWeights()[0].shape[0]; } catch { this.embeddingVocabSize = this.vocab.size; }
      this.ready = true; console.log(`[AKIE] Carregado. (steps: ${this.trainSteps})`); return true;
    } catch (err) {
      console.error(`[AKIE] Erro ao carregar: ${err.message}`); console.log('[AKIE] Limpando diretório...');
      try { for (const f of fs.readdirSync(dir)) fs.unlinkSync(path.join(dir, f)); console.log('[AKIE] Limpo.'); } catch (e) { console.error('[AKIE] Erro ao limpar:', e.message); }
      return false;
    }
  }

  async expandVocabulary(nvs) {
    if (!this.ready || nvs <= this.embeddingVocabSize) return;
    console.log(`[AKIE] Expandindo: ${this.embeddingVocabSize} → ${nvs}`);
    const ow = this.model.getWeights(); this.embeddingVocabSize = nvs; this.build(); const nw = this.model.getWeights();
    const uw = nw.map((w, i) => {
      const o = ow[i]; if (!o) return w;
      const os = o.shape, ns = w.shape;
      if (os.length === ns.length && os.every((d, j) => d === ns[j])) return o;
      if (os.length === 2 && ns.length === 2 && ns[0] > os[0] && os[1] === ns[1]) { const m = new Float32Array(w.dataSync()); m.set(o.dataSync()); return tf.tensor(m, ns, 'float32'); }
      if (os.length === 2 && ns.length === 2 && ns[1] > os[1] && os[0] === ns[0]) { const od = o.dataSync(), m = new Float32Array(w.dataSync()); const [r, oc] = os, nc = ns[1]; for (let r = 0; r < r; r++) for (let c = 0; c < oc; c++) m[r * nc + c] = od[r * oc + c]; return tf.tensor(m, ns, 'float32'); }
      if (os.length === 1 && ns.length === 1 && ns[0] > os[0]) { const m = new Float32Array(w.dataSync()); m.set(o.dataSync()); return tf.tensor(m, ns, 'float32'); }
      return w;
    });
    this.model.setWeights(uw); ow.forEach(w => { try { w.dispose(); } catch {} }); console.log('[AKIE] Expansão concluída.');
  }

  getStats() {
    return { architecture: 'Transformer Decoder', ready: this.ready, vocabSize: this.vocab.size, embeddingVocabSize: this.embeddingVocabSize, trainSteps: this.trainSteps, parameters: this.ready ? this.model.countParams() : 0, nlp: { natural: NaturalNLP.isReady } };
  }
  static stemTokens(tokens) { return tokens.map(t => NaturalNLP.stem(t)); }
}

module.exports = { AKIEModel, HPARAMS, NaturalNLP };
