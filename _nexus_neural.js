/**
 * _nexus_neural.js  —  AKIE Neural Core (Versão Transformer Decoder)
 *
 * Rede neural real baseada na arquitetura Causal Transformer Decoder.
 * Tecnologia: TensorFlow.js (tfjs-node) — treina localmente em CPU/GPU.
 *
 * Arquitetura:
 * TokenEmbedding + PositionalEmbedding → 2x Transformer Decoder Blocks (Multi-Head Causal Attention) 
 * → LayerNorm → ExtractLastToken → Dense(vocabSize) → Softmax
 *
 * Paradigma: Autoregressive Language Modeling (Next-token prediction).
 * Geração Avançada: Beam Search em Lote com Penalização de Repetição e Temperatura.
 *
 * NLP Layer: biblioteca `natural` para stemming PT-BR, TF-IDF e similaridade
 * — enriquece pares de treino e melhora qualidade do scoring de geração.
 *
 * CORREÇÕES v4:
 *   - load(): detecta vocab crescido e expande automaticamente
 *   - load(): diagnóstico de persistência do volume
 *   - expandVocabulary(): proteção contra undefined weights + fallback seguro
 *   - generate(): proteção contra beam vazio e tokens inválidos
 */

const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const { Vocabulary, makeTrainingPairs, SPECIAL } = require('./_akie_vocab');

// ---------------------------------------------------------------------------
// NLP Layer — biblioteca `natural` (gratuita, sem API externa)
// Funções: stemming PT-BR, TF-IDF, similaridade de strings
// ---------------------------------------------------------------------------

let NaturalNLP = null;

try {
  const natural = require('natural');

  // Stemmer para PT-BR (PorterStemmerPt remove sufixos morfológicos)
  const stemmer = natural.PorterStemmerPt;

  // TF-IDF para calcular relevância de termos nas frases de treino
  const tfidf = new natural.TfIdf();

  // Distância de Jaro-Winkler para similaridade entre tokens gerados e esperados
  const JaroWinklerDistance = natural.JaroWinklerDistance;

  NaturalNLP = {
    /**
     * Reduz token ao seu radical (stem).
     * Ex: "aprendendo" → "aprend", "correndo" → "corr"
     * Aumenta generalização: tokens morfologicamente relacionados
     * convergem para o mesmo stem no grafo semântico.
     */
    stem(token) {
      try { return stemmer.stem(token); }
      catch { return token; }
    },

    /**
     * Calcula stems de uma frase inteira.
     * Retorna array de stems para enriquecer o vocabulário com formas radicais.
     */
    stemSentence(sentence) {
      try {
        return stemmer.tokenizeAndStem(sentence, false); // false = não remover stopwords
      } catch {
        return sentence.toLowerCase().split(/\s+/).filter(Boolean);
      }
    },

    /**
     * Adiciona documento ao índice TF-IDF.
     * Permite identificar quais termos são mais relevantes em cada documento.
     */
    addDocument(text, key) {
      try { tfidf.addDocument(text, key); }
      catch { /* ignora */ }
    },

    /**
     * Retorna os N termos mais relevantes de um documento indexado.
     * Usado para selecionar os melhores tokens para expandir o vocab.
     */
    topTerms(docIndex, n = 5) {
      try {
        const terms = [];
        tfidf.listTerms(docIndex).slice(0, n).forEach(t => terms.push(t.term));
        return terms;
      } catch { return []; }
    },

    /**
     * Similaridade Jaro-Winkler entre dois strings (0.0 a 1.0).
     * Usada no scoring de geração: penaliza outputs muito distantes
     * semanticamente da query de entrada.
     */
    similarity(a, b) {
      try { return JaroWinklerDistance(a, b); }
      catch { return 0; }
    },

    /**
     * Score de qualidade de uma geração usando similaridade com o prompt.
     * Complementa o scoreGeneration existente no _akie_dataset.js.
     */
    generationScore(prompt, generated) {
      if (!prompt || !generated) return 0;
      const promptStems  = stemmer.tokenizeAndStem(prompt, false);
      const genStems     = stemmer.tokenizeAndStem(generated, false);
      if (!promptStems.length || !genStems.length) return 0;
      // Proporção de stems do prompt que aparecem na geração
      const overlap = promptStems.filter(s => genStems.includes(s)).length;
      return overlap / promptStems.length;
    },

    isReady: true,
  };

  console.log('[NLP] Biblioteca `natural` carregada. Stemming PT-BR, TF-IDF e similaridade ativos.');
} catch (e) {
  // natural não instalada — sistema continua sem degradação
  NaturalNLP = { 
    isReady: false, 
    stem: t => t, 
    stemSentence: s => s.toLowerCase().split(/\s+/).filter(Boolean), 
    similarity: () => 0, 
    generationScore: () => 0, 
    addDocument: () => {}, 
    topTerms: () => [] 
  };
  console.log('[NLP] `natural` não encontrada. Rodando sem NLP avançado. Execute: npm install natural');
}

module.exports.NaturalNLP = NaturalNLP;

// ---------------------------------------------------------------------------
// Hiperparâmetros
// ---------------------------------------------------------------------------

const HPARAMS = {
  embDim:            64,    // dimensão dos embeddings (multiplo de numHeads)
  hiddenSize:       128,    // mantido para compatibilidade de logs
  maxSeqLen:         32,    // comprimento máximo de contexto
  batchSize:         16,    // tamanho de batch para treino
  learningRate:   0.001,
  temperature:       0.8,
  maxGenTokens:      40,
  minNewData:         3,
  beamSize:           3,
  repetitionPenalty: 1.2,
};

// ---------------------------------------------------------------------------
// Camadas Customizadas do Transformer (Extensão Core do TF.js)
// ---------------------------------------------------------------------------

/**
 * Camada de Auto-Atenção Multi-Head com Máscara Causal (Look-Ahead Mask).
 * Impede que o modelo olhe para os tokens futuros durante o processamento.
 */
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

  computeOutputShape(inputShape) {
    return inputShape;
  }

  call(inputs) {
    return tf.tidy(() => {
      const x = inputs[0] || inputs; 
      const [batch, seqLen, encodingDim] = x.shape;

      // Projeções lineares achatadas para otimizar multiplicação de matrizes
      const xFlat = x.reshape([-1, this.embDim]);
      
      const q = xFlat.matMul(this.qDense.read()).reshape([batch, seqLen, this.numHeads, this.headDim]).transpose([0, 2, 1, 3]);
      const k = xFlat.matMul(this.kDense.read()).reshape([batch, seqLen, this.numHeads, this.headDim]).transpose([0, 2, 1, 3]);
      const v = xFlat.matMul(this.vDense.read()).reshape([batch, seqLen, this.numHeads, this.headDim]).transpose([0, 2, 1, 3]);

      // Matmul de atenção escalada: (Q @ K^T) / sqrt(d_k)
      let scores = q.matMul(k.transpose([0, 1, 3, 2]));
      scores = scores.div(tf.scalar(Math.sqrt(this.headDim)));

      // Injeção da máscara causal triangular inferior (Causal/Look-ahead Masking)
      const mask = tf.tidy(() => {
        const ones = tf.ones([seqLen, seqLen]);
        const tril = tf.linalg.bandPart(ones, -1, 0);
        return tf.where(tril.equal(1), tf.zeros([seqLen, seqLen]), tf.scalar(-1e9));
      });
      scores = scores.add(mask);

      const attentionProbs = tf.softmax(scores, -1);
      
      // Multiplicação pelo vetor de valores: AttentionProbs @ V
      const context = attentionProbs.matMul(v);
      const contextReshaped = context.transpose([0, 2, 1, 3]).reshape([batch, seqLen, this.embDim]);

      // Projeção final de saída
      const outFlat = contextReshaped.reshape([-1, this.embDim]);
      return outFlat.matMul(this.outDense.read()).reshape([batch, seqLen, this.embDim]);
    });
  }

  static get className() { return 'MultiHeadCausalAttention'; }
}
tf.serialization.registerClass(MultiHeadCausalAttention);

/**
 * Camada de Injeção de Embeddings Posicionais Treináveis.
 * Fornece a noção de ordem e sequência temporal exigida pela arquitetura do Transformer.
 */
class AddPositionalEmbedding extends tf.layers.Layer {
  constructor(config) {
    super(config || {});
    this.maxSeqLen = config.maxSeqLen;
    this.embDim = config.embDim;
  }

  build(inputShape) {
    this.posEmbedding = this.addWeight('pos_embedding', [this.maxSeqLen, this.embDim], 'float32', tf.initializers.glorotUniform());
    this.built = true;
  }

  computeOutputShape(inputShape) {
    return inputShape;
  }

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
 * Camada de Slicing Autoregressivo.
 * Captura estritamente o vetor correspondente ao último token gerado do contexto.
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
// Classe Principal: AKIEModel
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
    this._expandLock = null;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Construção do Modelo (Transformer Decoder)
  // ─────────────────────────────────────────────────────────────────────────

  build() {
    const { embDim, maxSeqLen } = this.hparams;
    const vocabSize = this.vocab.size;
    const numHeads = 4;
    const headDim = Math.floor(embDim / numHeads); // 64 / 4 = 16

    const input = tf.input({ shape: [maxSeqLen], dtype: 'int32', name: 'context' });

    // 1. Camada de Vetorização de Palavras (Tokens)
    const tokenEmbeddings = tf.layers.embedding({
      inputDim: vocabSize,
      outputDim: embDim,
      maskZero: false,
      name: 'embedding',
    }).apply(input);

    // 2. Acoplamento de Vetores Posicionais
    const x = new AddPositionalEmbedding({ maxSeqLen, embDim, name: 'pos_embed' }).apply(tokenEmbeddings);

    // ── Bloco Transformer Decoder 1 ──
    const norm1_1 = tf.layers.layerNormalization({ axis: -1, name: 'norm1_1' }).apply(x);
    const attn1 = new MultiHeadCausalAttention({ numHeads, headDim, name: 'attn_1' }).apply(norm1_1);
    const res1 = tf.layers.add({ name: 'add1_1' }).apply([x, attn1]);

    const norm1_2 = tf.layers.layerNormalization({ axis: -1, name: 'norm1_2' }).apply(res1);
    const ffn1_1 = tf.layers.dense({ units: embDim * 4, activation: 'relu', name: 'ffn1_1' }).apply(norm1_2);
    const ffn1_2 = tf.layers.dense({ units: embDim, name: 'ffn1_2' }).apply(ffn1_1);
    const block1Output = tf.layers.add({ name: 'add1_2' }).apply([res1, ffn1_2]);

    // ── Bloco Transformer Decoder 2 ──
    const norm2_1 = tf.layers.layerNormalization({ axis: -1, name: 'norm2_1' }).apply(block1Output);
    const attn2 = new MultiHeadCausalAttention({ numHeads, headDim, name: 'attn_2' }).apply(norm2_1);
    const res2 = tf.layers.add({ name: 'add2_1' }).apply([block1Output, attn2]);

    const norm2_2 = tf.layers.layerNormalization({ axis: -1, name: 'norm2_2' }).apply(res2);
    const ffn2_1 = tf.layers.dense({ units: embDim * 4, activation: 'relu', name: 'ffn2_1' }).apply(norm2_2);
    const ffn2_2 = tf.layers.dense({ units: embDim, name: 'ffn2_2' }).apply(ffn2_1);
    const block2Output = tf.layers.add({ name: 'add2_2' }).apply([res2, ffn2_2]);

    // 3. Normalização Final e Extração Autoregressiva
    const finalNorm = tf.layers.layerNormalization({ axis: -1, name: 'final_norm' }).apply(block2Output);
    const lastTokenTensor = new ExtractLastToken({ name: 'extract_last' }).apply(finalNorm);

    // 4. Projeção Linear de Probabilidade sobre o Vocabulário
    const output = tf.layers.dense({
      units: vocabSize,
      activation: 'softmax',
      name: 'output',
    }).apply(lastTokenTensor);

    this.model = tf.model({ inputs: input, outputs: output, name: 'AKIE_Transformer' });
    this.optimizer = tf.train.adam(this.hparams.learningRate);

    this.model.compile({
      optimizer: this.optimizer,
      loss: 'sparseCategoricalCrossentropy',
      metrics: ['accuracy'],
    });

    this.ready = true;
    this.embeddingVocabSize = vocabSize;
    console.log(`[AKIE] Transformer Decoder construído. Parâmetros: ${this.model.countParams().toLocaleString()}`);
    console.log(`[AKIE] Vocabulário ativo: ${vocabSize} tokens`);
    return this;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Treinamento Incremental
  // ─────────────────────────────────────────────────────────────────────────

  async trainBatch(pairs, epochs = 3) {
    if (!this.ready) throw new Error('Modelo não construído. Chame build() primeiro.');
    if (!pairs || pairs.length === 0) return { loss: null, accuracy: null, steps: 0 };

    // ── Filtro de segurança: descarta tokens que não cabem na embedding atual ──
    const validPairs = pairs.filter(p => {
      if (!Array.isArray(p.x)) return false;
      return p.x.every(id => id < this.embeddingVocabSize) && p.y < this.embeddingVocabSize;
    });

    if (validPairs.length === 0) {
      console.warn('[AKIE] Nenhum par válido (tokens fora do vocab do modelo). Pulando treino.');
      return { loss: null, accuracy: null, steps: 0 };
    }

    if (validPairs.length < pairs.length) {
      console.warn(`[AKIE] ${pairs.length - validPairs.length} pares descartados (tokens >= embeddingVocabSize).`);
    }

    // Enriquecimento NLP: se `natural` está disponível, adiciona os pares
    // ao índice TF-IDF para análise posterior de relevância de termos.
    if (NaturalNLP.isReady && validPairs.length > 10) {
      const sample = validPairs.slice(0, Math.min(50, validPairs.length));
      const text = sample.map(p => p.x.filter(id => id > 3).join(' ')).join(' ');
      NaturalNLP.addDocument(text, `batch_${this.trainSteps}`);
    }

    const xs = tf.tensor2d(validPairs.map(p => p.x), [validPairs.length, this.hparams.maxSeqLen], 'int32');
    // FIX: sparseCategoricalCrossentropy no tfjs-node 4.x exige float32 nos labels
    const ys = tf.tensor1d(validPairs.map(p => p.y), 'float32');

    let lastLoss = null;
    let lastAcc = null;

    try {
      const history = await this.model.fit(xs, ys, {
        epochs,
        batchSize: this.hparams.batchSize,
        shuffle: true,
        verbose: 0,
      });

      lastLoss = history.history.loss[history.history.loss.length - 1];
      lastAcc  = (history.history.acc || history.history.accuracy)[history.history.loss.length - 1];
      this.trainSteps += validPairs.length * epochs;
    } finally {
      xs.dispose();
      ys.dispose();
    }

    return { loss: lastLoss, accuracy: lastAcc, steps: this.trainSteps, samples: validPairs.length };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Geração Avançada com Beam Search em Lote
  // ─────────────────────────────────────────────────────────────────────────

  generate(prompt = '', maxTokens = null, options = {}) {
    if (!this.ready) return null;

    const maxT = maxTokens || this.hparams.maxGenTokens;
    const beamSize = options.beamSize || this.hparams.beamSize;
    const repPenalty = options.repetitionPenalty || this.hparams.repetitionPenalty;
    const temp = options.temperature !== null ? (options.temperature ?? this.hparams.temperature) : this.hparams.temperature;

    const promptIds = this.vocab.tokenize(prompt);
    const contextIds = promptIds.filter(id => id !== SPECIAL.EOS);

    let beams = [{ score: 0, ids: [...contextIds], generatedIds: [] }];
    const completedBeams = [];
    const padLen = this.hparams.maxSeqLen;

    for (let step = 0; step < maxT; step++) {
      if (beams.length === 0) break;

      const paddedBatch = beams.map(beam => [
        ...Array(Math.max(0, padLen - beam.ids.length)).fill(SPECIAL.PAD),
        ...beam.ids.slice(-padLen),
      ]);

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
          const seenTokens = new Set(beam.ids);
          logits = logits.map((logit, idx) => {
            if (seenTokens.has(idx)) {
              return logit > 0 ? logit / repPenalty : logit * repPenalty;
            }
            return logit;
          });
        }

        if (temp > 0 && temp !== 1.0) {
          logits = logits.map(l => l / temp);
        }

        const maxLogit = Math.max(...logits);
        const exps = logits.map(l => Math.exp(l - maxLogit));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        const logSoftmax = logits.map(l => (l - maxLogit) - Math.log(sumExps));

        const tokenRankings = logSoftmax.map((logProb, tokenIdx) => ({ logProb, tokenIdx }));
        tokenRankings.sort((a, b) => b.logProb - a.logProb);

        for (const candidate of tokenRankings.slice(0, beamSize)) {
          candidates.push({
            score: beam.score + candidate.logProb,
            ids: [...beam.ids, candidate.tokenIdx],
            generatedIds: [...beam.generatedIds, candidate.tokenIdx]
          });
        }
      }

      candidates.sort((a, b) => b.score - a.score);
      beams = [];

      for (const cand of candidates) {
        if (beams.length >= beamSize) break;
        const lastToken = cand.generatedIds[cand.generatedIds.length - 1];
        if (lastToken === SPECIAL.EOS) {
          completedBeams.push(cand);
        } else {
          beams.push(cand);
        }
      }
    }

    const finalSelection = [...completedBeams, ...beams];
    
    // ── Proteção: se todos os beams falharam ──
    if (finalSelection.length === 0 || !finalSelection[0]) {
      return '';
    }

    // Reranking com NLP semântico: se natural está disponível, combina
    // score do beam (log-prob) com similaridade de stems prompt→geração.
    // Peso 0.85 log-prob + 0.15 semântico — preserva qualidade neural
    // mas desempata beams próximos usando coerência temática.
    if (NaturalNLP.isReady && finalSelection.length > 1) {
      finalSelection.forEach(cand => {
        const generated = this.vocab.detokenize(
          cand.generatedIds.filter(id => id !== SPECIAL.EOS && id !== SPECIAL.PAD && id !== SPECIAL.BOS)
        );
        const nlpScore = NaturalNLP.generationScore(prompt, generated);
        // Normaliza o beam score (negativo, log-prob) para combinar com nlpScore (0–1)
        cand.combinedScore = cand.score * 0.85 + nlpScore * 0.15;
      });
      finalSelection.sort((a, b) => b.combinedScore - a.combinedScore);
    } else {
      finalSelection.sort((a, b) => b.score - a.score);
    }

    // ── Garantir que há tokens válidos ──
    const bestBeam = finalSelection[0];
    if (!bestBeam.generatedIds || bestBeam.generatedIds.length === 0) {
      return '';
    }

    const outputTokens = bestBeam.generatedIds.filter(
      id => id !== SPECIAL.EOS && id !== SPECIAL.PAD && id !== SPECIAL.BOS
    );

    // Se após filtrar não sobrar nada, retorna vazio
    if (outputTokens.length === 0) {
      return '';
    }

    return this.vocab.detokenize(outputTokens);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Persistência e Expansão Segura de Matrizes
  // ─────────────────────────────────────────────────────────────────────────

  async save(dir) {
    if (!this.ready) return;
    await fs.promises.mkdir(dir, { recursive: true });
    await this.model.save(`file://${dir}`);

    const meta = { 
      trainSteps: this.trainSteps, 
      vocabSize: this.vocab.size, 
      embeddingVocabSize: this.embeddingVocabSize,
      hparams: this.hparams, 
      savedAt: new Date().toISOString() 
    };
    await fs.promises.writeFile(path.join(dir, 'akie_meta.json'), JSON.stringify(meta, null, 2));
    await fs.promises.writeFile(path.join(dir, 'akie_vocab.json'), JSON.stringify(this.vocab.toJSON(), null, 2));
    console.log(`[AKIE] Transformer salvo em ${dir} (steps: ${this.trainSteps})`);
  }

  async load(dir) {
    try {
      const modelPath = path.join(dir, 'model.json');
      
      // ── Diagnóstico de persistência ──
      console.log(`[AKIE] Procurando modelo em: ${modelPath}`);
      console.log(`[AKIE] Diretório ${dir} existe? ${fs.existsSync(dir)}`);
      
      if (fs.existsSync(dir)) {
        const files = fs.readdirSync(dir);
        console.log(`[AKIE] Arquivos no diretório: ${files.join(', ') || 'nenhum'}`);
      }
      
      if (!fs.existsSync(modelPath)) {
        console.log(`[AKIE] Nenhum modelo encontrado em ${modelPath}`);
        return false;
      }

      // Verificar se o modelo é compatível com o vocab atual
      const metaPath = path.join(dir, 'akie_meta.json');
      let savedVocabSize = this.vocab.size;
      
      if (fs.existsSync(metaPath)) {
        const meta = JSON.parse(await fs.promises.readFile(metaPath, 'utf8'));
        this.trainSteps = meta.trainSteps || 0;
        savedVocabSize = meta.vocabSize || this.vocab.size;
      }

      // Se o vocab atual é maior que o salvo, precisa rebuild com expansão
      if (this.vocab.size > savedVocabSize) {
        console.log(`[AKIE] Vocab cresceu desde último save (${savedVocabSize} → ${this.vocab.size}).`);
        console.log(`[AKIE] Carregando modelo base e expandindo...`);
        
        this.model = await tf.loadLayersModel(`file://${dir}/model.json`);
        this.optimizer = tf.train.adam(this.hparams.learningRate);
        this.model.compile({ 
          optimizer: this.optimizer, 
          loss: 'sparseCategoricalCrossentropy', 
          metrics: ['accuracy'] 
        });

        try {
          this.embeddingVocabSize = this.model.getLayer('embedding').getWeights()[0].shape[0];
        } catch(e) {
          this.embeddingVocabSize = savedVocabSize;
        }

        this.ready = true;
        
        // Expandir para o tamanho atual do vocab
        if (this.vocab.size > this.embeddingVocabSize) {
          await this.expandVocabulary(this.vocab.size);
        }
        
        console.log(`[AKIE] Transformer carregado e expandido. (steps: ${this.trainSteps}, embVocab: ${this.embeddingVocabSize})`);
        return true;
      }

      // Carregamento normal (vocab compatível)
      this.model = await tf.loadLayersModel(`file://${dir}/model.json`);
      this.optimizer = tf.train.adam(this.hparams.learningRate);
      this.model.compile({ 
        optimizer: this.optimizer, 
        loss: 'sparseCategoricalCrossentropy', 
        metrics: ['accuracy'] 
      });

      try {
        this.embeddingVocabSize = this.model.getLayer('embedding').getWeights()[0].shape[0];
      } catch(e) {
        this.embeddingVocabSize = this.vocab.size;
      }
      
      this.ready = true;
      console.log(`[AKIE] Transformer carregado. (steps: ${this.trainSteps}, embVocab: ${this.embeddingVocabSize})`);
      return true;
      
     } catch (err) {
    console.error(`[AKIE] Falha ao carregar modelo: ${err.message}`);
    console.log('[AKIE] Modelo corrompido. Limpando diretório para reconstrução...');
    
    try {
      const files = fs.readdirSync(dir);
      for (const file of files) {
        fs.unlinkSync(path.join(dir, file));
      }
      console.log('[AKIE] Diretório limpo. Um novo modelo será construído.');
    } catch (cleanErr) {
      console.error('[AKIE] Erro ao limpar diretório:', cleanErr.message);
    }
    
    return false; // Força rebuild limpo
  }

  /**
   * Expansão Cirúrgica de Vocabulário em Runtime.
   * Redimensiona dinamicamente os kernels mantendo os pesos aprendidos intactos.
   * CORRIGIDO: proteção contra undefined weights e fallback seguro.
   */
  async expandVocabulary(newVocabSize) {
    if (!this.ready) return;
    if (newVocabSize <= this.embeddingVocabSize) return;

    console.log(`[AKIE] Reconfigurando dimensões do Transformer: ${this.embeddingVocabSize} → ${newVocabSize}`);

    const oldWeights = this.model.getWeights();
    const oldVocabSize = this.embeddingVocabSize;
    
    // Atualiza o vocab size antes do rebuild
    this.embeddingVocabSize = newVocabSize;
    
    // Reconstrói o modelo com novo tamanho
    this.build();

    const newWeights = this.model.getWeights();
    const updatedWeights = newWeights.map((w, i) => {
      const oldW = oldWeights[i];
      if (!oldW) {
        console.warn(`[AKIE] Peso ${i} não encontrado nos pesos antigos — inicializando do zero`);
        return w;
      }

      const oldShape = oldW.shape;
      const newShape = w.shape;

      // Dimensões iguais: copia direto
      if (oldShape.length === newShape.length && 
          oldShape.every((dim, idx) => dim === newShape[idx])) {
        return oldW;
      }

      // Caso 1: Kernel do Embedding ([vocabSize, embDim])
      if (oldShape.length === 2 && newShape.length === 2 &&
          newShape[0] > oldShape[0] && oldShape[1] === newShape[1]) {
        const merged = new Float32Array(w.dataSync());
        merged.set(oldW.dataSync());
        return tf.tensor(merged, newShape, 'float32');
      }

      // Caso 2: Kernel do Dense de Saída ([embDim, vocabSize])
      if (oldShape.length === 2 && newShape.length === 2 &&
          newShape[1] > oldShape[1] && oldShape[0] === newShape[0]) {
        const oldData = oldW.dataSync();
        const merged = new Float32Array(w.dataSync());
        const [rows, oldCols] = oldShape;
        const newCols = newShape[1];

        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < oldCols; c++) {
            merged[r * newCols + c] = oldData[r * oldCols + c];
          }
        }
        return tf.tensor(merged, newShape, 'float32');
      }

      // Caso 3: Bias do Dense de Saída ([vocabSize])
      if (oldShape.length === 1 && newShape.length === 1 &&
          newShape[0] > oldShape[0]) {
        const merged = new Float32Array(w.dataSync());
        merged.set(oldW.dataSync());
        return tf.tensor(merged, newShape, 'float32');
      }

      // Fallback: mantém o peso novo
      console.warn(`[AKIE] Peso ${i}: forma incompatível ${oldShape} → ${newShape} — inicializando do zero`);
      return w;
    });

    this.model.setWeights(updatedWeights);
    
    // Libera memória dos pesos antigos
    oldWeights.forEach(w => {
      try { w.dispose(); } catch(e) { /* já foi liberado */ }
    });
    
    console.log(`[AKIE] Expansão concluída com preservação de memória sináptica.`);
  }

  getStats() {
    return {
      architecture: 'Causal Transformer Decoder',
      ready: this.ready,
      vocabSize: this.vocab.size,
      embeddingVocabSize: this.embeddingVocabSize,
      trainSteps: this.trainSteps,
      parameters: this.ready ? this.model.countParams() : 0,
      hparams: this.hparams,
      nlp: { natural: NaturalNLP.isReady, features: NaturalNLP.isReady ? 'stemming+tfidf+similarity' : 'disabled' },
    };
  }

  /**
   * Expõe stemming via NaturalNLP para uso externo (ex: _akie_dataset.js).
   * Permite que o dataset normalize termos antes de tokenizar.
   */
  static stemTokens(tokens) {
    return tokens.map(t => NaturalNLP.stem(t));
  }
}

module.exports = { AKIEModel, HPARAMS, NaturalNLP };
