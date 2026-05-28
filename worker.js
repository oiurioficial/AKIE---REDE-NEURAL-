/**
 * worker.js v2.3  —  AKIE Training Worker + HTTP Endpoint
 * [PATCH] normalizedPrompt: remove \n entre u: e a: antes de gerar
 */

require('dotenv').config();
const admin = require('firebase-admin');
const http  = require('http');
const path  = require('path');
const fs    = require('fs');

const { AKIEModel }                       = require('./_nexus_neural');
const { Vocabulary, tokenizeText }        = require('./_akie_vocab');
const { runBootstrap }                    = require('./_akie_bootstrap');
const { runConversationalSeed }           = require('./_seed_conversacional');
const {
  fetchUserEpisodes,
  episodesToPairs,
  fetchGraphSentences,
  graphSentencesToPairs,
  fetchWebPatterns,
  selfPlayPairs,
} = require('./_akie_dataset');

const {
  processBehavior,
  getDefaultContext,
} = require('./akie_behavior');

const { generateSyntheticConversations }  = require('./_akie_synthetic');
const { runDataIngestion }                = require('./_akie_ingest');

const CONFIG = {
  intervalMs: 60_000,
  modelDir:   process.env.MODEL_DIR || '/data/akie_model',
  httpPort:   parseInt(process.env.PORT || '3000', 10),
  saveEveryN: 5,
  generationLimits: {
    social:  { maxTokens: 40,  temperature: 0.5 },
    analyze: { maxTokens: 200, temperature: 0.5 },
    code:    { maxTokens: 600, temperature: 0.3 },
    refino:  { maxTokens: 120, temperature: 0.4 },
  },
  hparams: {
    embDim:       128,
    hiddenSize:   256,
    maxSeqLen:    64,
    batchSize:    4,
    learningRate: 0.0005,
  },
  consolidation: {
    stagnationDelta:  0.005,
    stagnationCycles: 3,
  },
  behaviorDirectConfidenceThreshold: 0.85,
};

const MODE = {
  INTERACTIVE:    'INTERACTIVE',
  CONSOLIDATION:  'CONSOLIDATION',
  EXPANSION:      'EXPANSION',
  SYNTHETIC:      'SYNTHETIC',
  SELF_PLAY:      'SELF_PLAY',
};

const state = {
  cycle:                0,
  idleCycles:           0,
  lastSaveCycle:        0,
  totalPairsTrained:    0,
  model:                null,
  vocab:                null,
  db:                   null,
  modeHistory:          [],
  modelLoadedFromBinary: false,
  modelIncompatible:    false,
  metrics: {
    lastLoss:                     null,
    lastAccuracy:                 null,
    trainCycles:                  0,
    consolidationLossDelta:       null,
    consolidationStagnantCycles:  0,
    consolidationLastLoss:        null,
  },
  generationStats: {
    social:  0,
    analyze: 0,
    code:    0,
    refino:  0,
    direct:  0,
    neural:  0,
  },
};

let _trainLock = false;
let _saveLock  = false;

function cleanGeneratedText(raw) {
  if (!raw || typeof raw !== 'string') return null;

  let text = raw
    .replace(/<UNK>/gi, '')
    .replace(/\bu\s*:\s*/gi, '')
    .replace(/\ba\s*:\s*/gi, '')
    .replace(/\s([?.!,;:])/g, '$1')
    .replace(/([?.!])\s*\1+/g, '$1')
    .replace(/\s+/g, ' ')
    .trim();

  text = text.replace(/\b(\w+)(\s+\1){2,}/gi, '$1');

  const words = text.split(/\s+/).filter(w => w.length > 1 && /[a-záéíóúãõâêô]/i.test(w));
  if (words.length < 2) return null;

  return text.charAt(0).toUpperCase() + text.slice(1);
}

async function safeSave() {
  if (_saveLock) {
    console.log('[SAVE] Save já em andamento — ignorando chamada duplicada.');
    return;
  }
  _saveLock = true;
  try {
    await state.model.save(CONFIG.modelDir);
    await saveVocab(state.vocab);
    state.lastSaveCycle = state.cycle;
    await reportStatus('periodic');
  } catch (err) {
    console.error('[SAVE] Falha ao salvar:', err.message);
  } finally {
    _saveLock = false;
  }
}

function startHttpServer() {
  const server = http.createServer(async (req, res) => {
    const headers = {
      'Content-Type':                'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods':'POST, GET, OPTIONS',
      'Access-Control-Allow-Headers':'Content-Type, Authorization',
    };

    if (req.method === 'OPTIONS') {
      res.writeHead(204, headers);
      res.end();
      return;
    }

    if (req.method === 'GET' && req.url === '/status') {
      const stats = state.model ? state.model.getStats() : { ready: false };
      res.writeHead(200, headers);
      res.end(JSON.stringify({
        ok:                true,
        cycle:             state.cycle,
        idle_cycles:       state.idleCycles,
        metrics:           state.metrics,
        model:             stats,
        mode_history:      state.modeHistory.slice(-5),
        generation_stats:  state.generationStats,
        model_incompatible: state.modelIncompatible,
      }));
      return;
    }

    if (req.method === 'POST' && req.url === '/generate') {
      let body = '';
      req.on('data', chunk => { body += chunk; });
      req.on('end', async () => {
        try {
          const { prompt, max_tokens, temperature } = JSON.parse(body || '{}');

          if (!prompt) {
            res.writeHead(400, headers);
            res.end(JSON.stringify({ error: 'Campo prompt ausente.' }));
            return;
          }

          if (!state.model || !state.modelLoadedFromBinary) {
            console.log('[GENERATE] Modelo não em memória — carregando do binário...');
            const loaded = await tryLoadModelFromBinary();
            if (!loaded) {
              res.writeHead(503, headers);
              res.end(JSON.stringify({
                error:       'Modelo não disponível.',
                binary_path: path.join(CONFIG.modelDir, 'weights.bin'),
              }));
              return;
            }
          }

          if (!state.model.ready) {
            res.writeHead(503, headers);
            res.end(JSON.stringify({ error: 'Modelo carregado mas não pronto para inferência.' }));
            return;
          }

          const behavior = processBehavior(prompt, {
            language:          'pt',
            refineryAvailable: false,
          });

          console.log(`[GENERATE] Modo: ${behavior.mode} | Confidence: ${(behavior.confidence || 0).toFixed(2)} | Input: "${prompt.substring(0, 80)}${prompt.length > 80 ? '...' : ''}"`);

          const useDirectOutput = behavior.output &&
            behavior.mode !== 'social' &&
            (behavior.confidence || 0) >= CONFIG.behaviorDirectConfidenceThreshold;

          if (useDirectOutput) {
            state.generationStats.direct++;
            console.log(`[CHAT-DEBUG] Usuário: ${prompt} | IA: ${behavior.output} [direct]`);
            res.writeHead(200, headers);
            res.end(JSON.stringify({
              ok:         true,
              prompt:     prompt,
              generated:  behavior.output,
              mode:       behavior.mode,
              confidence: behavior.confidence,
              source:     'behavior_direct',
            }));
            return;
          }

          const modeLimits   = CONFIG.generationLimits[behavior.mode] || CONFIG.generationLimits.social;
          const genMaxTokens = max_tokens  || modeLimits.maxTokens;
          const genTemp      = temperature || modeLimits.temperature;

          // [PATCH v2.3.1] Normalizar \n entre u: e a: — evita token UNK no separador
          const normalizedPrompt = prompt.replace(/\n\s*/g, ' ').trim();
          const rawGenerated = await state.model.generate(normalizedPrompt, genMaxTokens, genTemp);

          const cleanGenerated = cleanGeneratedText(rawGenerated);

          if (!cleanGenerated) {
            const fallbackText = behavior.output || 'Ainda estou aprendendo. Pode reformular?';
            state.generationStats.direct = (state.generationStats.direct || 0) + 1;
            console.log(`[CHAT-DEBUG] Usuário: ${prompt} | IA: ${fallbackText} [fallback-vazio]`);
            res.writeHead(200, headers);
            res.end(JSON.stringify({
              ok:         true,
              prompt:     prompt,
              generated:  fallbackText,
              mode:       behavior.mode,
              confidence: behavior.confidence,
              source:     'behavior_fallback',
            }));
            return;
          }

          state.generationStats.neural = (state.generationStats.neural || 0) + 1;
          state.generationStats[behavior.mode] = (state.generationStats[behavior.mode] || 0) + 1;

          console.log(`[CHAT-DEBUG] Usuário: ${prompt} | IA: ${cleanGenerated} [neural]`);
          res.writeHead(200, headers);
          res.end(JSON.stringify({
            ok:         true,
            prompt:     prompt,
            generated:  cleanGenerated,
            mode:       behavior.mode,
            confidence: behavior.confidence,
            source:     'neural',
            tokens:     cleanGenerated.split(/\s+/).length,
          }));
        } catch (err) {
          console.error('[GENERATE] Erro:', err.message);
          res.writeHead(500, headers);
          res.end(JSON.stringify({ error: err.message }));
        }
      });
      return;
    }

    res.writeHead(404, headers);
    res.end(JSON.stringify({ error: 'Endpoint não encontrado.' }));
  });

  server.listen(CONFIG.httpPort, () => {
    console.log(`[HTTP] Servidor listening na porta ${CONFIG.httpPort}`);
  });

  server.on('error', err => {
    console.error('[HTTP] Erro no servidor:', err.message);
  });
}

async function init() {
  try {
    const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT || '{}');
    admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
    state.db = admin.firestore();
    console.log('[INIT] Firebase conectado.');
  } catch (err) {
    console.error('[INIT] Firebase indisponível:', err.message);
    state.db = null;
  }

  state.vocab = await loadOrCreateVocab();
  console.log(`[INIT] Vocabulário: ${state.vocab.size} tokens`);

  state.model = new AKIEModel(state.vocab, CONFIG.hparams);
  console.log(`[INIT] Modelo criado: embDim=${CONFIG.hparams.embDim}, hiddenSize=${CONFIG.hparams.hiddenSize}, maxSeqLen=${CONFIG.hparams.maxSeqLen}`);

  const loaded = await state.model.load(CONFIG.modelDir);
  if (!loaded) {
    console.log('[INIT] Nenhum modelo anterior — inicializando novo...');
    state.model.build();
    state.model.ready = true;
  }

  state.modelLoadedFromBinary = true;

  console.log(`[INIT] Modelo pronto: ${state.model.ready ? '✓' : '✗'}`);

  if (state.db) {
    try {
      const graphSnap = await state.db.collection('nexus_graph').limit(50).get();
      if (graphSnap.size < 50) {
        console.log(`[INIT] Grafo pequeno (${graphSnap.size} nós) — forçando seed conversacional...`);
        await state.db.collection('akie_worker_status').doc('seed_conversacional').delete().catch(() => {});
        await runConversationalSeed(state.db);
      }
    } catch (e) {
      console.error('[INIT] Erro ao verificar seed:', e.message);
    }
  }
}

async function tryLoadModelFromBinary() {
  try {
    const metaPath = path.join(CONFIG.modelDir, 'akie_meta.json');
    if (!fs.existsSync(metaPath)) {
      console.log('[LOAD] Meta não encontrado — novo modelo.');
      state.model.build();
      state.model.ready          = true;
      state.modelLoadedFromBinary = true;
      return true;
    }

    const meta          = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
    const loadedHparams = meta.hparams || {};

    if (
      loadedHparams.embDim     !== CONFIG.hparams.embDim     ||
      loadedHparams.hiddenSize !== CONFIG.hparams.hiddenSize  ||
      loadedHparams.maxSeqLen  !== CONFIG.hparams.maxSeqLen
    ) {
      console.warn('\n╔══════════════════════════════════════════════════════════════╗');
      console.warn('║  ⚠️  RESET AUTOMÁTICO DISPARADO                               ║');
      console.warn('╠══════════════════════════════════════════════════════════════╣');
      console.warn(`║  Carregado: [emb=${loadedHparams.embDim || 64} | hidden=${loadedHparams.hiddenSize || 128} | seq=${loadedHparams.maxSeqLen || 32}]`);
      console.warn(`║  Esperado:  [emb=${CONFIG.hparams.embDim} | hidden=${CONFIG.hparams.hiddenSize} | seq=${CONFIG.hparams.maxSeqLen}]`);
      console.warn('╚══════════════════════════════════════════════════════════════╝\n');

      const bakPath = `${CONFIG.modelDir}_v1_backup`;
      if (fs.existsSync(CONFIG.modelDir)) {
        try {
          await fs.promises.rm(bakPath, { recursive: true, force: true });
          await fs.promises.mkdir(path.dirname(bakPath), { recursive: true });
          await fs.promises.rename(CONFIG.modelDir, bakPath);
          console.log(`[LOAD] Backup criado em: ${bakPath}`);
        } catch (e) {
          console.warn(`[LOAD] Não foi possível fazer backup: ${e.message}`);
        }
      }

      state.modelIncompatible    = true;
      state.model.build();
      state.model.ready          = true;
      state.modelLoadedFromBinary = true;
      return true;
    }

    const loaded = await state.model.load(CONFIG.modelDir);
    state.modelLoadedFromBinary = true;
    return loaded || state.model.ready;
  } catch (err) {
    console.error('[LOAD] Erro ao carregar modelo:', err.message);
    state.model.build();
    state.model.ready          = true;
    state.modelLoadedFromBinary = true;
    return true;
  }
}

let schedulerRunning = false;

async function scheduler() {
  if (schedulerRunning) {
    console.log('[SCHEDULER] Ciclo anterior ainda em execução...');
    return;
  }

  schedulerRunning = true;
  const t0 = Date.now();

  try {
    state.cycle++;
    console.log(`\n[CYCLE ${state.cycle}] ─────────────────────────────────────────`);

    let mode     = MODE.INTERACTIVE;
    let episodes = [];

    try {
      episodes = state.db ? await fetchUserEpisodes(state.db, 1) : [];
    } catch (err) {
      console.error('[SCHEDULER] Erro ao buscar episódios:', err.message);
    }

    if (episodes.length > 0) {
      mode = MODE.INTERACTIVE;
    } else {
      const consolidationStuck =
        state.metrics.consolidationStagnantCycles >= CONFIG.consolidation.stagnationCycles;

      const slot = (state.idleCycles % 4);
      if (slot === 0) {
        mode = MODE.SYNTHETIC;
      } else if (slot === 1) {
        mode = consolidationStuck ? MODE.EXPANSION : MODE.CONSOLIDATION;
        if (consolidationStuck) {
          console.log(`[SCHEDULER] CONSOLIDATION estagnada (${state.metrics.consolidationStagnantCycles} ciclos) — redirecionando para EXPANSION`);
          state.metrics.consolidationStagnantCycles = 0;
        }
      } else if (slot === 2) {
        mode = MODE.EXPANSION;
      } else if (slot === 3) {
        mode = MODE.SELF_PLAY;
      }
    }

    await runMode(mode, { tag: `[${state.cycle}]`, episodes });

    if (episodes.length > 0) {
      state.idleCycles = 0;
    } else {
      state.idleCycles++;
    }

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    console.log(`[CYCLE ${state.cycle}] ✓ Concluído em ${elapsed}s\n`);
  } catch (err) {
    console.error(`[SCHEDULER] ERRO:`, err.message);
  } finally {
    schedulerRunning = false;
  }
}

async function runMode(mode, ctx = {}) {
  if (_trainLock) {
    const tag = ctx.tag || '';
    console.log(`${tag} trainLock ativo — ${mode} ignorado`);
    return;
  }

  const { tag = '', episodes = [] } = ctx;
  const t0   = Date.now();
  let pairs  = [];
  let desc   = '';

  console.log(`${tag} Modo: ${mode}`);

  switch (mode) {

    case MODE.INTERACTIVE: {
      await maybeExpandVocab(episodes);

      pairs = episodesToPairs(episodes, state.vocab);
      desc  = `${episodes.length} episódios do usuário → ${pairs.length} pares`;

      if (pairs.length === 0) {
        console.log(`${tag} INTERACTIVE: nenhum par gerado a partir dos episódios`);
        state.modeHistory.push(mode);
        if (state.modeHistory.length > 20) state.modeHistory.shift();
        return;
      }

      shuffleArray(pairs);
      _trainLock = true;
      let result;
      try {
        result = await state.model.trainBatch(pairs, 3);
      } catch (trainErr) {
        console.error(`${tag} Erro durante trainBatch (INTERACTIVE):`, trainErr.message);
      } finally {
        _trainLock = false;
      }

      const { normLoss, normAcc } = extractMetrics(result);
      const isFiniteNum = v => typeof v === 'number' && isFinite(v) && !isNaN(v);

      state.metrics.lastLoss     = isFiniteNum(normLoss) ? normLoss : state.metrics.lastLoss;
      state.metrics.lastAccuracy = isFiniteNum(normAcc)  ? normAcc  : state.metrics.lastAccuracy;
      state.metrics.trainCycles++;
      state.totalPairsTrained += pairs.length;

      const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
      const lossStr = isFiniteNum(normLoss) ? normLoss.toFixed(4) : 'n/a';
      const accStr  = isFiniteNum(normAcc)  ? (normAcc * 100).toFixed(1) + '%' : 'n/a';
      console.log(`${tag} ✓ ${desc} | loss=${lossStr} acc=${accStr} | ${elapsed}s`);

      await markEpisodesAsQueued(episodes);

      if (state.cycle - state.lastSaveCycle >= CONFIG.saveEveryN) {
        await safeSave();
      }

      state.modeHistory.push(mode);
      if (state.modeHistory.length > 20) state.modeHistory.shift();
      return;
    }

    case MODE.CONSOLIDATION: {
      const sentences = await fetchGraphSentences(state.db, 300);
      if (!sentences.length) {
        console.log(`${tag} Grafo vazio — pulando consolidação`);
        return;
      }
      pairs = graphSentencesToPairs(sentences, state.vocab);
      desc  = `${sentences.length} frases do grafo → ${pairs.length} pares`;
      break;
    }

    case MODE.EXPANSION: {
      const vocabBefore = state.vocab.size;

      // 1. Ingestão local (Tatoeba/conversacional)
      const ingestResult = await runDataIngestion(state.vocab, { batchSize: 500 });
      pairs = ingestResult.pairs;

      // 2. Web via Serper/Tavily — só roda se a key estiver configurada.
      // fetchWebPatterns busca 3 queries por ciclo, extrai sentenças,
      // expande o nexus_graph e retorna pares de teacher-forcing.
      if (process.env.SERPER_API_KEY || process.env.TAVILY_API_KEY) {
        try {
          const webPairs = await fetchWebPatterns(state.db, state.vocab, 3);
          if (webPairs.length > 0) {
            pairs = [...pairs, ...webPairs];
            console.log(`[EXPANSION] Serper adicionou ${webPairs.length} pares ao ciclo`);
          }
        } catch (webErr) {
          console.error('[EXPANSION] fetchWebPatterns falhou (não fatal):', webErr.message);
        }
      }

      desc = `ingest (${ingestResult.sentences} frases) + web → ${pairs.length} pares`;

      if (state.vocab.size > vocabBefore) {
        await saveVocab(state.vocab);
        await maybeRebuildForVocabGrowth();
      }
      break;
    }

    case MODE.SYNTHETIC: {
      const syntheticPairs = await generateSyntheticConversations(state.vocab, 200);
      pairs = syntheticPairs;
      desc  = `synthetic conversacional → ${pairs.length} pares`;
      break;
    }

    case MODE.SELF_PLAY: {
      pairs = await selfPlayPairs(state.model, state.db, state.vocab, 15);
      desc  = `self-play → ${pairs.length} pares`;
      break;
    }
  }

  if (pairs.length > 0) {
    shuffleArray(pairs);

    _trainLock = true;
    let result;
    try {
      result = await state.model.trainBatch(pairs, 3);
    } catch (trainErr) {
      console.error(`${tag} Erro durante trainBatch:`, trainErr.message);
    } finally {
      _trainLock = false;
    }

    const { normLoss, normAcc } = extractMetrics(result);
    const isFiniteNum = v => typeof v === 'number' && isFinite(v) && !isNaN(v);

    state.metrics.lastLoss     = isFiniteNum(normLoss) ? normLoss : state.metrics.lastLoss;
    state.metrics.lastAccuracy = isFiniteNum(normAcc)  ? normAcc  : state.metrics.lastAccuracy;
    state.metrics.trainCycles++;
    state.totalPairsTrained += pairs.length;

    if (mode === MODE.CONSOLIDATION && isFiniteNum(normLoss)) {
      const prevLoss = state.metrics.consolidationLastLoss;
      if (prevLoss !== null && prevLoss !== undefined) {
        const delta = Math.abs(normLoss - prevLoss);
        state.metrics.consolidationLossDelta = delta;

        if (delta < CONFIG.consolidation.stagnationDelta) {
          state.metrics.consolidationStagnantCycles++;
          console.log(`${tag} [P2] Estagnação CONSOLIDATION: delta=${delta.toFixed(5)} | ciclos=${state.metrics.consolidationStagnantCycles}/${CONFIG.consolidation.stagnationCycles}`);
        } else {
          state.metrics.consolidationStagnantCycles = 0;
        }
      }
      state.metrics.consolidationLastLoss = normLoss;
    }

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const lossStr = isFiniteNum(normLoss) ? normLoss.toFixed(4) : 'n/a';
    const accStr  = isFiniteNum(normAcc)  ? (normAcc * 100).toFixed(1) + '%' : 'n/a';
    console.log(`${tag} ✓ ${desc} | loss=${lossStr} acc=${accStr} | ${elapsed}s`);

    if (state.cycle - state.lastSaveCycle >= CONFIG.saveEveryN) {
      await safeSave();
    }
  } else {
    console.log(`${tag} Nenhum dado disponível para ${mode}`);
  }

  state.modeHistory.push(mode);
  if (state.modeHistory.length > 20) state.modeHistory.shift();
}

function extractMetrics(result) {
  const safeResult = result || { loss: null, accuracy: null };
  const rawLoss    = safeResult.loss;
  const rawAcc     = safeResult.accuracy;
  const normLoss   = (rawLoss != null && typeof rawLoss.dataSync === 'function')
    ? rawLoss.dataSync()[0]
    : rawLoss;
  const normAcc    = (rawAcc != null && typeof rawAcc.dataSync === 'function')
    ? rawAcc.dataSync()[0]
    : rawAcc;
  return { normLoss, normAcc };
}

async function maybeExpandVocab(episodes) {
  const before = state.vocab.size;
  for (const ep of episodes) {
    if (ep.input)  state.vocab.addTokens(tokenizeText(ep.input));
    if (ep.output) state.vocab.addTokens(tokenizeText(ep.output));
  }
  if (state.vocab.size > before) {
    console.log(`[VOCAB] Cresceu: ${before} → ${state.vocab.size}`);
    await maybeRebuildForVocabGrowth();
  }
}

async function maybeRebuildForVocabGrowth() {
  const curr    = state.vocab.size;
  const embSize = state.model.getStats().embeddingVocabSize;
  if (curr > embSize) {
    console.log(`[VOCAB] Expandindo embedding: ${embSize} → ${curr}`);
    await state.model.expandVocabulary(curr);
    state.model.vocab = state.vocab;
  }
}

async function markEpisodesAsQueued(episodes) {
  if (!state.db || !episodes.length) return;
  try {
    const batch = state.db.batch();
    const now   = new Date().toISOString();

    for (const ep of episodes) {
      if (!ep.id) continue;
      const ref = state.db.collection('user_episodes').doc(ep.id);
      batch.set(ref, {
        queued_for_consolidation: true,
        queued_at:                now,
        trained_at:               now,
      }, { merge: true });
    }

    await batch.commit();
    console.log(`[EPISODES] Marcados ${episodes.length} episódios como treinados`);
  } catch (err) {
    console.error('[EPISODES] Erro ao marcar episódios (não fatal):', err.message);
  }
}

async function loadOrCreateVocab() {
  const p = path.join(CONFIG.modelDir, 'akie_vocab.json');
  try {
    if (fs.existsSync(p)) return Vocabulary.fromJSON(JSON.parse(fs.readFileSync(p, 'utf8')));
  } catch (e) { /* ignora */ }

  try {
    if (state.db) {
      const doc = await state.db.collection('akie_worker_status').doc('vocabulary').get();
      if (doc.exists) return Vocabulary.fromJSON(doc.data());
    }
  } catch (e) { /* ignora */ }

  return new Vocabulary();
}

async function saveVocab(vocab) {
  try {
    fs.mkdirSync(CONFIG.modelDir, { recursive: true });
    fs.writeFileSync(
      path.join(CONFIG.modelDir, 'akie_vocab.json'),
      JSON.stringify(vocab.toJSON(), null, 2)
    );
  } catch (e) {
    try {
      if (state.db) {
        await state.db.collection('akie_worker_status').doc('vocabulary')
          .set(vocab.toJSON(), { merge: true });
      }
    } catch (e2) {
      console.error('[VOCAB] Falha ao salvar:', e2.message);
    }
  }
}

async function reportStatus(mode) {
  if (!state.db) return;
  try {
    await state.db.collection('akie_worker_status').doc('current').set({
      updated_at:       new Date().toISOString(),
      status:           'running',
      cycle:            state.cycle,
      idle_cycles:      state.idleCycles,
      last_mode:        mode,
      mode_history:     state.modeHistory,
      total_pairs:      state.totalPairsTrained,
      metrics:          state.metrics,
      model_stats:      state.model.getStats(),
      generation_stats: state.generationStats,
      model_version:    '2.3.1',
    }, { merge: true });
  } catch (e) { /* não crítico */ }
}

function shuffleArray(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

let lastHeartbeat = Date.now();

function startWatchdog() {
  setInterval(() => {
    if (Date.now() - lastHeartbeat > 5 * 60 * 1000) {
      console.error('[WATCHDOG] Loop travado. Forçando ciclo...');
      scheduler().catch(console.error);
      lastHeartbeat = Date.now();
    }
  }, 30_000);
}

async function main() {
  await init();
  startHttpServer();
  startWatchdog();

  await scheduler();
  lastHeartbeat = Date.now();

  setInterval(async () => {
    await scheduler();
    lastHeartbeat = Date.now();
  }, CONFIG.intervalMs);

  process.on('SIGTERM', async () => {
    console.log('\n[WORKER] SIGTERM — aguardando conclusão...');

    const deadline = Date.now() + 90_000;
    while ((_trainLock || _saveLock) && Date.now() < deadline) {
      await new Promise(r => setTimeout(r, 500));
    }
    if (_trainLock) console.warn('[WORKER] Timeout aguardando trainLock.');
    if (_saveLock)  console.warn('[WORKER] Timeout aguardando saveLock.');

    await safeSave();

    try {
      if (state.db) {
        await state.db.collection('akie_worker_status').doc('current').set({
          status:     'stopped',
          stopped_at: new Date().toISOString(),
        }, { merge: true });
      }
    } catch { /* ignora */ }

    process.exit(0);
  });
}

main().catch(err => {
  console.error('[WORKER] Erro fatal:', err);
  process.exit(1);
});
