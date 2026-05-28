/**
 * worker.js v2.1  —  AKIE Training Worker + HTTP Endpoint
 *
 * CORREÇÃO v2.1:
 *   [FIX-1] Firebase: .set({...}, {merge: true}) em vez de .update()
 *   [FIX-2] Scheduler: EXPANSION a cada 3 ciclos + novo modo SYNTHETIC
 *   [FIX-3] Injeção de dados conversacionais para destravar aprendizado
 *
 * UPGRADE MODERADO:
 *   embDim: 64 → 128
 *   hiddenSize: 128 → 256
 *   maxSeqLen: 32 → 64
 *   Parâmetros: ~260k → ~1.95M
 *
 * Roda 24h no Railway. Nunca para.
 * Expõe endpoint HTTP para o NEXUS chamar geração em tempo real.
 *
 *  ┌─────────────────────────────────────────────────────────────┐
 *  │  MODO 1 — INTERACTIVE  (usuário interagiu)                  │
 *  │  MODO 2 — CONSOLIDATION (replay do grafo semântico)         │
 *  │  MODO 3 — EXPANSION    (web crawl, lacunas do grafo)        │
 *  │  MODO 4 — SYNTHETIC    (injeção de dados conversacionais)   │
 *  │  MODO 5 — SELF_PLAY    (modelo gera e treina nos melhores)  │
 *  └─────────────────────────────────────────────────────────────┘
 */

require('dotenv').config();
const admin = require('firebase-admin');
const http = require('http');
const path = require('path');
const fs = require('fs');

const { AKIEModel } = require('./_nexus_neural');
const { Vocabulary, tokenizeText } = require('./_akie_vocab');
const { runBootstrap } = require('./_akie_bootstrap');
const { runConversationalSeed } = require('./_seed_conversacional');
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

// [FIX-2] Novo módulo de geração sintética
const { generateSyntheticConversations } = require('./_akie_synthetic');

// ---------------------------------------------------------------------------
// Configuração com suporte a upgrade
// ---------------------------------------------------------------------------

const CONFIG = {
  intervalMs: 60_000,
  modelDir: process.env.MODEL_DIR || '/data/akie_model',
  httpPort: parseInt(process.env.PORT || '3000', 10),
  saveEveryN: 5,
  consolidationAfter: 1,
  expansionAfter: 2,  // [FIX-2] reduzido de 3 para forçar EXPANSION a cada 3 ciclos
  syntheticAfter: 1,  // [FIX-2] novo: injetar dados sintéticos frequentemente
  selfPlayAfter: 2,
  // Upgrade v2.0: novos limites refletindo maior capacidade
  generationLimits: {
    social: { maxTokens: 80, temperature: 0.7 },
    analyze: { maxTokens: 200, temperature: 0.5 },
    code: { maxTokens: 600, temperature: 0.3 },
    refino: { maxTokens: 120, temperature: 0.4 },
  },
  // Hiperparâmetros do modelo v2.0
  hparams: {
    embDim: 128,
    hiddenSize: 256,
    maxSeqLen: 64,
    batchSize: 4,
    learningRate: 0.0005,
  },
};

const MODE = {
  INTERACTIVE: 'INTERACTIVE',
  CONSOLIDATION: 'CONSOLIDATION',
  EXPANSION: 'EXPANSION',
  SYNTHETIC: 'SYNTHETIC',      // [FIX-2] novo modo
  SELF_PLAY: 'SELF_PLAY',
};

// ---------------------------------------------------------------------------
// Estado global
// ---------------------------------------------------------------------------

const state = {
  cycle: 0,
  idleCycles: 0,
  lastSaveCycle: 0,
  totalPairsTrained: 0,
  model: null,
  vocab: null,
  db: null,
  modeHistory: [],
  modelLoadedFromBinary: false,
  modelIncompatible: false,
  metrics: {
    lastLoss: null,
    lastAccuracy: null,
    trainCycles: 0,
  },
  generationStats: {
    social: 0,
    analyze: 0,
    code: 0,
    refino: 0,
    direct: 0,
  },
};

// ---------------------------------------------------------------------------
// Mutexes de exclusão mútua
// ---------------------------------------------------------------------------

let _trainLock = false;
let _saveLock = false;

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

// ---------------------------------------------------------------------------
// Servidor HTTP
// ---------------------------------------------------------------------------

function startHttpServer() {
  const server = http.createServer(async (req, res) => {
    const headers = {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    };

    if (req.method === 'OPTIONS') {
      res.writeHead(204, headers);
      res.end();
      return;
    }

    // ── GET /status ────────────────────────────────────────────
    if (req.method === 'GET' && req.url === '/status') {
      const stats = state.model ? state.model.getStats() : { ready: false };
      res.writeHead(200, headers);
      res.end(JSON.stringify({
        ok: true,
        cycle: state.cycle,
        idle_cycles: state.idleCycles,
        metrics: state.metrics,
        model: stats,
        mode_history: state.modeHistory.slice(-5),
        generation_stats: state.generationStats,
        model_incompatible: state.modelIncompatible,
      }));
      return;
    }

    // ── POST /generate ─────────────────────────────────────────
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

          // Lazy load do modelo
          if (!state.model || !state.modelLoadedFromBinary) {
            console.log('[GENERATE] Modelo não em memória — carregando do binário...');
            const loaded = await tryLoadModelFromBinary();
            if (!loaded) {
              res.writeHead(503, headers);
              res.end(JSON.stringify({
                error: 'Modelo não disponível.',
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

          // Processar behavior
          const behavior = processBehavior(prompt, {
            language: 'pt',
            refineryAvailable: false,
          });

          console.log(`[GENERATE] Inferência | Modo: ${behavior.mode} | Input: "${prompt.substring(0, 80)}${prompt.length > 80 ? '...' : ''}"`);

          // Output direto
          if (behavior.output) {
            state.generationStats.direct++;
            res.writeHead(200, headers);
            res.end(JSON.stringify({
              ok: true,
              prompt: prompt,
              generated: behavior.output,
              mode: behavior.mode,
              source: 'behavior_direct',
            }));
            return;
          }

          // Inferência neural
          const modeLimits = CONFIG.generationLimits[behavior.mode] || CONFIG.generationLimits.social;
          const genMaxTokens = max_tokens || modeLimits.maxTokens;
          const genTemp = temperature || modeLimits.temperature;

          const generated = await state.model.generate(prompt, genMaxTokens, genTemp);

          if (!generated || generated.length === 0) {
            res.writeHead(503, headers);
            res.end(JSON.stringify({ error: 'Geração falhou — modelo pode estar treino/aquecimento.' }));
            return;
          }

          state.generationStats[behavior.mode] = (state.generationStats[behavior.mode] || 0) + 1;

          res.writeHead(200, headers);
          res.end(JSON.stringify({
            ok: true,
            prompt: prompt,
            generated: generated,
            mode: behavior.mode,
            source: 'neural',
            tokens: generated.split(/\s+/).length,
          }));
        } catch (err) {
          console.error('[GENERATE] Erro:', err.message);
          res.writeHead(500, headers);
          res.end(JSON.stringify({ error: err.message }));
        }
      });
      return;
    }

    // 404
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

// ---------------------------------------------------------------------------
// Inicialização
// ---------------------------------------------------------------------------

async function init() {
  try {
    const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT || '{}');
    admin.initializeApp({
      credential: admin.credential.cert(serviceAccount),
    });
    state.db = admin.firestore();
    console.log('[INIT] Firebase conectado.');
  } catch (err) {
    console.error('[INIT] Firebase indisponível:', err.message);
    state.db = null;
  }

  // Carregar ou criar vocabulário
  state.vocab = await loadOrCreateVocab();
  console.log(`[INIT] Vocabulário: ${state.vocab.size} tokens`);

  // Criar modelo
  state.model = new AKIEModel(state.vocab, CONFIG.hparams);
  console.log(`[INIT] Modelo criado com hparams: embDim=${CONFIG.hparams.embDim}, hiddenSize=${CONFIG.hparams.hiddenSize}, maxSeqLen=${CONFIG.hparams.maxSeqLen}`);

  // Tentar carregar modelo
  const loaded = await state.model.load(CONFIG.modelDir);
  if (!loaded) {
    console.log('[INIT] Nenhum modelo anterior — inicializando novo...');
    state.model.build();
    state.model.ready = true;
  }

  console.log(`[INIT] Modelo pronto: ${state.model.ready ? '✓' : '✗'}`);

  // Seed conversacional — roda sempre que o grafo estiver pequeno (< 50 nós)
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

/**
 * Detecta se um modelo carregado é incompatível e força reset
 */
async function tryLoadModelFromBinary() {
  try {
    const metaPath = path.join(CONFIG.modelDir, 'akie_meta.json');
    if (!fs.existsSync(metaPath)) {
      console.log('[LOAD] Meta não encontrado — novo modelo.');
      state.model.build();
      state.model.ready = true;
      state.modelLoadedFromBinary = true;
      return true;
    }

    const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
    const loadedHparams = meta.hparams || {};

    // Validar compatibilidade
    if (
      loadedHparams.embDim !== CONFIG.hparams.embDim ||
      loadedHparams.hiddenSize !== CONFIG.hparams.hiddenSize ||
      loadedHparams.maxSeqLen !== CONFIG.hparams.maxSeqLen
    ) {
      console.warn('\n╔══════════════════════════════════════════════════════════════╗');
      console.warn('║  ⚠️  RESET AUTOMÁTICO DISPARADO                                ║');
      console.warn('╠══════════════════════════════════════════════════════════════╣');
      console.warn(`║  Carregado: [emb=${loadedHparams.embDim || 64} | hidden=${loadedHparams.hiddenSize || 128} | seq=${loadedHparams.maxSeqLen || 32}]`);
      console.warn(`║  Esperado:  [emb=${CONFIG.hparams.embDim} | hidden=${CONFIG.hparams.hiddenSize} | seq=${CONFIG.hparams.maxSeqLen}]`);
      console.warn('║  → Removendo modelo antigo e inicializando novo...              ║');
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

      state.modelIncompatible = true;
      state.model.build();
      state.model.ready = true;
      state.modelLoadedFromBinary = true;
      return true;
    }

    // Compatível — carregar normalmente
    const loaded = await state.model.load(CONFIG.modelDir);
    state.modelLoadedFromBinary = true;
    return loaded || state.model.ready;
  } catch (err) {
    console.error('[LOAD] Erro ao carregar modelo:', err.message);
    state.model.build();
    state.model.ready = true;
    state.modelLoadedFromBinary = true;
    return true;
  }
}

// ---------------------------------------------------------------------------
// Scheduler principal
// ---------------------------------------------------------------------------

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

    // Decidir modo
    let mode = MODE.INTERACTIVE;
    let episodes = [];

    try {
      episodes = state.db ? await fetchUserEpisodes(state.db, 1) : [];
    } catch (err) {
      console.error('[SCHEDULER] Erro ao buscar episódios:', err.message);
    }

    if (episodes.length > 0) {
      mode = MODE.INTERACTIVE;
    } else {
      // [FIX-2] Rotação compactada para destravar aprendizado:
      //   padrão de 4 ciclos: SYNTHETIC, CONSOLIDATION, EXPANSION, SELF_PLAY
      const slot = (state.idleCycles % 4);
      if (slot === 0) {
        mode = MODE.SYNTHETIC;        // Injeta dados conversacionais novos
      } else if (slot === 1) {
        mode = MODE.CONSOLIDATION;    // Aprende sobre grafo existente
      } else if (slot === 2) {
        mode = MODE.EXPANSION;        // Web crawl + novos padrões
      } else if (slot === 3) {
        mode = MODE.SELF_PLAY;        // Auto-geração (se houver treino anterior)
      }
    }

    // Executar modo
    await runMode(mode, { tag: `[${state.cycle}]`, episodes });

    // idleCycles conta ciclos consecutivos sem interação real de usuário
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

// ---------------------------------------------------------------------------
// Executor de modo
// ---------------------------------------------------------------------------

async function runMode(mode, ctx = {}) {
  if (_trainLock) {
    const tag = ctx.tag || '';
    console.log(`${tag} trainLock ativo — ${mode} ignorado`);
    return;
  }

  const { tag = '', episodes = [] } = ctx;
  const t0 = Date.now();
  let pairs = [];
  let desc = '';

  console.log(`${tag} Modo: ${mode}`);

  switch (mode) {
    case MODE.INTERACTIVE: {
      await maybeExpandVocab(episodes);
      await markEpisodesAsQueued(episodes);  // [FIX-1] agora sem NOT_FOUND
      console.log(`${tag} INTERACTIVE: ${episodes.length} episódios enfileirados`);
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
      desc = `${sentences.length} frases do grafo → ${pairs.length} pares`;
      break;
    }
    case MODE.EXPANSION: {
      const vocabBefore = state.vocab.size;
      pairs = await fetchWebPatterns(state.db, state.vocab, 1);
      desc = `web crawl → ${pairs.length} pares`;
      if (state.vocab.size > vocabBefore) {
        await saveVocab(state.vocab);
        await maybeRebuildForVocabGrowth();
      }
      break;
    }
    // [FIX-2] Novo modo SYNTHETIC para injetar dados conversacionais
    case MODE.SYNTHETIC: {
      const syntheticPairs = await generateSyntheticConversations(state.vocab, 200);
      pairs = syntheticPairs;
      desc = `synthetic conversacional → ${pairs.length} pares`;
      break;
    }
    case MODE.SELF_PLAY: {
      pairs = await selfPlayPairs(state.model, state.db, state.vocab, 15);
      desc = `self-play → ${pairs.length} pares`;
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

    const safeResult = result || { loss: null, accuracy: null };

    const rawLoss = safeResult.loss;
    const rawAcc  = safeResult.accuracy;
    const normLoss = (rawLoss != null && typeof rawLoss.dataSync === 'function')
      ? rawLoss.dataSync()[0]
      : rawLoss;
    const normAcc  = (rawAcc != null && typeof rawAcc.dataSync === 'function')
      ? rawAcc.dataSync()[0]
      : rawAcc;
    const isFiniteNum = v => typeof v === 'number' && isFinite(v) && !isNaN(v);

    state.metrics.lastLoss     = isFiniteNum(normLoss) ? normLoss : state.metrics.lastLoss;
    state.metrics.lastAccuracy = isFiniteNum(normAcc)  ? normAcc  : state.metrics.lastAccuracy;
    state.metrics.trainCycles++;
    state.totalPairsTrained += pairs.length;

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const loss = isFiniteNum(normLoss) ? normLoss.toFixed(4) : 'n/a';
    const acc  = isFiniteNum(normAcc)  ? (normAcc * 100).toFixed(1) + '%' : 'n/a';
    console.log(`${tag} ✓ ${desc} | loss=${loss} acc=${acc} | ${elapsed}s`);

    if (state.cycle - state.lastSaveCycle >= CONFIG.saveEveryN) {
      await safeSave();
    }
  } else {
    console.log(`${tag} Nenhum dado disponível para ${mode}`);
  }

  state.modeHistory.push(mode);
  if (state.modeHistory.length > 20) state.modeHistory.shift();
}

// ---------------------------------------------------------------------------
// Utilidades de vocab e persistência
// ---------------------------------------------------------------------------

async function maybeExpandVocab(episodes) {
  const before = state.vocab.size;
  for (const ep of episodes) {
    if (ep.input) state.vocab.addTokens(tokenizeText(ep.input));
    if (ep.output) state.vocab.addTokens(tokenizeText(ep.output));
  }
  if (state.vocab.size > before) {
    console.log(`[VOCAB] Cresceu: ${before} → ${state.vocab.size}`);
    await maybeRebuildForVocabGrowth();
  }
}

async function maybeRebuildForVocabGrowth() {
  const curr = state.vocab.size;
  const embSize = state.model.getStats().embeddingVocabSize;
  if (curr > embSize) {
    console.log(`[VOCAB] Expandindo embedding: ${embSize} → ${curr}`);
    await state.model.expandVocabulary(curr);
    state.model.vocab = state.vocab;
  }
}

// [FIX-1] CORREÇÃO CRÍTICA: use .set com merge em vez de .update
async function markEpisodesAsQueued(episodes) {
  if (!state.db || !episodes.length) return;
  try {
    const batch = state.db.batch();
    for (const ep of episodes) {
      if (ep.id) {
        // Usar .set({...}, {merge: true}) para criar se não existir
        batch.set(
          state.db.collection('user_episodes').doc(ep.id),
          {
            queued_for_consolidation: true,
            queued_at: new Date().toISOString(),
          },
          { merge: true }
        );
      }
    }
    await batch.commit();
    console.log(`[EPISODES] Marcados ${episodes.length} episódios para consolidação`);
  } catch (err) {
    console.error('[EPISODES] Erro ao marcar episódios:', err.message);
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
    fs.writeFileSync(path.join(CONFIG.modelDir, 'akie_vocab.json'), JSON.stringify(vocab.toJSON(), null, 2));
  } catch (e) {
    try {
      if (state.db) {
        await state.db.collection('akie_worker_status').doc('vocabulary').set(vocab.toJSON(), { merge: true });
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
      updated_at: new Date().toISOString(),
      status: 'running',
      cycle: state.cycle,
      idle_cycles: state.idleCycles,
      last_mode: mode,
      mode_history: state.modeHistory,
      total_pairs: state.totalPairsTrained,
      metrics: state.metrics,
      model_stats: state.model.getStats(),
      generation_stats: state.generationStats,
      model_version: '2.1',  // [FIX] versão corrigida
    }, { merge: true });
  } catch (e) { /* não crítico */ }
}

// ---------------------------------------------------------------------------
// Utilitários
// ---------------------------------------------------------------------------

function shuffleArray(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// ---------------------------------------------------------------------------
// Watchdog
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

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
    if (_saveLock) console.warn('[WORKER] Timeout aguardando saveLock.');

    await safeSave();

    try {
      if (state.db) {
        await state.db.collection('akie_worker_status').doc('current').set({
          status: 'stopped',
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
