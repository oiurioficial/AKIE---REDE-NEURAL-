/**
 * worker.js  —  AKIE Training Worker + HTTP Endpoint
 *
 * Roda 24h no Railway. Nunca para.
 * Expõe endpoint HTTP para o NEXUS chamar geração em tempo real.
 *
 *  ┌─────────────────────────────────────────────────────────────┐
 *  │  MODO 1 — INTERACTIVE  (usuário interagiu)                  │
 *  │  MODO 2 — CONSOLIDATION (replay do grafo semântico)         │
 *  │  MODO 3 — EXPANSION    (web crawl, lacunas do grafo)        │
 *  │  MODO 4 — SELF_PLAY    (modelo gera e treina nos melhores)  │
 *  └─────────────────────────────────────────────────────────────┘
 *
 *  CORREÇÕES v2:
 *    - Coleção: nexus_nodes → nexus_graph (alinhado com o motor)
 *    - Campo:   confirmed   → confidence === 'confirmed'
 *    - Adicionado: servidor HTTP na porta 3000
 *    - Adicionado: endpoint POST /generate para o NEXUS
 *    - Adicionado: endpoint GET  /status para monitoramento
 */

require('dotenv').config();
const admin  = require('firebase-admin');
const http   = require('http');
const path   = require('path');
const fs     = require('fs');

const { AKIEModel }              = require('./_nexus_neural');
const { Vocabulary, tokenizeText } = require('./_akie_vocab');
const { runBootstrap }             = require('./_akie_bootstrap');
const {
  fetchUserEpisodes,
  episodesToPairs,
  fetchGraphSentences,
  graphSentencesToPairs,
  fetchWebPatterns,
  selfPlayPairs,
} = require('./_akie_dataset');

// ---------------------------------------------------------------------------
// Configuração
// ---------------------------------------------------------------------------

const CONFIG = {
  intervalMs:          60_000,
  modelDir:            process.env.MODEL_DIR || '/data/akie_model',
  httpPort:            parseInt(process.env.PORT || '3000', 10),
  saveEveryN:          5,
  consolidationAfter:  1,
  expansionAfter:      3,
  selfPlayAfter:       2,
};

const MODE = {
  INTERACTIVE:   'INTERACTIVE',
  CONSOLIDATION: 'CONSOLIDATION',
  EXPANSION:     'EXPANSION',
  SELF_PLAY:     'SELF_PLAY',
};

// ---------------------------------------------------------------------------
// Estado global
// ---------------------------------------------------------------------------

const state = {
  cycle:             0,
  idleCycles:        0,
  lastSaveCycle:     0,
  totalPairsTrained: 0,
  model:             null,
  vocab:             null,
  db:                null,
  modeHistory:       [],
  metrics: {
    lastLoss:        null,
    lastAccuracy:    null,
    trainCycles:     0,
  },
};

// ---------------------------------------------------------------------------
// Servidor HTTP — endpoint para o NEXUS chamar geração
// ---------------------------------------------------------------------------

function startHttpServer() {
  const server = http.createServer(async (req, res) => {
    const headers = {
      'Content-Type':                'application/json',
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
        ok:          true,
        cycle:       state.cycle,
        idle_cycles: state.idleCycles,
        metrics:     state.metrics,
        model:       stats,
        mode_history: state.modeHistory.slice(-5),
      }));
      return;
    }

    // ── POST /generate ─────────────────────────────────────────
    // Chamado pelo NEXUS quando quer geração neural
    // Body: { prompt: string, max_tokens?: number, temperature?: number }
    if (req.method === 'POST' && req.url === '/generate') {
      let body = '';
      req.on('data', chunk => { body += chunk; });
      req.on('end', () => {
        try {
          const { prompt, max_tokens, temperature } = JSON.parse(body || '{}');

          if (!prompt) {
            res.writeHead(400, headers);
            res.end(JSON.stringify({ error: 'Campo prompt ausente.' }));
            return;
          }

          if (!state.model || !state.model.ready) {
            res.writeHead(503, headers);
            res.end(JSON.stringify({ error: 'Modelo não está pronto ainda.' }));
            return;
          }

          const generated = state.model.generate(
            prompt,
            max_tokens  || 30,
            temperature || 0.7
          );

          // Marcar episódio para treino futuro (fire-and-forget)
          saveGenerationEpisode(prompt, generated).catch(() => {});

          res.writeHead(200, headers);
          res.end(JSON.stringify({
            ok:        true,
            prompt:    prompt,
            generated: generated || '',
            steps:     state.model.trainSteps,
          }));

        } catch (err) {
          res.writeHead(500, headers);
          res.end(JSON.stringify({ error: err.message }));
        }
      });
      return;
    }

    // ── POST /feedback ─────────────────────────────────────────
    // O NEXUS envia feedback sobre uma geração (positivo/negativo)
    // Body: { episode_id: string, feedback: 'positive' | 'negative' }
    if (req.method === 'POST' && req.url === '/feedback') {
      let body = '';
      req.on('data', chunk => { body += chunk; });
      req.on('end', async () => {
        try {
          const { episode_id, feedback } = JSON.parse(body || '{}');
          if (episode_id && feedback && state.db) {
            await state.db.collection('nexus_episodes')
              .doc(episode_id)
              .update({ feedback, feedback_at: new Date().toISOString() });
          }
          res.writeHead(200, headers);
          res.end(JSON.stringify({ ok: true }));
        } catch (err) {
          res.writeHead(500, headers);
          res.end(JSON.stringify({ error: err.message }));
        }
      });
      return;
    }

    res.writeHead(404, headers);
    res.end(JSON.stringify({ error: 'Rota não encontrada.' }));
  });

  server.listen(CONFIG.httpPort, () => {
    console.log(`[HTTP] Servidor rodando na porta ${CONFIG.httpPort}`);
    console.log(`[HTTP] Endpoints: GET /status | POST /generate | POST /feedback`);
  });

  return server;
}

// Salva geração como episódio para treino futuro
async function saveGenerationEpisode(prompt, generated) {
  if (!state.db || !generated) return;
  await state.db.collection('nexus_episodes').add({
    input:        prompt,
    output:       generated,
    layer:        'akie_generation',
    feedback:     null,
    processed:    false,
    created_at:   new Date().toISOString(),
  });
}

// ---------------------------------------------------------------------------
// Inicialização
// ---------------------------------------------------------------------------

async function init() {
  console.log('════════════════════════════════════════');
  console.log('  AKIE Training Worker v2 — iniciando');
  console.log(`  ${new Date().toISOString()}`);
  console.log('════════════════════════════════════════');

  // Firebase
  const raw = process.env.FIREBASE_SERVICE_ACCOUNT;
  if (!raw) throw new Error('FIREBASE_SERVICE_ACCOUNT não definido.');
  const serviceAccount = JSON.parse(raw);

  if (!admin.apps.length) {
    admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
  }
  state.db = admin.firestore();
  console.log('[INIT] Firebase conectado. Coleção ativa: nexus_graph');

  // Vocabulário
  state.vocab = await loadOrCreateVocab();

  // Bootstrap — garante que o sistema nunca inicie com grafo vazio
  // Idempotente: só roda uma vez na vida do Firestore
  const bootstrapped = await runBootstrap(state.db, state.vocab);
  if (bootstrapped) {
    // Salvar vocab expandido pelo bootstrap antes de construir o modelo
    await saveVocab(state.vocab);
  }

  console.log(`[INIT] Vocabulário: ${state.vocab.size} tokens`);

  // Modelo
  state.model = new AKIEModel(state.vocab);
  const loaded = await state.model.load(CONFIG.modelDir);

  if (!loaded) {
    console.log('[INIT] Nenhum modelo encontrado. Construindo novo...');
    await primeVocabulary(state.vocab, state.db);
    // Re-executar bootstrap de vocab caso primeVocabulary tenha adicionado tokens
    state.model = new AKIEModel(state.vocab);
    state.model.build();
    await state.model.save(CONFIG.modelDir);
  } else {
    // Modelo carregado — verificar se vocab cresceu desde o último save
    await maybeRebuildForVocabGrowth();
  }

  await state.db.collection('akie_worker_status').doc('current').set({
    started_at:  new Date().toISOString(),
    status:      'running',
    model_stats: state.model.getStats(),
  });

  console.log('[INIT] Worker pronto.\n');
}

/**
 * Alimenta vocabulário com tokens do grafo NEXUS antes do primeiro build.
 * CORRIGIDO: usa nexus_graph (não nexus_nodes).
 */
async function primeVocabulary(vocab, db) {
  console.log('[INIT] Carregando vocabulário base do grafo nexus_graph...');
  const snap = await db.collection('nexus_graph').limit(500).get();
  const sentences = [];

  snap.forEach(doc => {
    const d = doc.data();
    if (d.label) sentences.push(d.label);
    if (d.id)    sentences.push(d.id.replace(/_/g, ' '));
    (d.relations || []).forEach(r => {
      if (r.target) sentences.push(r.target.replace(/_/g, ' '));
      if (r.type)   sentences.push(r.type.replace(/_/g, ' '));
    });
    (d.contexts || []).forEach(c => sentences.push(c));
    (d.verbs    || []).forEach(v => sentences.push(v));
  });

  for (const s of sentences) {
    vocab.addTokens(tokenizeText(s));
  }
  await saveVocab(vocab);
  console.log(`[INIT] Vocabulário primário: ${vocab.size} tokens`);
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

async function scheduler() {
  state.cycle++;
  const tag = `[C${String(state.cycle).padStart(4, '0')}]`;

  try {
    // Prioridade 1: episódios reais de usuário
    const episodes = await fetchUserEpisodes(state.db, 100);
    if (episodes.length > 0) {
      state.idleCycles = 0;
      await runMode(MODE.INTERACTIVE, { episodes, tag });
      return;
    }

    state.idleCycles++;
    const n = state.idleCycles;

    if (n % CONFIG.selfPlayAfter === 0 && state.model.trainSteps > 50) {
      await runMode(MODE.SELF_PLAY, { tag });
    } else if (n % CONFIG.expansionAfter === 0) {
      await runMode(MODE.EXPANSION, { tag });
    } else {
      await runMode(MODE.CONSOLIDATION, { tag });
    }

  } catch (err) {
    console.error(`${tag} ERRO:`, err.message);
  }
}

// ---------------------------------------------------------------------------
// Executor de modo
// ---------------------------------------------------------------------------

async function runMode(mode, ctx = {}) {
  const { tag = '', episodes = [] } = ctx;
  const t0 = Date.now();
  let pairs = [];
  let desc  = '';

  console.log(`${tag} Modo: ${mode}`);

  switch (mode) {
    case MODE.INTERACTIVE: {
      pairs = episodesToPairs(episodes, state.vocab);
      desc  = `${episodes.length} episódios → ${pairs.length} pares`;
      await maybeExpandVocab(episodes);
      break;
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
      pairs = await fetchWebPatterns(state.db, state.vocab, 3);
      desc  = `web crawl → ${pairs.length} pares`;
      // Sempre verificar crescimento de vocab, mesmo sem pairs
      // O fetchWebPatterns adiciona tokens ao vocab independentemente de gerar pares
      if (state.vocab.size > vocabBefore) {
        await saveVocab(state.vocab);
        await maybeRebuildForVocabGrowth();
      }
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
    const result = await state.model.trainBatch(pairs, 3);
    state.metrics.lastLoss     = result.loss;
    state.metrics.lastAccuracy = result.accuracy;
    state.metrics.trainCycles++;
    state.totalPairsTrained += pairs.length;

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const loss    = result.loss    != null ? result.loss.toFixed(4) : 'n/a';
    const acc     = result.accuracy != null ? (result.accuracy * 100).toFixed(1) + '%' : 'n/a';
    console.log(`${tag} ✓ ${desc} | loss=${loss} acc=${acc} | ${elapsed}s`);

    if (state.cycle - state.lastSaveCycle >= CONFIG.saveEveryN) {
      await state.model.save(CONFIG.modelDir);
      await saveVocab(state.vocab);
      state.lastSaveCycle = state.cycle;
      await reportStatus(mode);
    }
  } else {
    console.log(`${tag} Nenhum dado disponível para ${mode}`);
  }

  state.modeHistory.push(mode);
  if (state.modeHistory.length > 20) state.modeHistory.shift();
}

// ---------------------------------------------------------------------------
// Vocabulário e persistência
// ---------------------------------------------------------------------------

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
  const curr        = state.vocab.size;
  const embSize     = state.model.getStats().embeddingVocabSize; // tamanho REAL da camada
  if (curr > embSize) {
    console.log(`[VOCAB] Expandindo camada de embedding: ${embSize} → ${curr}`);
    await state.model.expandVocabulary(curr);
    state.model.vocab = state.vocab;
  }
}

async function loadOrCreateVocab() {
  const p = path.join(CONFIG.modelDir, 'akie_vocab.json');
  try {
    if (fs.existsSync(p)) return Vocabulary.fromJSON(JSON.parse(fs.readFileSync(p, 'utf8')));
  } catch (e) { /* ignora */ }

  try {
    const doc = await state.db.collection('akie_worker_status').doc('vocabulary').get();
    if (doc.exists) return Vocabulary.fromJSON(doc.data());
  } catch (e) { /* ignora */ }

  return new Vocabulary();
}

async function saveVocab(vocab) {
  try {
    fs.mkdirSync(CONFIG.modelDir, { recursive: true });
    fs.writeFileSync(path.join(CONFIG.modelDir, 'akie_vocab.json'), JSON.stringify(vocab.toJSON(), null, 2));
  } catch (e) {
    try {
      await state.db.collection('akie_worker_status').doc('vocabulary').set(vocab.toJSON());
    } catch (e2) {
      console.error('[VOCAB] Falha ao salvar:', e2.message);
    }
  }
}

async function reportStatus(mode) {
  try {
    await state.db.collection('akie_worker_status').doc('current').set({
      updated_at:  new Date().toISOString(),
      status:      'running',
      cycle:       state.cycle,
      idle_cycles: state.idleCycles,
      last_mode:   mode,
      mode_history: state.modeHistory,
      total_pairs: state.totalPairsTrained,
      metrics:     state.metrics,
      model_stats: state.model.getStats(),
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
    console.log('\n[WORKER] SIGTERM — salvando...');
    try {
      await state.model.save(CONFIG.modelDir);
      await saveVocab(state.vocab);
      await state.db.collection('akie_worker_status').doc('current').update({
        status: 'stopped', stopped_at: new Date().toISOString(),
      });
    } catch (e) { /* ignora */ }
    process.exit(0);
  });
}

main().catch(err => {
  console.error('[WORKER] Erro fatal:', err);
  process.exit(1);
});
