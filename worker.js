/**
 * worker.js  —  AKIE Training Worker
 *
 * Roda 24h no Railway. Nunca para.
 * Quatro modos de treino em cascata — sempre tem algo para aprender.
 *
 *  ┌─────────────────────────────────────────────────────────────┐
 *  │  MODO 1 — INTERACTIVE  (usuário interagiu)                  │
 *  │  MODO 2 — CONSOLIDATION (replay do grafo semântico)         │
 *  │  MODO 3 — EXPANSION    (web crawl, lacunas do grafo)        │
 *  │  MODO 4 — SELF_PLAY    (modelo gera e treina nos melhores)  │
 *  └─────────────────────────────────────────────────────────────┘
 *
 *  Intervalo base: 60s
 *  A cada ciclo, o scheduler decide qual modo executar.
 *  Modo 1 tem prioridade. Modos 2-4 rodam em rodízio quando ocioso.
 */

require('dotenv').config();
const admin  = require('firebase-admin');
const path   = require('path');
const fs     = require('fs');

// Importar módulos AKIE
const { AKIEModel }              = require('./_nexus_neural');
const { Vocabulary }             = require('./_akie_vocab');
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
  intervalMs:       60_000,   // ciclo principal: 60s
  modelDir:         process.env.MODEL_DIR || '/data/akie_model',
  saveEveryN:       5,        // salvar modelo a cada N ciclos de treino
  logEveryN:        1,        // logar métricas a cada N ciclos
  // Quantos ciclos sem dados de usuário antes de ativar cada modo
  consolidationAfter: 1,      // grafo: quase sempre
  expansionAfter:     3,      // web: a cada 3 ciclos ociosos
  selfPlayAfter:      2,      // self-play: a cada 2 ciclos ociosos
};

// Modos possíveis
const MODE = {
  INTERACTIVE:    'INTERACTIVE',
  CONSOLIDATION:  'CONSOLIDATION',
  EXPANSION:      'EXPANSION',
  SELF_PLAY:      'SELF_PLAY',
  IDLE:           'IDLE',
};

// ---------------------------------------------------------------------------
// Estado do worker
// ---------------------------------------------------------------------------

const state = {
  cycle:          0,
  idleCycles:     0,
  lastSaveCycle:  0,
  totalPairsTrained: 0,
  model:          null,
  vocab:          null,
  db:             null,
  modeHistory:    [],   // últimos 20 modos executados
  metrics: {
    lastLoss:     null,
    lastAccuracy: null,
    trainCycles:  0,
  },
};

// ---------------------------------------------------------------------------
// Inicialização
// ---------------------------------------------------------------------------

async function init() {
  console.log('════════════════════════════════════════');
  console.log('  AKIE Training Worker — iniciando');
  console.log(`  ${new Date().toISOString()}`);
  console.log('════════════════════════════════════════');

  // Firebase
  const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
  if (!admin.apps.length) {
    admin.initializeApp({
      credential: admin.credential.cert(serviceAccount),
    });
  }
  state.db = admin.firestore();
  console.log('[INIT] Firebase conectado.');

  // Carregar ou criar vocabulário
  state.vocab = await loadOrCreateVocab();
  console.log(`[INIT] Vocabulário: ${state.vocab.size} tokens`);

  // Carregar ou criar modelo
  state.model = new AKIEModel(state.vocab);
  const loaded = await state.model.load(CONFIG.modelDir);

  if (!loaded) {
    console.log('[INIT] Nenhum modelo encontrado. Construindo novo...');
    // Pre-alimentar vocabulário com dados do grafo antes de construir
    await primeVocabulary(state.vocab, state.db);
    state.model = new AKIEModel(state.vocab);
    state.model.build();
    await state.model.save(CONFIG.modelDir);
  }

  // Registrar início no Firestore
  await state.db.collection('akie_worker_status').doc('current').set({
    started_at: new Date().toISOString(),
    status: 'running',
    model_stats: state.model.getStats(),
  });

  console.log('[INIT] Worker pronto. Iniciando loop contínuo...\n');
}

/**
 * Alimenta o vocabulário com tokens do grafo antes do primeiro build.
 * Garante que o modelo nasce já sabendo os tokens existentes.
 */
async function primeVocabulary(vocab, db) {
  console.log('[INIT] Carregando vocabulário base do grafo...');
  const snap = await db.collection('nexus_nodes').limit(500).get();
  const sentences = [];

  snap.forEach(doc => {
    const d = doc.data();
    if (d.label) sentences.push(d.label);
    if (d.id) sentences.push(d.id.replace(/_/g, ' '));
    (d.relations || []).forEach(r => {
      if (r.target) sentences.push(r.target.replace(/_/g, ' '));
    });
  });

  const { tokenizeText } = require('./_akie_vocab');
  for (const s of sentences) {
    vocab.addTokens(tokenizeText(s));
  }
  await saveVocab(vocab);
  console.log(`[INIT] Vocabulário primário: ${vocab.size} tokens`);
}

// ---------------------------------------------------------------------------
// Scheduler — decide qual modo executar
// ---------------------------------------------------------------------------

async function scheduler() {
  state.cycle++;
  const cycleTag = `[C${String(state.cycle).padStart(4, '0')}]`;

  try {
    // ── Verificar dados de usuário (MODO 1 — prioridade máxima)
    const episodes = await fetchUserEpisodes(state.db, 100);

    if (episodes.length > 0) {
      state.idleCycles = 0;
      await runMode(MODE.INTERACTIVE, { episodes, cycleTag });
      return;
    }

    // ── Sem dados de usuário — modos autônomos em rodízio
    state.idleCycles++;

    // Ciclo ocioso: decidir modo por módulo do contador
    const idleMod = state.idleCycles;

    if (idleMod % CONFIG.selfPlayAfter === 0 && state.model.trainSteps > 50) {
      // Self-play: só ativa depois de 50 steps (modelo precisa ter alguma base)
      await runMode(MODE.SELF_PLAY, { cycleTag });

    } else if (idleMod % CONFIG.expansionAfter === 0) {
      // Web crawl
      await runMode(MODE.EXPANSION, { cycleTag });

    } else {
      // Consolidação do grafo (mais frequente)
      await runMode(MODE.CONSOLIDATION, { cycleTag });
    }

  } catch (err) {
    console.error(`${state.cycle > 0 ? `[C${String(state.cycle).padStart(4, '0')}]` : '[INIT]'} ERRO no ciclo:`, err.message);
    // Não para o worker — erro em um ciclo não derruba os outros
  }
}

// ---------------------------------------------------------------------------
// Executor de modo
// ---------------------------------------------------------------------------

async function runMode(mode, ctx = {}) {
  const { cycleTag = '', episodes = [] } = ctx;
  const startMs = Date.now();
  let pairs = [];
  let dataDesc = '';

  console.log(`${cycleTag} Modo: ${mode}`);

  switch (mode) {

    // ── MODO 1: Dados reais de usuário
    case MODE.INTERACTIVE: {
      pairs = episodesToPairs(episodes, state.vocab);
      dataDesc = `${episodes.length} episódios → ${pairs.length} pares`;

      // Expandir vocabulário se novos tokens apareceram
      await maybeExpandVocab(episodes);
      break;
    }

    // ── MODO 2: Grafo semântico → frases → treino
    case MODE.CONSOLIDATION: {
      const sentences = await fetchGraphSentences(state.db, 300);
      if (sentences.length === 0) {
        console.log(`${cycleTag} Grafo vazio — pulando consolidação`);
        return;
      }
      pairs = graphSentencesToPairs(sentences, state.vocab);
      dataDesc = `${sentences.length} frases do grafo → ${pairs.length} pares`;
      break;
    }

    // ── MODO 3: Web crawl
    case MODE.EXPANSION: {
      pairs = await fetchWebPatterns(state.db, state.vocab, 3);
      dataDesc = `web crawl → ${pairs.length} pares`;

      if (pairs.length > 0) {
        // Vocab pode ter crescido — salvar e verificar expansão
        await saveVocab(state.vocab);
        await maybeRebuildForVocabGrowth();
      }
      break;
    }

    // ── MODO 4: Self-play
    case MODE.SELF_PLAY: {
      pairs = await selfPlayPairs(state.model, state.db, state.vocab, 15);
      dataDesc = `self-play → ${pairs.length} pares`;
      break;
    }
  }

  // ── Treinar se tiver dados
  if (pairs.length > 0) {
    // Embaralhar pares
    shuffleArray(pairs);

    const result = await state.model.trainBatch(pairs, 3);
    state.metrics.lastLoss     = result.loss;
    state.metrics.lastAccuracy = result.accuracy;
    state.metrics.trainCycles++;
    state.totalPairsTrained += pairs.length;

    const elapsed = ((Date.now() - startMs) / 1000).toFixed(1);
    const lossStr = result.loss !== null ? result.loss.toFixed(4) : 'n/a';
    const accStr  = result.accuracy !== null ? (result.accuracy * 100).toFixed(1) + '%' : 'n/a';

    console.log(`${cycleTag} ✓ ${dataDesc} | loss=${lossStr} acc=${accStr} | ${elapsed}s`);

    // Salvar modelo periodicamente
    if (state.cycle - state.lastSaveCycle >= CONFIG.saveEveryN) {
      await state.model.save(CONFIG.modelDir);
      await saveVocab(state.vocab);
      state.lastSaveCycle = state.cycle;
      await reportStatus(mode);
    }

  } else {
    console.log(`${cycleTag} Nenhum dado disponível para ${mode}`);
  }

  // Registrar modo no histórico
  state.modeHistory.push(mode);
  if (state.modeHistory.length > 20) state.modeHistory.shift();
}

// ---------------------------------------------------------------------------
// Expansão de vocabulário
// ---------------------------------------------------------------------------

async function maybeExpandVocab(episodes) {
  const { tokenizeText } = require('./_akie_vocab');
  const sizeBefore = state.vocab.size;

  for (const ep of episodes) {
    if (ep.input)  state.vocab.addTokens(tokenizeText(ep.input));
    if (ep.output) state.vocab.addTokens(tokenizeText(ep.output));
  }

  if (state.vocab.size > sizeBefore) {
    console.log(`[VOCAB] Cresceu: ${sizeBefore} → ${state.vocab.size} tokens`);
    await maybeRebuildForVocabGrowth();
  }
}

async function maybeRebuildForVocabGrowth() {
  const currentSize = state.vocab.size;
  const modelSize   = state.model.getStats().vocabSize;

  if (currentSize > modelSize) {
    await state.model.expandVocabulary(currentSize);
    // Atualizar referência do vocab no modelo
    state.model.vocab = state.vocab;
  }
}

// ---------------------------------------------------------------------------
// Persistência de vocabulário e status
// ---------------------------------------------------------------------------

async function loadOrCreateVocab() {
  const vocabPath = path.join(CONFIG.modelDir, 'akie_vocab.json');

  try {
    if (fs.existsSync(vocabPath)) {
      const data = JSON.parse(fs.readFileSync(vocabPath, 'utf8'));
      return Vocabulary.fromJSON(data);
    }
  } catch (e) {
    console.log('[VOCAB] Falha ao carregar do disco, tentando Firestore...');
  }

  // Tentar Firestore como fallback
  try {
    const doc = await state.db.collection('akie_worker_status').doc('vocabulary').get();
    if (doc.exists) {
      console.log('[VOCAB] Vocabulário carregado do Firestore');
      return Vocabulary.fromJSON(doc.data());
    }
  } catch (e) { /* ignora */ }

  console.log('[VOCAB] Criando vocabulário novo');
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
    // Fallback: Firestore
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
      updated_at:    new Date().toISOString(),
      status:        'running',
      cycle:         state.cycle,
      idle_cycles:   state.idleCycles,
      last_mode:     mode,
      mode_history:  state.modeHistory,
      total_pairs:   state.totalPairsTrained,
      metrics:       state.metrics,
      model_stats:   state.model.getStats(),
    }, { merge: true });
  } catch (e) {
    // Não crítico — só logging
  }
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
// Watchdog — reinicia o loop se travar por > 5 minutos
// ---------------------------------------------------------------------------

let lastHeartbeat = Date.now();

function startWatchdog() {
  setInterval(() => {
    const elapsed = Date.now() - lastHeartbeat;
    if (elapsed > 5 * 60 * 1000) {
      console.error('[WATCHDOG] Loop travado há ' + Math.round(elapsed/1000) + 's. Forçando ciclo...');
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
  startWatchdog();

  // Primeiro ciclo imediato (não espera 60s)
  await scheduler();
  lastHeartbeat = Date.now();

  // Loop principal: a cada 60 segundos
  setInterval(async () => {
    await scheduler();
    lastHeartbeat = Date.now();
  }, CONFIG.intervalMs);

  // Graceful shutdown
  process.on('SIGTERM', async () => {
    console.log('\n[WORKER] SIGTERM recebido. Salvando estado...');
    try {
      await state.model.save(CONFIG.modelDir);
      await saveVocab(state.vocab);
      await state.db.collection('akie_worker_status').doc('current').update({
        status: 'stopped',
        stopped_at: new Date().toISOString(),
      });
    } catch (e) { /* ignora */ }
    process.exit(0);
  });
}

main().catch(err => {
  console.error('[WORKER] Erro fatal na inicialização:', err);
  process.exit(1);
});
