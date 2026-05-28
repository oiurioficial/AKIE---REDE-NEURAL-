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
 *
 *  NOVO v3:
 *    - Integrado akie_behavior.js: pré-processamento de intenção
 *    - Fallback inteligente: refino sem modelo, social/analyze com modelo
 *    - Contexto dinâmico baseado no modo detectado
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

// ── NOVO: Comportamento padrão do AKIE ──────────────────────────
const {
  processBehavior,
  getDefaultContext,
} = require('./akie_behavior');

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
  // NOVO: limites de geração por modo
  generationLimits: {
    social:  { maxTokens: 60,  temperature: 0.7 },
    analyze: { maxTokens: 150, temperature: 0.5 },
    code:    { maxTokens: 500, temperature: 0.3 },
    refino:  { maxTokens: 100, temperature: 0.4 },
  },
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
  // Flag explícita: modelo foi carregado do binário e está pronto para inferência
  modelLoadedFromBinary: false,
  metrics: {
    lastLoss:        null,
    lastAccuracy:    null,
    trainCycles:     0,
  },
  // Contadores de geração por modo
  generationStats: {
    social:  0,
    analyze: 0,
    code:    0,
    refino:  0,
    direct:  0,
  },
};

// ---------------------------------------------------------------------------
// Mutexes de exclusão mútua
// ---------------------------------------------------------------------------

// Impede que dois ciclos executem model.fit() ao mesmo tempo
// (ex: CONSOLIDATION ~75s iniciada quando o scheduler de 60s dispara novamente)
let _trainLock = false;

// Impede save concorrente entre ciclo periódico e SIGTERM
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
        generation_stats: state.generationStats,
      }));
      return;
    }

    // ── POST /generate ─────────────────────────────────────────
    // INFERÊNCIA PURA — NÃO executa treino, NÃO atualiza pesos.
    // Body: { prompt: string, max_tokens?: number, temperature?: number }
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

          // ── Garantir modelo carregado do binário (lazy load) ──
          // Não recria modelo vazio — apenas carrega se ainda não estiver em memória
          if (!state.model || !state.modelLoadedFromBinary) {
            console.log('[GENERATE] Modelo não em memória — carregando do binário...');
            const loaded = await tryLoadModelFromBinary();
            if (!loaded) {
              res.writeHead(503, headers);
              res.end(JSON.stringify({
                error: 'Modelo não disponível. weights.bin ausente ou corrompido.',
                binary_path: path.join(CONFIG.modelDir, 'weights.bin'),
              }));
              return;
            }
          }

          // Verificação de prontidão para inferência
          if (!state.model.ready) {
            res.writeHead(503, headers);
            res.end(JSON.stringify({ error: 'Modelo carregado mas não pronto para inferência.' }));
            return;
          }

          // ── Processar intenção (classificação apenas, sem treino) ──
          const behavior = processBehavior(prompt, {
            language: 'pt',
            refineryAvailable: false,
          });

          console.log(`[GENERATE] Inferência | Modo: ${behavior.mode} | Input: "${prompt.substring(0, 80)}${prompt.length > 80 ? '...' : ''}"`);

          // ── Output direto do behavior engine (sem modelo neural) ──
          if (behavior.output) {
            state.generationStats.direct++;
            res.writeHead(200, headers);
            res.end(JSON.stringify({
              ok:        true,
              prompt:    prompt,
              generated: behavior.output,
              mode:      behavior.mode,
              source:    'behavior_direct',
            }));
            return;
          }

          // ── INFERÊNCIA com modelo carregado do weights.bin ────
          // CRÍTICO: apenas model.generate() — zero trainBatch(), zero fit()
          const modeLimits = CONFIG.generationLimits[behavior.mode] || CONFIG.generationLimits.social;
          const finalMaxTokens   = max_tokens  || modeLimits.maxTokens;
          const finalTemperature = temperature || modeLimits.temperature;

          const enrichedPrompt = behavior.context
            ? `${behavior.context}\n\n[ENTRADA DO USUÁRIO]\n${behavior.input}`
            : behavior.input;

          let generated;
          try {
            generated = state.model.generate(enrichedPrompt, finalMaxTokens, finalTemperature);
          } catch (genErr) {
            console.error('[GENERATE] Falha na geração:', genErr.message);
            res.writeHead(500, headers);
            res.end(JSON.stringify({ error: `Falha na geração: ${genErr.message}` }));
            return;
          }

          // Atualizar contadores de uso (não de treino)
          state.generationStats[behavior.mode] = (state.generationStats[behavior.mode] || 0) + 1;

          // Persistir episódio para treino FUTURO (fire-and-forget)
          // Treino acontece apenas em CONSOLIDATION/SELF_PLAY, nunca aqui
          saveGenerationEpisode(prompt, generated, behavior.mode).catch(() => {});

          res.writeHead(200, headers);
          res.end(JSON.stringify({
            ok:            true,
            prompt:        prompt,
            generated:     generated || '',
            mode:          behavior.mode,
            steps:         state.model.trainSteps,
            source:        'model_inference',
            binary_loaded: state.modelLoadedFromBinary,
          }));

        } catch (err) {
          console.error('[GENERATE] Erro:', err.message);
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
    console.log(`[HTTP] Behavior engine: ATIVO (akie_behavior.js)`);
  });

  return server;
}

// ---------------------------------------------------------------------------
// Helpers de modelo e episódios
// ---------------------------------------------------------------------------

/**
 * Carrega o modelo do binário (model.json + weights.bin) sem recriar do zero.
 * Retorna true se carregado com sucesso, false caso contrário.
 * Preserva o estado entre chamadas — se já estiver carregado, retorna imediatamente.
 */
async function tryLoadModelFromBinary() {
  // Já carregado — não recarregar desnecessariamente
  if (state.model && state.modelLoadedFromBinary && state.model.ready) {
    return true;
  }

  const modelJsonPath  = path.join(CONFIG.modelDir, 'model.json');
  const weightsBinPath = path.join(CONFIG.modelDir, 'weights.bin');

  if (!fs.existsSync(modelJsonPath) || !fs.existsSync(weightsBinPath)) {
    console.warn('[LOAD] Arquivos binários não encontrados:', { modelJsonPath, weightsBinPath });
    return false;
  }

  try {
    // Garantir vocab carregado antes do modelo
    if (!state.vocab) {
      state.vocab = await loadOrCreateVocab();
    }

    // Instanciar sem build() — load() reconstruirá com os pesos do binário
    if (!state.model) {
      state.model = new AKIEModel(state.vocab);
    }

    const loaded = await state.model.load(CONFIG.modelDir);
    if (loaded) {
      state.modelLoadedFromBinary = true;
      console.log('[LOAD] Modelo carregado do binário com sucesso.');
      return true;
    }

    console.warn('[LOAD] model.load() retornou false — binário pode estar corrompido.');
    return false;

  } catch (err) {
    console.error('[LOAD] Erro ao carregar binário:', err.message);
    return false;
  }
}

/**
 * Marca episódios como enfileirados para consolidação futura.
 * Modo INTERACTIVE não treina — apenas registra para o próximo ciclo de CONSOLIDATION.
 */
async function markEpisodesAsQueued(episodes) {
  if (!state.db || !episodes.length) return;
  try {
    const batch = state.db.batch();
    for (const ep of episodes) {
      if (ep._docId) {
        const ref = state.db.collection('nexus_episodes').doc(ep._docId);
        batch.update(ref, {
          processed:    false,  // mantém disponível para CONSOLIDATION
          queued_at:    new Date().toISOString(),
          queued_mode:  'CONSOLIDATION',
        });
      }
    }
    await batch.commit();
  } catch (e) {
    // não crítico
  }
}

// Salva geração como episódio para treino futuro
// Inclui o modo detectado no episódio
async function saveGenerationEpisode(prompt, generated, mode) {
  if (!state.db || !generated) return;
  try {
    await state.db.collection('nexus_episodes').add({
      input:        prompt,
      output:       generated,
      layer:        `akie_${mode || 'generation'}`,
      feedback:     null,
      processed:    false,
      created_at:   new Date().toISOString(),
    });
  } catch (e) {
    // fire-and-forget — não crítico
  }
}

// ---------------------------------------------------------------------------
// Inicialização
// ---------------------------------------------------------------------------

async function init() {
  console.log('════════════════════════════════════════');
  console.log('  AKIE Training Worker v3 — iniciando');
  console.log('  Behavior Engine: akie_behavior.js');
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
    await saveVocab(state.vocab);
    console.log('[INIT] Recriando modelo após bootstrap...');
    state.model = new AKIEModel(state.vocab);
    state.model.build();
    await state.model.save(CONFIG.modelDir);
    // Após bootstrap o modelo é válido para inferência
    state.modelLoadedFromBinary = true;
  }

  console.log(`[INIT] Vocabulário: ${state.vocab.size} tokens`);

  // Modelo — carregar do binário (model.json + weights.bin)
  state.model = new AKIEModel(state.vocab);
  const loaded = await state.model.load(CONFIG.modelDir);

  if (!loaded) {
    console.log('[INIT] Nenhum modelo encontrado. Construindo novo...');
    await primeVocabulary(state.vocab, state.db);
    state.model = new AKIEModel(state.vocab);
    state.model.build();
    await state.model.save(CONFIG.modelDir);
    // Modelo recém-construído também é válido para inferência
    state.modelLoadedFromBinary = true;
    console.log('[INIT] Novo modelo construído e salvo em binário.');
  } else {
    // Modelo carregado do binário (model.json + weights.bin)
    state.modelLoadedFromBinary = true;
    console.log('[INIT] Modelo carregado do binário. Pronto para inferência.');
    await maybeRebuildForVocabGrowth();
  }

  await state.db.collection('akie_worker_status').doc('current').set({
    started_at:  new Date().toISOString(),
    status:      'running',
    model_stats: state.model.getStats(),
    behavior_engine: 'akie_behavior.js',
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
    // IMPORTANTE: INTERACTIVE NÃO treina automaticamente.
    // Episódios são expandidos no vocab e enfileirados para CONSOLIDATION futura.
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
  // Guard: se outro ciclo ainda está treinando, ignora este
  if (_trainLock) {
    const tag = ctx.tag || '';
    console.log(`${tag} trainLock ativo (ciclo anterior ainda treinando) — ${mode} ignorado`);
    return;
  }

  const { tag = '', episodes = [] } = ctx;
  const t0 = Date.now();
  let pairs = [];
  let desc  = '';

  console.log(`${tag} Modo: ${mode}`);

  switch (mode) {
    case MODE.INTERACTIVE: {
      // MODO INTERACTIVE: NÃO treina. Apenas expande o vocab com novos tokens
      // e marca episódios como processados para reaproveitamento em CONSOLIDATION.
      // O treino só ocorre em modos explícitos: CONSOLIDATION, EXPANSION, SELF_PLAY.
      await maybeExpandVocab(episodes);
      await markEpisodesAsQueued(episodes);
      console.log(`${tag} INTERACTIVE: ${episodes.length} episódios enfileirados para consolidação futura`);
      state.modeHistory.push(mode);
      if (state.modeHistory.length > 20) state.modeHistory.shift();
      return; // sai sem chamar trainBatch
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
      pairs = await fetchWebPatterns(state.db, state.vocab, 1);
      desc  = `web crawl → ${pairs.length} pares`;
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

    _trainLock = true;
    let result;
    try {
      result = await state.model.trainBatch(pairs, 3);
    } finally {
      _trainLock = false;
    }

    state.metrics.lastLoss     = result.loss;
    state.metrics.lastAccuracy = result.accuracy;
    state.metrics.trainCycles++;
    state.totalPairsTrained += pairs.length;

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const loss    = result.loss    != null ? result.loss.toFixed(4) : 'n/a';
    const acc     = result.accuracy != null ? (result.accuracy * 100).toFixed(1) + '%' : 'n/a';
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
      generation_stats: state.generationStats,
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
    console.log('\n[WORKER] SIGTERM — aguardando conclusão do ciclo...');

    // Espera fit() E save periódico terminarem antes de prosseguir.
    // Matar o processo no meio de um model.save() gera shapes [,] no model.json.
    const deadline = Date.now() + 90_000;
    while ((_trainLock || _saveLock) && Date.now() < deadline) {
      await new Promise(r => setTimeout(r, 500));
    }
    if (_trainLock) console.warn('[WORKER] Timeout aguardando trainLock.');
    if (_saveLock)  console.warn('[WORKER] Timeout aguardando saveLock.');

    await safeSave();

    try {
      await state.db.collection('akie_worker_status').doc('current').update({
        status: 'stopped', stopped_at: new Date().toISOString(),
      });
    } catch { /* ignora */ }

    process.exit(0);
  });
}

main().catch(err => {
  console.error('[WORKER] Erro fatal:', err);
  process.exit(1);
});
