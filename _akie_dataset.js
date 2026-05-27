/**
 * _akie_dataset.js
 *
 * Pipeline de dados para treino contínuo do AKIE.
 *
 * CORREÇÕES v2:
 *   - nexus_nodes → nexus_graph (alinhado com o motor NEXUS)
 *   - confirmed === true → confidence === 'confirmed'
 *   - Campo weight → usar confidence + relations[].weight
 *   - fetchGraphSentences usa nexus_graph corretamente
 *   - fetchWebPatterns usa nexus_graph com filtro por score baixo
 */

const { tokenizeText, makeTrainingPairs } = require('./_akie_vocab');

// ---------------------------------------------------------------------------
// FONTE 1 — Episódios reais de interação
// ---------------------------------------------------------------------------

/**
 * Busca episódios não processados do Firestore.
 * processed=false → ainda não foram usados para treino.
 */
async function fetchUserEpisodes(db, limit = 50) {
  const snap = await db.collection('nexus_episodes')
    .where('processed', '==', false)
    .orderBy('created_at', 'asc')
    .limit(limit)
    .get();

  if (snap.empty) return [];

  const episodes = [];
  const batch    = db.batch();

  snap.forEach(doc => {
    const d = doc.data();
    // Normaliza campos — o motor salva como input/response, o worker precisa input/output
    const input  = d.input  || d.user_message    || '';
    const output = d.response || d.output || d.system_message || '';

    if (input && output) {
      episodes.push({
        id:       doc.id,
        input,
        output,
        feedback: d.feedback || null,
        layer:    d.layer    || null,
      });
    }
    batch.update(doc.ref, { processed: true, processed_at: new Date().toISOString() });
  });

  await batch.commit();
  return episodes;
}

/**
 * Converte episódios em pares de treino.
 * Feedback positivo → peso 2x. Negativo → descartado.
 */
function episodesToPairs(episodes, vocab) {
  const allPairs = [];

  for (const ep of episodes) {
    if (ep.feedback === 'negative') continue;
    if (!ep.input || !ep.output)   continue;

    const fullText = `${ep.input} ${ep.output}`;
    const ids      = vocab.tokenize(fullText);
    const pairs    = makeTrainingPairs(ids);
    const weight   = ep.feedback === 'positive' ? 2 : 1;

    for (let i = 0; i < weight; i++) allPairs.push(...pairs);
  }

  return allPairs;
}

// ---------------------------------------------------------------------------
// FONTE 2 — Grafo semântico NEXUS → frases de treino
// CORRIGIDO: usa nexus_graph + confidence === 'confirmed'
// ---------------------------------------------------------------------------

/**
 * Busca nós confirmados do nexus_graph e converte em frases.
 *
 * Estrutura do nó em nexus_graph:
 *   { id, label, confidence, relations: [{type, target, weight}], contexts[], verbs[] }
 */
async function fetchGraphSentences(db, limit = 200) {
  const snap = await db.collection('nexus_graph')
    .where('confidence', '==', 'confirmed')
    .limit(limit)
    .get();

  const sentences = [];

  snap.forEach(doc => {
    const node = doc.data();
    const label = node.label || node.id.replace(/_/g, ' ');

    // Relações com peso significativo
    const rels = (node.relations || []).filter(r => r.weight >= 0.60);

    for (const rel of rels) {
      const target   = (rel.target || '').replace(/_/g, ' ');
      const sentence = relationToSentence(label, rel.type, target);
      if (sentence) sentences.push(sentence);
    }

    // Contextos como frases simples
    (node.contexts || []).forEach(ctx => {
      if (ctx && ctx.length > 2) {
        sentences.push(`${label} aparece em ${ctx.replace(/_/g, ' ')}`);
      }
    });

    // Verbos como frases simples
    (node.verbs || []).slice(0, 2).forEach(verb => {
      if (verb && verb.length > 3) {
        sentences.push(`${label} ${verb.replace(/_/g, ' ')}`);
      }
    });
  });

  return sentences;
}

/**
 * Converte tipo de relação NEXUS em frase natural PT-BR.
 * Cobre todos os tipos definidos em REL_TYPE_VERBS do _nexus_seed.js.
 */
function relationToSentence(source, relType, target) {
  const s = source.replace(/_/g, ' ');
  const t = target.replace(/_/g, ' ');
  if (!s || !t) return null;

  const templates = {
    'e_tipo':         [`${s} é um tipo de ${t}`, `${s} é ${t}`],
    'gera':           [`${s} gera ${t}`, `${s} produz ${t}`],
    'requer':         [`${s} requer ${t}`, `${s} precisa de ${t}`],
    'relacionado_a':  [`${s} está relacionado a ${t}`, `${s} aparece com ${t}`],
    'usa':            [`${s} usa ${t}`, `${s} aplica ${t}`],
    'implica':        [`${s} implica ${t}`, `${s} sugere ${t}`],
    'contrasta':      [`${s} contrasta com ${t}`],
    'parte_de':       [`${s} faz parte de ${t}`, `${s} é componente de ${t}`],
    'define':         [`${s} define ${t}`, `${s} determina ${t}`],
    'influencia':     [`${s} influencia ${t}`, `${s} afeta ${t}`],
    'valida':         [`${s} valida ${t}`, `${s} confirma ${t}`],
    'resolve':        [`${s} resolve ${t}`, `${s} soluciona ${t}`],
    'compoe':         [`${s} compõe ${t}`, `${s} forma ${t}`],
    'depende_de':     [`${s} depende de ${t}`, `${s} se baseia em ${t}`],
    'facilita':       [`${s} facilita ${t}`],
    'reduz':          [`${s} reduz ${t}`, `${s} diminui ${t}`],
    'melhora':        [`${s} melhora ${t}`, `${s} aprimora ${t}`],
    'segue_de':       [`${s} decorre de ${t}`, `${s} vem de ${t}`],
    'precede':        [`${s} precede ${t}`, `${s} vem antes de ${t}`],
    'guia':           [`${s} guia ${t}`, `${s} orienta ${t}`],
    'organiza':       [`${s} organiza ${t}`, `${s} estrutura ${t}`],
    'ilustra':        [`${s} ilustra ${t}`, `${s} exemplifica ${t}`],
    'conecta':        [`${s} conecta ${t}`, `${s} liga ${t}`],
    'expressa':       [`${s} expressa ${t}`, `${s} representa ${t}`],
    'clarifica':      [`${s} clarifica ${t}`, `${s} esclarece ${t}`],
    'ajusta':         [`${s} ajusta ${t}`, `${s} corrige ${t}`],
    'fortalece':      [`${s} fortalece ${t}`, `${s} reforça ${t}`],
    'afeta':          [`${s} afeta ${t}`, `${s} impacta ${t}`],
    'constroi':       [`${s} constrói ${t}`, `${s} desenvolve ${t}`],
    'indica':         [`${s} indica ${t}`, `${s} sinaliza ${t}`],
    'emerge_de':      [`${s} emerge de ${t}`, `${s} surge de ${t}`],
    'permite':        [`${s} permite ${t}`, `${s} possibilita ${t}`],
    'bloqueia':       [`${s} bloqueia ${t}`, `${s} impede ${t}`],
    'suporta':        [`${s} suporta ${t}`, `${s} sustenta ${t}`],
    'tem':            [`${s} tem ${t}`, `${s} possui ${t}`],
    'mede':           [`${s} mede ${t}`, `${s} avalia ${t}`],
    'comunica':       [`${s} comunica ${t}`, `${s} transmite ${t}`],
    'resulta_de':     [`${s} resulta de ${t}`, `${s} vem de ${t}`],
    'dificulta':      [`${s} dificulta ${t}`, `${s} complica ${t}`],
    'explica':        [`${s} explica ${t}`, `${s} justifica ${t}`],
    'construido_por': [`${s} é construído por ${t}`, `${s} é formado por ${t}`],
    'expresso_por':   [`${s} é expresso por ${t}`, `${s} é representado por ${t}`],
  };

  const options = templates[relType];
  if (!options) return `${s} ${relType.replace(/_/g, ' ')} ${t}`;

  // Alterna entre variações para diversificar
  return options[Math.floor(Math.random() * options.length)];
}

function graphSentencesToPairs(sentences, vocab) {
  const allPairs = [];
  for (const sentence of sentences) {
    if (!sentence) continue;
    const ids = vocab.tokenize(sentence);
    if (ids.length < 3) continue;
    allPairs.push(...makeTrainingPairs(ids));
  }
  return allPairs;
}

// ---------------------------------------------------------------------------
// FONTE 3 — Web crawl
// CORRIGIDO: usa nexus_graph, filtra por nós com usage_count baixo
// ---------------------------------------------------------------------------

async function fetchWebPatterns(db, vocab, maxQueries = 3) {
  const serperKey = process.env.SERPER_API_KEY;
  const tavilyKey = process.env.TAVILY_API_KEY;

  if (!serperKey && !tavilyKey) {
    console.log('[DATASET] Nenhuma API web configurada. Pulando EXPANSION.');
    return [];
  }

  // Nós com menor uso (precisam de mais conhecimento)
  // CORRIGIDO: nexus_graph, sem filtro por campo 'weight' (não existe no schema)
  const snap = await db.collection('nexus_graph')
    .where('confidence', 'in', ['provisional', 'stale'])
    .limit(15)
    .get();

  if (snap.empty) {
    // Fallback: pegar qualquer nó para expandir vocabulário
    const allSnap = await db.collection('nexus_graph').limit(10).get();
    if (allSnap.empty) return [];
    const candidates = [];
    allSnap.forEach(doc => {
      const d = doc.data();
      if (d.id) candidates.push(d.id.replace(/_/g, ' '));
    });
    return runWebQueries(candidates.slice(0, maxQueries), vocab, serperKey, tavilyKey);
  }

  const candidates = [];
  snap.forEach(doc => {
    const d = doc.data();
    if (d.id) candidates.push(d.id.replace(/_/g, ' '));
  });

  shuffleInPlace(candidates);
  return runWebQueries(candidates.slice(0, maxQueries), vocab, serperKey, tavilyKey);
}

async function runWebQueries(queries, vocab, serperKey, tavilyKey) {
  const allPairs = [];

  for (const query of queries) {
    const searchPromises = [];
    if (serperKey) searchPromises.push(serperSearch(query, serperKey).catch(() => null));
    if (tavilyKey) searchPromises.push(tavilySearch(query, tavilyKey).catch(() => null));

    const results  = await Promise.all(searchPromises);
    const sentences = [];

    for (const result of results) {
      if (!result) continue;
      sentences.push(...extractSentences(result));
    }

    const unique   = [...new Map(sentences.map(s => [s.toLowerCase(), s])).values()];
    const relevant = unique.filter(s => s.length > 15 && s.length < 200).slice(0, 30);

    for (const sentence of relevant) {
      const tokens = tokenizeText(sentence);
      vocab.addTokens(tokens);
      const ids = vocab.tokenize(sentence);
      if (ids.length >= 3) allPairs.push(...makeTrainingPairs(ids));
    }

    const src = [serperKey ? 'Serper' : null, tavilyKey ? 'Tavily' : null]
      .filter(Boolean).join('+');
    console.log(`[DATASET] Web (${src}): "${query}" → ${relevant.length} frases`);
  }

  return allPairs;
}

// ── Serper API ──────────────────────────────────────────────────────────────

async function serperSearch(query, apiKey) {
  const https = require('https');
  const body  = JSON.stringify({ q: query, gl: 'br', hl: 'pt-br', num: 5 });

  return new Promise((resolve, reject) => {
    const req = https.request({
      hostname: 'google.serper.dev',
      path: '/search',
      method: 'POST',
      headers: {
        'X-API-KEY': apiKey,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
      },
    }, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        try { resolve({ _source: 'serper', ...JSON.parse(data) }); }
        catch (e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

// ── Tavily API ──────────────────────────────────────────────────────────────

async function tavilySearch(query, apiKey) {
  const https = require('https');
  const body  = JSON.stringify({ api_key: apiKey, query, search_depth: 'basic', max_results: 3 });

  return new Promise((resolve, reject) => {
    const req = https.request({
      hostname: 'api.tavily.com',
      path: '/search',
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
    }, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        try { resolve({ _source: 'tavily', ...JSON.parse(data) }); }
        catch (e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

// ── Extrator unificado ──────────────────────────────────────────────────────

function extractSentences(response) {
  const raw = [];

  if (response._source === 'serper') {
    (response.organic || []).forEach(r => {
      if (r.snippet) raw.push(r.snippet);
    });
    if (response.knowledgeGraph?.description) raw.push(response.knowledgeGraph.description);
    if (response.answerBox?.answer)  raw.push(response.answerBox.answer);
    if (response.answerBox?.snippet) raw.push(response.answerBox.snippet);
  } else {
    (response.results || []).forEach(r => raw.push(r.content || r.snippet || ''));
  }

  const sentences = [];
  for (const block of raw) {
    if (!block) continue;
    block.replace(/([.!?])\s+/g, '$1\n').split('\n').map(s => s.trim()).filter(s => {
      if (s.length < 15 || s.length > 200) return false;
      return /\bé\b|\bsão\b|\bsignifica\b|\brefere-se\b|\bpermite\b|\bpossui\b|\bfaz\b|\busa\b/.test(s);
    }).forEach(s => sentences.push(s));
  }

  return sentences;
}

function shuffleInPlace(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// ---------------------------------------------------------------------------
// FONTE 4 — Self-play
// ---------------------------------------------------------------------------

async function selfPlayPairs(model, db, vocab, rounds = 10) {
  if (!model.ready) return [];

  // CORRIGIDO: nexus_graph + confidence === 'confirmed'
  const snap = await db.collection('nexus_graph')
    .where('confidence', '==', 'confirmed')
    .limit(30)
    .get();

  if (snap.empty) return [];

  const nodes = [];
  snap.forEach(doc => nodes.push(doc.data()));

  const allPairs = [];
  let good = 0;

  for (let i = 0; i < Math.min(rounds, nodes.length); i++) {
    const node   = nodes[Math.floor(Math.random() * nodes.length)];
    const prompt = node.label || (node.id || '').replace(/_/g, ' ');

    try {
      const generated = model.generate(prompt, 20, 0.7);
      if (!generated) continue;

      const quality = scoreGeneration(generated, vocab);
      if (quality >= 0.5) {
        const ids = vocab.tokenize(`${prompt} ${generated}`);
        if (ids.length >= 3) {
          allPairs.push(...makeTrainingPairs(ids));
          good++;
        }
      }
    } catch (e) { /* continua */ }
  }

  if (good > 0) console.log(`[DATASET] Self-play: ${good}/${rounds} gerações aprovadas`);
  return allPairs;
}

function scoreGeneration(text, vocab) {
  if (!text || text.length < 5) return 0;
  const tokens  = tokenizeText(text);
  if (tokens.length < 2) return 0;
  const unique  = new Set(tokens);
  const divScore  = unique.size / tokens.length;
  const knownCount = tokens.filter(t => vocab.token2id[t] !== undefined).length;
  const knownScore = knownCount / tokens.length;
  const lenScore   = tokens.length >= 5 && tokens.length <= 30 ? 1.0 : 0.5;
  return divScore * 0.4 + knownScore * 0.4 + lenScore * 0.2;
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

module.exports = {
  fetchUserEpisodes,
  episodesToPairs,
  fetchGraphSentences,
  graphSentencesToPairs,
  fetchWebPatterns,
  selfPlayPairs,
  scoreGeneration,
  extractSentences,
  serperSearch,
  tavilySearch,
};
