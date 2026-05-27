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
 *
 * NOVO v3 — EXPANSÃO AUTOMÁTICA DO GRAFO:
 *   - extractAndExpandGraph(): extrai palavras-chave de frases e cria nós
 *   - Integrado em fetchWebPatterns() e selfPlayPairs()
 *   - Evita duplicação, limita 20 nós/ciclo
 *   - Nós gerados com confidence: 'generated'
 *
 * CORREÇÕES v4:
 *   - isNaN(w) → !/^\d+$/.test(w) (filtro correto de números)
 *   - STOP_WORDS expandida com palavras problemáticas
 *   - Filtro de qualidade para pares do web crawl
 *   - Fallback com queries fixas quando grafo sem candidatos
 *   - selfPlayPairs com fallback para nós 'generated'
 */

const { tokenizeText, makeTrainingPairs } = require('./_akie_vocab');

// ---------------------------------------------------------------------------
// CONFIGURAÇÕES DE EXPANSÃO DO GRAFO
// ---------------------------------------------------------------------------

const EXPANSION_CONFIG = {
  MAX_NEW_NODES_PER_CYCLE: 20,       // máximo de nós novos por ciclo
  MIN_WORD_LENGTH: 3,                // tamanho mínimo da palavra para virar nó
  COOCCURRENCE_WINDOW: 5,            // janela para detectar co-ocorrência
  STOP_WORDS: new Set([              // palavras ignoradas na extração
    'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas',
    'de', 'do', 'da', 'dos', 'das', 'em', 'no', 'na', 'nos', 'nas',
    'por', 'pelo', 'pela', 'pelos', 'pelas', 'para', 'com', 'sem',
    'sobre', 'sob', 'entre', 'até', 'desde', 'após', 'antes',
    'e', 'ou', 'mas', 'porém', 'pois', 'porque', 'que', 'se',
    'quando', 'onde', 'como', 'embora', 'ainda', 'já', 'também',
    'não', 'sim', 'muito', 'pouco', 'mais', 'menos', 'bem', 'mal',
    'sempre', 'nunca', 'então', 'assim', 'apenas', 'somente',
    'ser', 'estar', 'ter', 'haver', 'fazer', 'ir', 'vir', 'dar',
    'ver', 'saber', 'poder', 'querer', 'dever', 'precisar',
    'falar', 'dizer', 'pedir', 'responder', 'perguntar',
    'pensar', 'sentir', 'conhecer', 'entender', 'aprender',
    'é', 'são', 'foi', 'foram', 'era', 'eram', 'está', 'estão',
    'ele', 'ela', 'eles', 'elas', 'este', 'esta', 'esse', 'essa',
    'isso', 'isto', 'aquilo', 'meu', 'minha', 'seu', 'sua',
    'nosso', 'nossa', 'qual', 'quais', 'quem', 'cujo', 'cuja',
    // ── Palavras problemáticas detectadas em produção ──
    'unk', 'pode', 'faz', 'vai', 'tem', 'pra', 'pro',
    'tudo', 'nada', 'algo', 'cada', 'todo', 'toda',
    'outro', 'outra', 'outros', 'outras',
    'vez', 'vezes', 'parte',
    'forma', 'maneira', 'modo', 'jeito',
    'exemplo', 'caso', 'questão',
    'aqui', 'ali', 'lá', 'cá',
    'hoje', 'amanhã', 'ontem', 'agora',
    'dia', 'noite', 'tarde', 'manhã',
    'coisa', 'coisas', 'gente',
    'tipo', 'tipos', 'tal', 'tais',
    'assunto', 'tema', 'texto', 'frase',
    'dado', 'dados', 'informação', 'informações',
  ]),
};

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
// EXPANSÃO AUTOMÁTICA DO GRAFO
// ---------------------------------------------------------------------------

/**
 * Extrai palavras-chave de um conjunto de frases e expande o nexus_graph.
 *
 * @param {FirebaseFirestore.Firestore} db
 * @param {string[]} sentences - Frases coletadas (web, self-play, etc.)
 * @param {string} source - Origem: 'web', 'selfplay', 'episodes'
 * @returns {number} Quantidade de nós inseridos
 */
async function extractAndExpandGraph(db, sentences, source = 'unknown') {
  if (!sentences || sentences.length === 0) return 0;

  // ── 1. Extrair palavras-chave de todas as frases ──────────────────────
  const keywordMap = new Map(); // palavra → { count, cooccurrences: Set }

  for (const sentence of sentences) {
    if (!sentence) continue;

    // Tokenizar e limpar
    const words = sentence
      .toLowerCase()
      .replace(/[^a-záàâãéèêíìîóòôõúùûç0-9\s]/g, ' ')
      .split(/\s+/)
      .filter(w =>
        w.length >= EXPANSION_CONFIG.MIN_WORD_LENGTH &&
        !EXPANSION_CONFIG.STOP_WORDS.has(w) &&
        !/^\d+$/.test(w) // ignora números puros
      );

    // Contar frequência
    for (const word of words) {
      if (!keywordMap.has(word)) {
        keywordMap.set(word, { count: 0, cooccurrences: new Set() });
      }
      keywordMap.get(word).count++;
    }

    // Detectar co-ocorrências (janela deslizante)
    for (let i = 0; i < words.length; i++) {
      const windowEnd = Math.min(i + EXPANSION_CONFIG.COOCCURRENCE_WINDOW, words.length);
      for (let j = i + 1; j < windowEnd; j++) {
        if (words[i] !== words[j]) {
          keywordMap.get(words[i])?.cooccurrences.add(words[j]);
          keywordMap.get(words[j])?.cooccurrences.add(words[i]);
        }
      }
    }
  }

  // ── 2. Filtrar palavras mais relevantes (mínimo 2 ocorrências) ───────
  const candidates = Array.from(keywordMap.entries())
    .filter(([, data]) => data.count >= 2)
    .sort((a, b) => b[1].count - a[1].count)
    .slice(0, EXPANSION_CONFIG.MAX_NEW_NODES_PER_CYCLE * 2); // margem extra

  if (candidates.length === 0) {
    console.log(`[EXPANSION] Nenhuma palavra-chave relevante extraída de ${source}`);
    return 0;
  }

  // ── 3. Verificar quais já existem no grafo ───────────────────────────
  const candidateIds = candidates.map(([word]) => sanitizeNodeId(word));
  const existingIds  = new Set();

  // Busca em batches (Firestore permite até 30 IDs no 'in')
  for (let i = 0; i < candidateIds.length; i += 30) {
    const chunk = candidateIds.slice(i, i + 30);
    const snap  = await db.collection('nexus_graph')
      .where('__name__', 'in', chunk)
      .get();
    snap.forEach(doc => existingIds.add(doc.id));
  }

  // ── 4. Criar novos nós ───────────────────────────────────────────────
  const newNodes = [];
  const addedIds = new Set();

  for (const [word, data] of candidates) {
    const nodeId = sanitizeNodeId(word);

    // Pular se já existe ou já foi adicionado neste ciclo
    if (existingIds.has(nodeId) || addedIds.has(nodeId)) continue;
    if (newNodes.length >= EXPANSION_CONFIG.MAX_NEW_NODES_PER_CYCLE) break;

    // Criar relações com co-ocorrências mais frequentes
    const relations = [];
    const cooccurArray = Array.from(data.cooccurrences)
      .filter(co =>
        co.length >= EXPANSION_CONFIG.MIN_WORD_LENGTH &&
        !EXPANSION_CONFIG.STOP_WORDS.has(co) // também filtra stop words nas relações
      )
      .slice(0, 5);

    for (const coWord of cooccurArray) {
      relations.push({
        target: sanitizeNodeId(coWord),
        type: 'related_to',
        weight: 0.3, // peso inicial baixo para relações geradas
      });
    }

    // Criar nó
    const node = {
      id: nodeId,
      label: word,
      contexts: [`${source}_generated`],
      verbs: [],
      relations: relations,
      confidence: 'generated',
      created_at: new Date().toISOString(),
      usage_count: 0,
      source: source,
      occurrence_count: data.count,
    };

    newNodes.push(node);
    addedIds.add(nodeId);
  }

  // ── 5. Salvar no Firestore em batch ──────────────────────────────────
  if (newNodes.length > 0) {
    const BATCH_SIZE = 400;
    for (let i = 0; i < newNodes.length; i += BATCH_SIZE) {
      const batch = db.batch();
      const chunk = newNodes.slice(i, i + BATCH_SIZE);
      for (const node of chunk) {
        const ref = db.collection('nexus_graph').doc(node.id);
        batch.set(ref, node);
      }
      await batch.commit();
    }
    console.log(`[EXPANSION] ${newNodes.length} novos nós inseridos no nexus_graph (source: ${source})`);
    console.log(`[EXPANSION] Palavras: ${newNodes.map(n => n.label).join(', ')}`);
  } else {
    console.log(`[EXPANSION] Nenhum nó novo para adicionar (source: ${source})`);
  }

  return newNodes.length;
}

/**
 * Sanitiza uma palavra para uso como ID de nó no Firestore.
 */
function sanitizeNodeId(word) {
  return word
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '') // remove acentos
    .replace(/[^a-z0-9]/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '')
    .substring(0, 80); // limite de tamanho
}

/**
 * Extrai frases de um array de objetos {input, output} para expandir o grafo.
 */
function episodesToSentences(episodes) {
  const sentences = [];
  for (const ep of episodes) {
    if (ep.input) sentences.push(ep.input);
    if (ep.output) sentences.push(ep.output);
  }
  return sentences;
}

// ---------------------------------------------------------------------------
// FONTE 3 — Web crawl
// ---------------------------------------------------------------------------

async function fetchWebPatterns(db, vocab, maxQueries = 3) {
  const serperKey = process.env.SERPER_API_KEY;
  const tavilyKey = process.env.TAVILY_API_KEY;

  if (!serperKey && !tavilyKey) {
    console.log('[DATASET] Nenhuma API web configurada. Pulando EXPANSION.');
    return [];
  }

  // Nós com menor uso (precisam de mais conhecimento)
  const snap = await db.collection('nexus_graph')
    .where('confidence', 'in', ['provisional', 'stale'])
    .limit(15)
    .get();

  let queries = [];

  if (snap.empty) {
    // Fallback: queries fixas de conhecimento geral
    const fallbackQueries = [
      'inteligência artificial aprendizado',
      'linguagem natural processamento',
      'como funciona rede neural',
      'significado de conhecimento',
      'tecnologia e inovação',
    ];
    shuffleInPlace(fallbackQueries);
    queries = fallbackQueries.slice(0, maxQueries);
    console.log('[DATASET] Web: usando queries fixas (grafo sem candidatos)');
  } else {
    snap.forEach(doc => {
      const d = doc.data();
      if (d.id) queries.push(d.id.replace(/_/g, ' '));
    });
    shuffleInPlace(queries);
  }

  queries = queries.slice(0, maxQueries);

  // Executar buscas e coletar frases
  const allPairs = [];
  const allSentences = [];

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

    // Coletar frases para expansão
    allSentences.push(...relevant);

    // Gerar pares de treino
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

  // ── Expandir grafo com frases coletadas da web ──────────────────
  if (allSentences.length > 0) {
    try {
      const inserted = await extractAndExpandGraph(db, allSentences, 'web');
      console.log(`[DATASET] Expansão do grafo via web: ${inserted} nós adicionados`);
    } catch (err) {
      console.error('[DATASET] Erro na expansão do grafo (web):', err.message);
    }
  }

  // ── FILTRO DE QUALIDADE ──────────────────────────────────────────
  const vocabSize = vocab.size;
  const filteredPairs = allPairs.filter(pair => {
    const knownTokens = pair.filter(id => id < vocabSize);
    return knownTokens.length === 2; // ambos os tokens precisam ser conhecidos
  });

  if (filteredPairs.length < allPairs.length) {
    console.log(`[DATASET] Web: ${allPairs.length - filteredPairs.length} pares descartados (qualidade)`);
  }

  return filteredPairs;
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

  // Primeira tentativa: nós confirmados
  let snap = await db.collection('nexus_graph')
    .where('confidence', '==', 'confirmed')
    .limit(30)
    .get();

  // Fallback: incluir nós gerados também
  if (snap.empty) {
    console.log('[DATASET] Self-play: sem nós confirmed, usando generated...');
    snap = await db.collection('nexus_graph')
      .where('confidence', 'in', ['confirmed', 'generated'])
      .limit(30)
      .get();
  }

  if (snap.empty) {
    console.log('[DATASET] Self-play: nenhum nó disponível no grafo');
    return [];
  }

  const nodes = [];
  snap.forEach(doc => nodes.push(doc.data()));

  const allPairs = [];
  const allGeneratedTexts = [];
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
          // Coletar texto para expansão
          allGeneratedTexts.push(`${prompt} ${generated}`);
        }
      }
    } catch (e) { /* continua */ }
  }

  if (good > 0) console.log(`[DATASET] Self-play: ${good}/${rounds} gerações aprovadas`);

  // ── Expandir grafo com textos gerados pelo self-play ────────────
  if (allGeneratedTexts.length > 0) {
    try {
      const inserted = await extractAndExpandGraph(db, allGeneratedTexts, 'selfplay');
      console.log(`[DATASET] Expansão do grafo via self-play: ${inserted} nós adicionados`);
    } catch (err) {
      console.error('[DATASET] Erro na expansão do grafo (self-play):', err.message);
    }
  }

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
  // Expansão do grafo
  extractAndExpandGraph,
  episodesToSentences,
  sanitizeNodeId,
  EXPANSION_CONFIG,
};
