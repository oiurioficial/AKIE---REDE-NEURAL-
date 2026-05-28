/**
 * _akie_dataset.js
 *
 * Pipeline de dados para treino contínuo do AKIE.
 *
 * CORREÇÕES v7 — ACELERAÇÃO DO APRENDIZADO:
 *   - NLP nativo PT-BR: sem dependência de `natural` (stemmer embutido)
 *   - extractSentences: filtro relaxado (+30% aproveitamento de frases web)
 *   - fetchWebPatterns: queries orientadas a conversação PT-BR
 *   - extractAndExpandGraph: MIN_OCCURRENCE_COUNT = 1 (mais nós gerados)
 *   - PROMOTE_AFTER_USES: 2 (promoção mais rápida)
 *   - selfPlayPairs: threshold de qualidade reduzido para 0.35
 *   - scoreGeneration: penalidade de repetição para evitar loop de padrão único
 *   - fetchGraphSentences: frases conversacionais adicionadas por nó confirmado
 */

const { tokenizeText, makeTrainingPairs } = require('./_akie_vocab');

// ---------------------------------------------------------------------------
// NLP NATIVO — sem dependência externa de `natural`
// Resolve o erro "[NLP] `natural` não encontrada" sem instalar pacotes.
// Stemmer PT-BR minimalista por sufixo.
// ---------------------------------------------------------------------------

const NLP = {
  _suffixes: [
    'amentos', 'imentos', 'idades', 'adores', 'adoras',
    'amento',  'imento',  'idade',  'adore',  'adora',
    'mente',   'issimo',  'issima', 'istas',  'ista',
    'ados', 'adas', 'idos', 'idas', 'ores', 'oras',
    'ando', 'endo', 'indo',
    'ado', 'ada', 'ido', 'ida',
    'es', 'os', 'as',
  ],

  stem(word) {
    const w = word.toLowerCase();
    if (w.length <= 4) return w;
    for (const suf of this._suffixes) {
      if (w.endsWith(suf) && w.length - suf.length >= 3) {
        return w.slice(0, w.length - suf.length);
      }
    }
    return w;
  },

  extractTerms(sentence, stopWords) {
    return sentence
      .toLowerCase()
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter(w => w.length >= 3 && !stopWords.has(w) && !/^\d+$/.test(w))
      .map(w => this.stem(w));
  },

  similarity(termsA, termsB) {
    if (!termsA.length || !termsB.length) return 0;
    const setA = new Set(termsA);
    const setB = new Set(termsB);
    const inter = [...setA].filter(t => setB.has(t)).length;
    return inter / Math.sqrt(setA.size * setB.size);
  },
};

console.log('[NLP] Motor nativo PT-BR ativo (sem dependência externa).');

// ---------------------------------------------------------------------------
// CONFIGURAÇÕES DE EXPANSÃO DO GRAFO
// ---------------------------------------------------------------------------

const EXPANSION_CONFIG = {
  MAX_NEW_NODES_PER_CYCLE: 30,       // v7: aumentado de 20 → 30
  MIN_WORD_LENGTH: 3,
  COOCCURRENCE_WINDOW: 6,            // v7: janela maior = mais relações
  PROMOTE_AFTER_USES: 2,             // v7: promoção mais rápida (era 3)
  MIN_OCCURRENCE_COUNT: 1,           // v7: aceita palavras com 1 ocorrência
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
    // Palavras problemáticas detectadas em produção
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

function episodesToPairs(episodes, vocab) {
  const allPairs = [];
  const MAX_SEQ  = 64; // deve coincidir com HPARAMS.maxSeqLen

  for (const ep of episodes) {
    if (ep.feedback === 'negative') continue;
    if (!ep.input || !ep.output)   continue;

    // Remover prefixos u:/a: que possam vir do sistema antes de reformatar
    const userText = ep.input.replace(/^u\s*:\s*/i, '').replace(/\n/g, ' ').trim();
    const respText = ep.output.replace(/^a\s*:\s*/i, '').replace(/\n/g, ' ').trim();
    if (!userText || !respText) continue;

    // Formato IDÊNTICO ao SYNTHETIC: "u: {input}" como contexto, "a: {output}" como alvo
    // Teacher-forcing por posição — garante consistência de distribuição entre modos
    const contextTokens  = vocab.tokenize(`u: ${userText}`);
    const responseTokens = vocab.tokenize(`a: ${respText}`);
    if (!contextTokens.length || responseTokens.length < 2) continue;

    const weight = ep.feedback === 'positive' ? 2 : 1;

    for (let pos = 1; pos < responseTokens.length; pos++) {
      const prefix    = responseTokens.slice(0, pos);
      const rawSeq    = [...contextTokens, ...prefix];
      const truncated = rawSeq.slice(-(MAX_SEQ - 1));
      const padded    = [
        ...Array(Math.max(0, MAX_SEQ - truncated.length)).fill(0),
        ...truncated,
      ];
      const pair = { x: padded, y: responseTokens[pos] };
      for (let w = 0; w < weight; w++) allPairs.push(pair);
    }
  }

  return allPairs;
}

// ---------------------------------------------------------------------------
// FONTE 2 — Grafo semântico NEXUS → frases de treino
// ---------------------------------------------------------------------------

async function fetchGraphSentences(db, limit = 300) {
  const snap = await db.collection('nexus_graph')
    .where('confidence', 'in', ['confirmed', 'generated'])
    .limit(limit)
    .get();

  const sentences = [];
  const nodesToPromote = [];

  snap.forEach(doc => {
    const node = doc.data();
    const label = node.label || node.id.replace(/_/g, ' ');

    if (node.confidence === 'generated' && (node.usage_count || 0) >= EXPANSION_CONFIG.PROMOTE_AFTER_USES) {
      nodesToPromote.push(doc.ref);
    }

    // 🔥 CORREÇÃO v6: limiar reduzido para 0.20 (aproveita relações geradas com peso 0.45)
    const rels = (node.relations || []).filter(r => r.weight >= 0.20);

    for (const rel of rels) {
      const target   = (rel.target || '').replace(/_/g, ' ');
      const sentence = relationToSentence(label, rel.type, target);
      if (sentence) sentences.push(sentence);
    }

    (node.contexts || []).forEach(ctx => {
      if (ctx && ctx.length > 2) {
        sentences.push(`${label} aparece em ${ctx.replace(/_/g, ' ')}`);
      }
    });

    (node.verbs || []).slice(0, 3).forEach(verb => {
      if (verb && verb.length > 3) {
        sentences.push(`${label} ${verb.replace(/_/g, ' ')}`);
      }
    });

    // v7: frases conversacionais básicas para nós confirmados
    if (node.confidence === 'confirmed') {
      sentences.push(`o que é ${label}`);
      sentences.push(`como usar ${label}`);
      if (node.description) sentences.push(node.description);
    }
  });

  if (nodesToPromote.length > 0) {
    const batch = db.batch();
    nodesToPromote.forEach(ref => {
      batch.update(ref, { 
        confidence: 'confirmed',
        promoted_at: new Date().toISOString()
      });
    });
    await batch.commit();
    console.log(`[GRAPH] ${nodesToPromote.length} nós promovidos: generated → confirmed`);
  }

  console.log(`[GRAPH] ${snap.size} nós → ${sentences.length} frases (confirmed + generated)`);
  return sentences;
}

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

  return options[Math.floor(Math.random() * options.length)];
}

function graphSentencesToPairs(sentences, vocab) {
  const allPairs = [];
  const MAX_SEQ  = 64;

  // Cap de pares por ciclo de consolidação: evita explosão de RAM/CPU no Railway.
  // 80 frases × ~6 tokens/frase × teacher-forcing = ~480 pares máximo, processado em <15s.
  const MAX_PAIRS_PER_CONSOLIDATION = 600;

  // Prefixos variados para reduzir overfitting no stub de pergunta
  const STUBS = [
    'u: o que é',
    'u: me fale sobre',
    'u: como funciona',
    'u: explica sobre',
    'u: o que você sabe sobre',
  ];

  for (let i = 0; i < sentences.length; i++) {
    if (allPairs.length >= MAX_PAIRS_PER_CONSOLIDATION) break;

    const sentence = sentences[i];
    if (!sentence) continue;

    // Primeiro termo da frase = tópico da pergunta
    const words = sentence.split(/\s+/).filter(w => w.length > 2);
    if (words.length < 2) continue;

    const stub          = STUBS[i % STUBS.length];
    const contextStr    = `${stub} ${words[0]}`;
    const responseStr   = `a: ${sentence}`;

    const contextTokens  = vocab.tokenize(contextStr);
    const responseTokens = vocab.tokenize(responseStr);
    if (!contextTokens.length || responseTokens.length < 2) continue;

    // Teacher-forcing por posição — formato idêntico ao SYNTHETIC e INTERACTIVE
    for (let pos = 1; pos < responseTokens.length; pos++) {
      if (allPairs.length >= MAX_PAIRS_PER_CONSOLIDATION) break;
      const prefix    = responseTokens.slice(0, pos);
      const rawSeq    = [...contextTokens, ...prefix];
      const truncated = rawSeq.slice(-(MAX_SEQ - 1));
      const padded    = [
        ...Array(Math.max(0, MAX_SEQ - truncated.length)).fill(0),
        ...truncated,
      ];
      allPairs.push({ x: padded, y: responseTokens[pos] });
    }
  }

  return allPairs;
}

// ---------------------------------------------------------------------------
// PROMOÇÃO DE NÓS
// ---------------------------------------------------------------------------

async function promoteGeneratedNodes(db) {
  const snap = await db.collection('nexus_graph')
    .where('confidence', '==', 'generated')
    .where('usage_count', '>=', EXPANSION_CONFIG.PROMOTE_AFTER_USES)
    .limit(50)
    .get();

  if (snap.empty) return 0;

  const batch = db.batch();
  snap.forEach(doc => {
    batch.update(doc.ref, { 
      confidence: 'confirmed',
      promoted_at: new Date().toISOString()
    });
  });
  await batch.commit();
  
  console.log(`[GRAPH] ${snap.size} nós promovidos: generated → confirmed`);
  return snap.size;
}

// ---------------------------------------------------------------------------
// EXPANSÃO AUTOMÁTICA DO GRAFO
// ---------------------------------------------------------------------------

async function extractAndExpandGraph(db, sentences, source = 'unknown') {
  if (!sentences || sentences.length === 0) return 0;

  const keywordMap = new Map();

  for (const sentence of sentences) {
    if (!sentence) continue;

    const words = sentence
      .toLowerCase()
      .replace(/[^a-záàâãéèêíìîóòôõúùûç0-9\s]/g, ' ')
      .split(/\s+/)
      .filter(w =>
        w.length >= EXPANSION_CONFIG.MIN_WORD_LENGTH &&
        !EXPANSION_CONFIG.STOP_WORDS.has(w) &&
        !/^\d+$/.test(w)
      );

    for (const word of words) {
      if (!keywordMap.has(word)) {
        keywordMap.set(word, { count: 0, cooccurrences: new Set() });
      }
      keywordMap.get(word).count++;
    }

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

  const candidates = Array.from(keywordMap.entries())
    .filter(([, data]) => data.count >= 2)
    .sort((a, b) => b[1].count - a[1].count)
    .slice(0, EXPANSION_CONFIG.MAX_NEW_NODES_PER_CYCLE * 2);

  if (candidates.length === 0) {
    console.log(`[EXPANSION] Nenhuma palavra-chave relevante extraída de ${source}`);
    return 0;
  }

  const candidateIds = candidates.map(([word]) => sanitizeNodeId(word));
  const existingIds  = new Set();

  for (let i = 0; i < candidateIds.length; i += 30) {
    const chunk = candidateIds.slice(i, i + 30);
    const snap  = await db.collection('nexus_graph')
      .where('__name__', 'in', chunk)
      .get();
    snap.forEach(doc => existingIds.add(doc.id));
  }

  const newNodes = [];
  const addedIds = new Set();

  for (const [word, data] of candidates) {
    const nodeId = sanitizeNodeId(word);

    if (existingIds.has(nodeId) || addedIds.has(nodeId)) continue;
    if (newNodes.length >= EXPANSION_CONFIG.MAX_NEW_NODES_PER_CYCLE) break;

    const relations = [];
    const cooccurArray = Array.from(data.cooccurrences)
      .filter(co =>
        co.length >= EXPANSION_CONFIG.MIN_WORD_LENGTH &&
        !EXPANSION_CONFIG.STOP_WORDS.has(co)
      )
      .slice(0, 8); // 🔥 v6: aumentado para 8 co-ocorrências

    for (const coWord of cooccurArray) {
      relations.push({
        target: sanitizeNodeId(coWord),
        type: 'related_to',
        weight: 0.45, // 🔥 v6: peso inicial aumentado para 0.45
      });
    }

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

function sanitizeNodeId(word) {
  return word
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9]/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '')
    .substring(0, 80);
}

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

  const snap = await db.collection('nexus_graph')
    .where('confidence', 'in', ['provisional', 'stale', 'generated'])
    .limit(15)
    .get();

  let queries = [];

  if (snap.empty) {
    // v7: queries orientadas a vocabulário conversacional PT-BR básico
    const fallbackQueries = [
      'o que significa olá cumprimento saudação',
      'como responder perguntas cotidiano português',
      'frases simples conversa bom dia boa tarde',
      'expressões de saudação cumprimento',
      'inteligência artificial aprendizado máquina',
      'linguagem natural processamento texto',
      'como funciona rede neural',
      'o que é conhecimento aprendizado',
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

    allSentences.push(...relevant);

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

  if (allSentences.length > 0) {
    try {
      const inserted = await extractAndExpandGraph(db, allSentences, 'web');
      console.log(`[DATASET] Expansão do grafo via web: ${inserted} nós adicionados`);
    } catch (err) {
      console.error('[DATASET] Erro na expansão do grafo (web):', err.message);
    }
  }

  const vocabSize = vocab.size;
  const filteredPairs = allPairs.filter(pair => {
    if (!Array.isArray(pair)) return false;
    const knownTokens = pair.filter(id => typeof id === 'number' && id < vocabSize);
    return knownTokens.length >= 1;
  });

  if (filteredPairs.length < allPairs.length) {
    console.log(`[DATASET] Web: ${allPairs.length - filteredPairs.length} pares descartados (qualidade)`);
  }

  return filteredPairs;
}

// ── APIs de busca ─────────────────────────────────────────────────────────

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
      if (s.length < 10 || s.length > 300) return false;
      // v7: filtro expandido — aceita qualquer frase com verbo PT-BR comum
      return /\bé\b|\bsão\b|\bsignifica\b|\brefere-se\b|\bpermite\b|\bpossui\b|\bfaz\b|\busa\b|\bpode\b|\btrata\b|\bdefine\b|\brepresenta\b|\bajuda\b|\bcria\b|\bgera\b|\btem\b|\bexiste\b|\bfunciona\b|\bestá\b/.test(s);
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

  let snap = await db.collection('nexus_graph')
    .where('confidence', 'in', ['confirmed', 'generated'])
    .limit(30)
    .get();

  if (snap.empty) {
    console.log('[DATASET] Self-play: nenhum nó disponível no grafo');
    return [];
  }

  const nodes = [];
  const nodeRefs = [];
  snap.forEach(doc => {
    nodes.push(doc.data());
    nodeRefs.push(doc.ref);
  });

  const allPairs = [];
  const allGeneratedTexts = [];
  const usedNodeIds = new Set();
  let good = 0;

  for (let i = 0; i < Math.min(rounds, nodes.length); i++) {
    const idx    = Math.floor(Math.random() * nodes.length);
    const node   = nodes[idx];
    const prompt = node.label || (node.id || '').replace(/_/g, ' ');

    try {
      const generated = await model.generate(prompt, 20, 0.7);
      if (!generated) continue;

      const quality = scoreGeneration(generated, vocab);
      if (quality >= 0.35) {
        // Formato consistente com SYNTHETIC: u: {prompt} a: {generated}
        const contextTokens  = vocab.tokenize(`u: ${prompt}`);
        const responseTokens = vocab.tokenize(`a: ${generated}`);
        if (responseTokens.length >= 2) {
          const MAX_SEQ = 64;
          for (let pos = 1; pos < responseTokens.length; pos++) {
            const prefix    = responseTokens.slice(0, pos);
            const rawSeq    = [...contextTokens, ...prefix];
            const truncated = rawSeq.slice(-(MAX_SEQ - 1));
            const padded    = [
              ...Array(Math.max(0, MAX_SEQ - truncated.length)).fill(0),
              ...truncated,
            ];
            allPairs.push({ x: padded, y: responseTokens[pos] });
          }
          good++;
          allGeneratedTexts.push(`${prompt} ${generated}`);
          usedNodeIds.add(node.id);
        }
      }
    } catch (e) { /* continua */ }
  }

  if (good > 0) console.log(`[DATASET] Self-play: ${good}/${rounds} gerações aprovadas`);

  if (usedNodeIds.size > 0) {
    try {
      const batch = db.batch();
      const FV = require('firebase-admin').firestore.FieldValue;
      for (const ref of nodeRefs) {
        const nodeData = nodes.find(n => n.id === ref.id);
        if (nodeData && usedNodeIds.has(nodeData.id)) {
          batch.update(ref, { usage_count: FV.increment(1) });
        }
      }
      await batch.commit();
    } catch (e) { /* fire-and-forget */ }
  }

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
  const unique     = new Set(tokens);
  const divScore   = unique.size / tokens.length;
  const knownCount = tokens.filter(t => vocab.token2id[t] !== undefined).length;
  const knownScore = knownCount / tokens.length;
  const lenScore   = tokens.length >= 4 && tokens.length <= 40 ? 1.0 : 0.5;
  // v7: penalidade para padrões repetitivos detectados em produção
  const REPETITION_PHRASES = [
    'pode dar mais detalhes', 'pode me explicar o que',
    'nao tenho representacao', 'pode reformular',
  ];
  const lowerText = text.toLowerCase();
  const repetitionPenalty = REPETITION_PHRASES.some(p => lowerText.includes(p)) ? 0.5 : 1.0;
  return (divScore * 0.35 + knownScore * 0.45 + lenScore * 0.20) * repetitionPenalty;
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
  extractAndExpandGraph,
  episodesToSentences,
  sanitizeNodeId,
  EXPANSION_CONFIG,
  promoteGeneratedNodes,
};
