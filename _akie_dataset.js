/**
 * _akie_dataset.js
 *
 * Pipeline de dados para treino contínuo do AKIE.
 * Quatro fontes independentes — o worker nunca fica sem dados.
 *
 *  FONTE 1 — Episódios reais (usuário interagindo)
 *  FONTE 2 — Grafo semântico (relações do Firestore → frases)
 *  FONTE 3 — Web crawl via Tavily (quando há lacunas no grafo)
 *  FONTE 4 — Self-play (modelo gera, avalia, treina nos melhores)
 */

const { tokenizeText, makeTrainingPairs, SPECIAL } = require('./_akie_vocab');

// ---------------------------------------------------------------------------
// FONTE 1 — Episódios reais de interação
// ---------------------------------------------------------------------------

/**
 * Busca episódios não processados do Firestore.
 * Marca como processed=true após extrair.
 */
async function fetchUserEpisodes(db, limit = 50) {
  const snap = await db.collection('nexus_episodes')
    .where('processed', '==', false)
    .orderBy('created_at', 'asc')
    .limit(limit)
    .get();

  if (snap.empty) return [];

  const episodes = [];
  const batch = db.batch();

  snap.forEach(doc => {
    const d = doc.data();
    episodes.push({
      input:    d.input    || d.user_message || '',
      output:   d.output   || d.system_message || '',
      feedback: d.feedback || null,  // 'positive' | 'negative' | null
      layer:    d.layer    || null,
    });
    // Marcar como processado
    batch.update(doc.ref, { processed: true, processed_at: new Date().toISOString() });
  });

  await batch.commit();
  return episodes;
}

/**
 * Converte episódios em pares de treino.
 * Formato: "INPUT: <texto> RESPOSTA: <texto>"
 * Feedback positivo → peso 2x (reforço).
 * Feedback negativo → descartado.
 */
function episodesToPairs(episodes, vocab) {
  const allPairs = [];

  for (const ep of episodes) {
    if (ep.feedback === 'negative') continue; // não treinar em exemplos ruins

    if (!ep.input || !ep.output) continue;

    // Construir sequência completa: contexto + resposta
    const fullText = `${ep.input} ${ep.output}`;
    const ids = vocab.tokenize(fullText);
    const pairs = makeTrainingPairs(ids, vocab /* maxSeqLen via vocab config */);

    const weight = ep.feedback === 'positive' ? 2 : 1;
    for (let i = 0; i < weight; i++) {
      allPairs.push(...pairs);
    }
  }

  return allPairs;
}

// ---------------------------------------------------------------------------
// FONTE 2 — Síntese a partir do grafo semântico
// ---------------------------------------------------------------------------

/**
 * Busca nós confirmados do Firestore e converte em frases de treino.
 *
 * Cada relação no grafo vira uma frase:
 *   ola --[e_tipo]--> saudacao  → "ola é um tipo de saudacao"
 *   interno --[significa]--> de_dentro → "interno significa de dentro"
 */
async function fetchGraphSentences(db, limit = 200) {
  const snap = await db.collection('nexus_nodes')
    .where('confirmed', '==', true)
    .limit(limit)
    .get();

  const sentences = [];

  snap.forEach(doc => {
    const node = doc.data();
    const rels = (node.relations || []).filter(r => r.confirmed && r.weight >= 0.70);

    for (const rel of rels) {
      const sentence = relationToSentence(
        node.label || node.id,
        rel.type,
        rel.target.replace(/_/g, ' ')
      );
      if (sentence) sentences.push(sentence);
    }
  });

  return sentences;
}

/**
 * Converte tipo de relação em frase natural.
 */
function relationToSentence(source, relType, target) {
  const s = source.replace(/_/g, ' ');
  const t = target.replace(/_/g, ' ');

  const templates = {
    'e_tipo':       [`${s} é um tipo de ${t}`, `${s} é ${t}`],
    'significa':    [`${s} significa ${t}`, `o significado de ${s} é ${t}`],
    'serve_para':   [`${s} serve para ${t}`, `${s} é usado para ${t}`],
    'usado_para':   [`${s} é usado para ${t}`],
    'representa':   [`${s} representa ${t}`],
    'parte_de':     [`${s} faz parte de ${t}`, `${s} é parte de ${t}`],
    'similar_a':    [`${s} é similar a ${t}`, `${s} é parecido com ${t}`],
    'descrito_como':[`${s} pode ser descrito como ${t}`],
  };

  const options = templates[relType];
  if (!options) return null;

  // Alterna entre variações para diversificar o dataset
  return options[Math.floor(Math.random() * options.length)];
}

/**
 * Converte frases do grafo em pares de treino.
 */
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
// FONTE 3 — Web crawl (Serper + Tavily em paralelo, fallback automático)
// ---------------------------------------------------------------------------

/**
 * Busca texto da web sobre conceitos com baixa cobertura no grafo.
 * Usa Serper (Google) e Tavily em paralelo — mais resultados, mais diversidade.
 * Extrai PADRÕES relacionais — não armazena texto bruto.
 *
 * Prioridade de busca:
 *   1. Serper (Google Search) — snippets ricos, knowledge graph quando disponível
 *   2. Tavily — fallback ou complemento
 *   Se nenhuma chave disponível → pula Modo EXPANSION silenciosamente
 */
async function fetchWebPatterns(db, vocab, maxQueries = 3) {
  const serperKey = process.env.SERPER_API_KEY;
  const tavilyKey = process.env.TAVILY_API_KEY;

  if (!serperKey && !tavilyKey) {
    console.log('[DATASET] Nenhuma API web configurada. Pulando EXPANSION.');
    return [];
  }

  // Encontrar nós com peso baixo (precisam de mais conhecimento)
  const snap = await db.collection('nexus_nodes')
    .where('weight', '<', 0.65)
    .limit(15)
    .get();

  if (snap.empty) return [];

  const candidates = [];
  snap.forEach(doc => {
    const d = doc.data();
    if (d.id && d.id.length > 2) candidates.push(d.id.replace(/_/g, ' '));
  });

  if (candidates.length === 0) return [];

  // Embaralhar para não buscar sempre os mesmos nós
  shuffleInPlace(candidates);
  const queries = candidates.slice(0, maxQueries);
  const allPairs = [];

  for (const query of queries) {
    const sentences = [];

    // Buscar em paralelo quando ambas as chaves disponíveis
    const searchPromises = [];
    if (serperKey) searchPromises.push(
      serperSearch(query, serperKey).catch(e => {
        console.error(`[DATASET] Serper falhou "${query}":`, e.message);
        return null;
      })
    );
    if (tavilyKey) searchPromises.push(
      tavilySearch(query, tavilyKey).catch(e => {
        console.error(`[DATASET] Tavily falhou "${query}":`, e.message);
        return null;
      })
    );

    const results = await Promise.all(searchPromises);

    for (const result of results) {
      if (!result) continue;
      sentences.push(...extractSentences(result));
    }

    // Deduplicar frases (Serper e Tavily podem retornar as mesmas)
    const unique = [...new Map(sentences.map(s => [s.toLowerCase(), s])).values()];

    const relevant = unique
      .filter(s => s.length > 15 && s.length < 200)
      .slice(0, 30); // máximo 30 frases por query

    for (const sentence of relevant) {
      const tokens = tokenizeText(sentence);
      vocab.addTokens(tokens);
      const ids = vocab.tokenize(sentence);
      if (ids.length >= 3) {
        allPairs.push(...makeTrainingPairs(ids));
      }
    }

    const sources = [serperKey ? 'Serper' : null, tavilyKey ? 'Tavily' : null]
      .filter(Boolean).join('+');
    console.log(`[DATASET] Web (${sources}): "${query}" → ${relevant.length} frases`);
  }

  return allPairs;
}

// ── Serper API (Google Search) ───────────────────────────────────────────────

async function serperSearch(query, apiKey) {
  const https = require('https');
  const body = JSON.stringify({
    q: query,
    gl: 'br',      // resultados em PT-BR
    hl: 'pt-br',
    num: 5,
  });

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
    }, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
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

// ── Tavily API ───────────────────────────────────────────────────────────────

async function tavilySearch(query, apiKey) {
  const https = require('https');
  const body = JSON.stringify({
    api_key: apiKey,
    query,
    search_depth: 'basic',
    max_results: 3,
    include_raw_content: false,
  });

  return new Promise((resolve, reject) => {
    const req = https.request({
      hostname: 'api.tavily.com',
      path: '/search',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
      },
    }, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
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

// ── Extrator unificado (Serper + Tavily) ─────────────────────────────────────

/**
 * Extrai frases relacionais de qualquer resposta de API de busca.
 * Serper tem estrutura diferente de Tavily — normaliza aqui.
 */
function extractSentences(response) {
  const raw = [];

  if (response._source === 'serper') {
    // Serper: organic results + answerBox + knowledgeGraph
    const organics = response.organic || [];
    for (const r of organics) {
      if (r.snippet) raw.push(r.snippet);
      if (r.sitelinks) {
        r.sitelinks.forEach(sl => sl.snippet && raw.push(sl.snippet));
      }
    }

    // Knowledge Graph — ouro puro para extração semântica
    if (response.knowledgeGraph) {
      const kg = response.knowledgeGraph;
      if (kg.description) raw.push(kg.description);
      if (kg.attributes) {
        Object.entries(kg.attributes).forEach(([key, val]) => {
          if (typeof val === 'string' && val.length < 120) {
            // Usar "é" só quando o key não é já um verbo
            const verb = /^(é|são|faz|usa|tem|possui|pertence)/.test(key) ? '' : ' é';
            raw.push(`${kg.title || ''} ${key}${verb} ${val}`);
          }
        });
      }
    }

    // Answer box
    if (response.answerBox?.answer) raw.push(response.answerBox.answer);
    if (response.answerBox?.snippet) raw.push(response.answerBox.snippet);

  } else {
    // Tavily
    const results = response.results || [];
    for (const r of results) {
      raw.push(r.content || r.snippet || '');
    }
  }

  // Fragmentar em frases e filtrar predicativas
  const sentences = [];
  for (const block of raw) {
    if (!block) continue;
    const parts = block
      .replace(/([.!?])\s+/g, '$1\n')
      .split('\n')
      .map(s => s.trim())
      .filter(s => {
        if (s.length < 15 || s.length > 200) return false;
        return /\bé\b|\bsão\b|\bsignifica\b|\brefere-se\b|\bpermite\b|\bpossui\b|\bfaz\b|\busa\b/.test(s);
      });
    sentences.push(...parts);
  }

  return sentences;
}

// Utilitário local (não exportado)
function shuffleInPlace(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// ---------------------------------------------------------------------------
// FONTE 4 — Self-play
// ---------------------------------------------------------------------------

/**
 * O modelo gera respostas para prompts do grafo,
 * avalia a coerência, treina nos melhores outputs.
 *
 * Critérios de qualidade:
 *   - comprimento mínimo (não vazio)
 *   - sem repetição excessiva de tokens
 *   - tokens presentes no vocabulário confirmado
 */
async function selfPlayPairs(model, db, vocab, rounds = 10) {
  if (!model.ready) return [];

  // Buscar prompts candidatos do grafo
  const snap = await db.collection('nexus_nodes')
    .where('confirmed', '==', true)
    .limit(30)
    .get();

  if (snap.empty) return [];

  const nodes = [];
  snap.forEach(doc => nodes.push(doc.data()));

  const allPairs = [];
  let goodGenerations = 0;

  for (let i = 0; i < Math.min(rounds, nodes.length); i++) {
    const node = nodes[Math.floor(Math.random() * nodes.length)];
    const prompt = node.label || node.id.replace(/_/g, ' ');

    try {
      const generated = model.generate(prompt, 20, 0.7);

      if (!generated) continue;

      const quality = scoreGeneration(generated, vocab);

      if (quality >= 0.5) {
        // Boa geração: treinar nela
        const fullSeq = `${prompt} ${generated}`;
        const ids = vocab.tokenize(fullSeq);
        if (ids.length >= 3) {
          allPairs.push(...makeTrainingPairs(ids));
          goodGenerations++;
        }
      }
    } catch (err) {
      // Continua para o próximo
    }
  }

  if (goodGenerations > 0) {
    console.log(`[DATASET] Self-play: ${goodGenerations}/${rounds} gerações aprovadas`);
  }

  return allPairs;
}

/**
 * Pontua qualidade de uma geração.
 * Retorna 0.0–1.0. Threshold para uso: >= 0.5
 */
function scoreGeneration(text, vocab) {
  if (!text || text.length < 5) return 0;

  const tokens = tokenizeText(text);
  if (tokens.length < 2) return 0;

  // Penalizar repetição de tokens
  const unique = new Set(tokens);
  const diversityScore = unique.size / tokens.length;

  // Percentual de tokens conhecidos no vocabulário
  const knownCount = tokens.filter(t => vocab.token2id[t] !== undefined).length;
  const knownScore = knownCount / tokens.length;

  // Comprimento ideal: 5–30 tokens
  const lenScore = tokens.length >= 5 && tokens.length <= 30 ? 1.0 : 0.5;

  return (diversityScore * 0.4 + knownScore * 0.4 + lenScore * 0.2);
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
