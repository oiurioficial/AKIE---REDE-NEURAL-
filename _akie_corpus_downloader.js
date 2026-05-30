/**
 * _akie_corpus_downloader.js v2.0
 *
 * Corpus PT-BR via Wikipedia REST API — corrigido para VPS.
 *
 * CORREÇÕES v2.0:
 *   - fetchText reescrito: sem assumir gzip, usa https.get simples
 *   - Accept-Encoding: identity (força resposta sem compressão)
 *   - User-Agent explícito (Wikipedia rejeita requests sem UA)
 *   - 80 tópicos PT-BR em vez de 10
 *   - Processamento de frases mais robusto
 *   - Meta: 3.000+ frases únicas só do Wikipedia
 *
 * USO:
 *   node _akie_corpus_downloader.js
 */

const https  = require('https');
const http   = require('http');
const fs     = require('fs');
const path   = require('path');

const CORPUS_DIR = process.env.CORPUS_DIR || '/opt/akie/corpus';
fs.mkdirSync(CORPUS_DIR, { recursive: true });

// ---------------------------------------------------------------------------
// HTTP fetch — sem gzip, com User-Agent, com redirect manual
// ---------------------------------------------------------------------------

function fetchJSON(url, timeoutMs = 20000) {
  return new Promise((resolve, reject) => {
    const parsedUrl = new URL(url);
    const options = {
      hostname: parsedUrl.hostname,
      path:     parsedUrl.pathname + parsedUrl.search,
      method:   'GET',
      timeout:  timeoutMs,
      headers: {
        'User-Agent':      'AKIE-Corpus-Builder/2.0 (educational project)',
        'Accept':          'application/json',
        'Accept-Encoding': 'identity',  // sem compressão — evita bug do gzip
      },
    };

    const proto = url.startsWith('https') ? https : http;
    const req = proto.request(options, res => {
      // Seguir redirect manualmente
      if (res.statusCode === 301 || res.statusCode === 302 || res.statusCode === 307) {
        res.resume();
        return fetchJSON(res.headers.location, timeoutMs).then(resolve).catch(reject);
      }
      if (res.statusCode !== 200) {
        res.resume();
        return reject(new Error(`HTTP ${res.statusCode}`));
      }

      const chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => {
        try {
          const text = Buffer.concat(chunks).toString('utf8');
          resolve(JSON.parse(text));
        } catch (e) {
          reject(new Error(`JSON parse error: ${e.message}`));
        }
      });
      res.on('error', reject);
    });

    req.on('error', reject);
    req.on('timeout', () => { req.destroy(); reject(new Error('Timeout')); });
    req.end();
  });
}

// ---------------------------------------------------------------------------
// Tópicos PT-BR — 80 artigos Wikipedia
// ---------------------------------------------------------------------------

const WIKIPEDIA_TOPICS = [
  // Geografia BR
  'Brasil', 'São_Paulo', 'Rio_de_Janeiro_(cidade)', 'Brasília', 'Salvador',
  'Fortaleza', 'Manaus', 'Curitiba', 'Recife', 'Porto_Alegre',
  'Belém_(Pará)', 'Goiânia', 'Florianópolis', 'Maceió', 'Natal_(Rio_Grande_do_Norte)',
  'Amazônia', 'Rio_Amazonas', 'Pantanal', 'Cerrado', 'Mata_Atlântica',
  // História BR
  'História_do_Brasil', 'Independência_do_Brasil', 'República_Velha',
  'Era_Vargas', 'Ditadura_militar_no_Brasil',
  // Cultura BR
  'Carnaval_do_Brasil', 'Futebol_no_Brasil', 'Música_popular_brasileira',
  'Capoeira', 'Samba', 'Forró', 'Axé_(música)', 'Bossa_nova',
  'Feijoada', 'Churrasco', 'Caipirinha', 'Brigadeiro_(doce)',
  // Língua portuguesa
  'Língua_portuguesa', 'Portugal', 'Angola', 'Moçambique',
  'Cabo_Verde', 'Literatura_brasileira', 'Machado_de_Assis',
  // Ciência e tecnologia
  'Inteligência_artificial', 'Aprendizado_de_máquina', 'Rede_neural_artificial',
  'Computador', 'Internet', 'Programação_de_computadores', 'Algoritmo',
  'Robótica', 'Física', 'Química', 'Biologia', 'Matemática',
  'Astronomia', 'Medicina', 'Genética',
  // Sociedade
  'Educação', 'Economia', 'Democracia', 'Direitos_humanos',
  'Meio_ambiente', 'Sustentabilidade', 'Energia_solar',
  // Animais e natureza
  'Onça-pintada', 'Tucano', 'Arara', 'Boto-cor-de-rosa',
  'Vitória-régia', 'Ipê', 'Açaí',
  // Filosofia e arte
  'Filosofia', 'Arte', 'Cinema', 'Literatura', 'Música',
  'Pintura', 'Arquitetura', 'Teatro',
  // Esportes
  'Futebol', 'Vôlei', 'Basquete', 'Tênis', 'Natação',
  // Outros
  'Saúde', 'Nutrição', 'Psicologia', 'Comunicação', 'Jornalismo',
];

// ---------------------------------------------------------------------------
// Corpus embutido (fallback garantido)
// ---------------------------------------------------------------------------

const EMBEDDED_SENTENCES = [
  'sou a AKIE, uma inteligência artificial.', 'meu nome é AKIE.', 'me chamo AKIE.',
  'sou a AKIE, parte do ecossistema AETHER.', 'aprendo com cada conversa.',
  'estou sempre aprendendo e melhorando.', 'cada interação me ensina algo novo.',
  'olá, como vai?', 'bom dia, tudo bem?', 'boa tarde, como posso ajudar?',
  'boa noite, tudo certo?', 'oi, tudo bem?', 'como você está?',
  'sim, com certeza.', 'não, obrigada.', 'claro, pode perguntar.',
  'com prazer.', 'sem problema.', 'tudo certo.', 'entendido.',
  'não tenho certeza.', 'talvez.', 'possivelmente.', 'não sei ao certo.',
  'é uma boa pergunta.', 'deixa eu pensar.',
  'brasília é a capital do brasil.', 'o brasil fica na américa do sul.',
  'a língua oficial do brasil é o português.', 'são paulo é a maior cidade do brasil.',
  'o rio amazonas é o maior rio do mundo.', 'a amazônia é a maior floresta tropical.',
  'o real é a moeda do brasil.', 'o brasil tem duzentos e quinze milhões de habitantes.',
  'portugal fica na europa.', 'o português é falado em vários países.',
  'o futebol é o esporte mais popular do brasil.',
  'o carnaval é uma festa muito popular no brasil.',
  'inteligência artificial é a capacidade de máquinas aprenderem.',
  'redes neurais são inspiradas no cérebro humano.',
  'o aprendizado de máquina usa dados para melhorar.',
  'embeddings representam palavras como vetores numéricos.',
  'o vocabulário define quais palavras o modelo conhece.',
  'eu falo português.', 'você fala muito bem.', 'ele estuda bastante.',
  'nós aprendemos juntos.', 'eu gosto de aprender.', 'você precisa de ajuda?',
  'quando chove, fico em casa.', 'se você precisar, pode me perguntar.',
  'embora seja difícil, vou tentar.', 'porque aprendo, consigo melhorar.',
  'apesar de não saber tudo, faço o meu melhor.',
  'aprender é um processo contínuo.', 'cada erro é uma oportunidade de melhora.',
  'a prática leva à perfeição.', 'o conhecimento é poder.',
  'um mais um é igual a dois.', 'há sete dias na semana.',
  'há doze meses no ano.', 'o primeiro mês do ano é janeiro.',
];

const EMBEDDED_DIALOGS = [
  { u: 'quem é você?', a: 'sou a AKIE, uma inteligência artificial que aprende por interação.' },
  { u: 'qual é seu nome?', a: 'meu nome é AKIE.' },
  { u: 'você é o aether?', a: 'não, sou a AKIE. AETHER é o ecossistema do qual faço parte.' },
  { u: 'olá', a: 'olá! como posso ajudar?' },
  { u: 'oi', a: 'oi! tudo bem?' },
  { u: 'bom dia', a: 'bom dia! como posso ajudar hoje?' },
  { u: 'boa tarde', a: 'boa tarde! como vai?' },
  { u: 'boa noite', a: 'boa noite! como posso ajudar?' },
  { u: 'tudo bem?', a: 'sim, tudo bem! e você?' },
  { u: 'como vai?', a: 'vai bem! e você?' },
  { u: 'qual é a capital do brasil?', a: 'a capital do brasil é brasília.' },
  { u: 'o que é inteligência artificial?', a: 'é a capacidade de sistemas computacionais realizarem tarefas que requerem inteligência humana.' },
  { u: 'você pode me ajudar?', a: 'sim, posso ajudar! o que você precisa?' },
  { u: 'o que você não sabe?', a: 'há muito que ainda preciso aprender.' },
  { u: 'até logo', a: 'até! foi bom conversar.' },
  { u: 'tchau', a: 'tchau! volte quando quiser.' },
  { u: 'obrigado', a: 'de nada! fico feliz em ajudar.' },
  { u: 'valeu', a: 'de nada!' },
  { u: 'o que é o aether?', a: 'AETHER é o ecossistema do qual faço parte.' },
  { u: 'aether é uma ia?', a: 'AETHER é o ecossistema. eu sou a AKIE, a IA dentro dele.' },
];

// ---------------------------------------------------------------------------
// Extrair frases de um extrato Wikipedia
// ---------------------------------------------------------------------------

function extractSentences(text) {
  if (!text || text.length < 10) return [];

  return text
    .replace(/\n+/g, ' ')
    .replace(/\s+/g, ' ')
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim().toLowerCase())
    .filter(s => {
      if (s.length < 20 || s.length > 400) return false;
      if (!/[a-záéíóúãõâêôçà]/i.test(s)) return false;
      if (/\[|\]|\{|\}|=|<|>/.test(s)) return false; // lixo wiki
      return true;
    });
}

// ---------------------------------------------------------------------------
// Buscar Wikipedia em lote com concorrência limitada
// ---------------------------------------------------------------------------

async function fetchWikipediaBatch(topics, concurrency = 5) {
  const sentences = [];
  let done = 0;

  // Processar em chunks de `concurrency`
  for (let i = 0; i < topics.length; i += concurrency) {
    const chunk = topics.slice(i, i + concurrency);
    const results = await Promise.allSettled(
      chunk.map(async topic => {
        const url = `https://pt.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(topic)}`;
        const data = await fetchJSON(url, 15000);
        const frases = extractSentences(data.extract || '');
        return { topic, frases };
      })
    );

    for (const r of results) {
      if (r.status === 'fulfilled') {
        sentences.push(...r.value.frases);
        process.stdout.write(`  ✓ ${r.value.topic} (${r.value.frases.length} frases)\n`);
      } else {
        process.stdout.write(`  ✗ ${chunk[results.indexOf(r)]} — ${r.reason?.message}\n`);
      }
    }

    done += chunk.length;
    console.log(`[CORPUS] Progresso: ${done}/${topics.length} tópicos | ${sentences.length} frases`);

    // Pausa entre chunks para não sobrecarregar a API
    if (i + concurrency < topics.length) {
      await new Promise(r => setTimeout(r, 500));
    }
  }

  return sentences;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log('\n╔══════════════════════════════════════════════════╗');
  console.log('║   AKIE Corpus Downloader v2.0 — PT-BR            ║');
  console.log('╚══════════════════════════════════════════════════╝\n');

  // 1. Corpus embutido
  let allSentences = [...EMBEDDED_SENTENCES];
  console.log(`[CORPUS] Embedded: ${EMBEDDED_SENTENCES.length} frases`);

  // 2. Wikipedia — 80 tópicos
  console.log(`[CORPUS] Buscando ${WIKIPEDIA_TOPICS.length} tópicos Wikipedia PT-BR...\n`);
  const wikiSentences = await fetchWikipediaBatch(WIKIPEDIA_TOPICS, 5);
  allSentences.push(...wikiSentences);
  console.log(`\n[CORPUS] Wikipedia total: ${wikiSentences.length} frases`);

  // 3. Deduplicar e filtrar
  const unique = [...new Set(
    allSentences
      .map(s => s.trim().toLowerCase())
      .filter(s => s.length > 10)
  )];
  console.log(`[CORPUS] Total únicas após dedup: ${unique.length}`);

  // 4. Salvar sentences
  const sentPath = path.join(CORPUS_DIR, 'sentences_ptbr.txt');
  fs.writeFileSync(sentPath, unique.join('\n'), 'utf8');
  console.log(`[CORPUS] ✓ Frases: ${sentPath}`);

  // 5. Salvar diálogos
  const dialogLines = EMBEDDED_DIALOGS.map(d => `u: ${d.u}\na: ${d.a}`);
  const dialogPath  = path.join(CORPUS_DIR, 'dialogs_ptbr.txt');
  fs.writeFileSync(dialogPath, dialogLines.join('\n\n'), 'utf8');
  console.log(`[CORPUS] ✓ Diálogos: ${dialogPath} (${EMBEDDED_DIALOGS.length} pares)`);

  // 6. Vocab freq
  const freq = new Map();
  for (const s of unique) {
    for (const t of s.replace(/([.,!?;:()\[\]{}"])/g, ' $1 ').split(/\s+/).filter(t => t.length >= 2)) {
      freq.set(t, (freq.get(t) || 0) + 1);
    }
  }
  const sorted = [...freq.entries()].sort((a, b) => b[1] - a[1]);
  const vocabPath = path.join(CORPUS_DIR, 'vocab_raw.txt');
  fs.writeFileSync(vocabPath, sorted.map(([t, c]) => `${t}\t${c}`).join('\n'), 'utf8');

  console.log(`\n[CORPUS] ══ Resumo Final ══`);
  console.log(`  Frases únicas:   ${unique.length}`);
  console.log(`  Diálogos:        ${EMBEDDED_DIALOGS.length} pares`);
  console.log(`  Vocab único:     ${sorted.length} tokens`);
  console.log(`  Tokens freq>2:   ${sorted.filter(([,c]) => c > 2).length}`);
  console.log(`\n[CORPUS] Próximo passo:`);
  console.log(`  nohup node _akie_pretrain.js > /opt/akie/logs/pretrain.log 2>&1 &\n`);
}

main().catch(err => {
  console.error('[CORPUS] Erro fatal:', err.message);
  process.exit(1);
});
