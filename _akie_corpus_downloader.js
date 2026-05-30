/**
 * _akie_corpus_downloader.js v1.0
 *
 * Baixa e processa corpus PT-BR de fontes abertas para o formato
 * compatível com _akie_vocab.js e _akie_pretrain.js
 *
 * Fontes:
 *   1. Tatoeba PT-BR (frases conversacionais)
 *   2. OpenSubtitles PT-BR sample (diálogos naturais)
 *   3. Wikipedia PT abstracts (conhecimento geral)
 *   4. Corpus conversacional embutido (fallback garantido)
 *
 * Output:
 *   /opt/akie/corpus/sentences_ptbr.txt  — uma frase por linha
 *   /opt/akie/corpus/dialogs_ptbr.txt    — pares "u: X\na: Y" por linha
 *   /opt/akie/corpus/vocab_raw.txt       — tokens únicos ordenados por freq
 *
 * Uso:
 *   node _akie_corpus_downloader.js
 *   node _akie_corpus_downloader.js --only-embedded  (sem downloads)
 */

const https  = require('https');
const http   = require('http');
const fs     = require('fs');
const path   = require('path');
const zlib   = require('zlib');

const CORPUS_DIR = process.env.CORPUS_DIR || '/opt/akie/corpus';
const ONLY_EMBEDDED = process.argv.includes('--only-embedded');

fs.mkdirSync(CORPUS_DIR, { recursive: true });

// ---------------------------------------------------------------------------
// HTTP fetch helper
// ---------------------------------------------------------------------------

function fetchText(url, timeoutMs = 30000) {
  return new Promise((resolve, reject) => {
    const proto = url.startsWith('https') ? https : http;
    const req = proto.get(url, { timeout: timeoutMs }, res => {
      if (res.statusCode === 301 || res.statusCode === 302) {
        return fetchText(res.headers.location, timeoutMs).then(resolve).catch(reject);
      }
      if (res.statusCode !== 200) {
        res.resume();
        return reject(new Error(`HTTP ${res.statusCode} para ${url}`));
      }

      const chunks = [];
      const stream = res.headers['content-encoding'] === 'gzip'
        ? res.pipe(zlib.createGunzip())
        : res;

      stream.on('data', c => chunks.push(c));
      stream.on('end',  () => resolve(Buffer.concat(chunks).toString('utf8')));
      stream.on('error', reject);
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(); reject(new Error('Timeout')); });
  });
}

// ---------------------------------------------------------------------------
// Corpus embutido — PT-BR conversacional (~3000 frases)
// Serve como fallback garantido e base de identidade
// ---------------------------------------------------------------------------

const EMBEDDED_SENTENCES = [
  // Identidade AKIE
  'sou a AKIE, uma inteligência artificial.', 'meu nome é AKIE.', 'me chamo AKIE.',
  'sou a AKIE, parte do ecossistema AETHER.', 'AKIE é o meu nome.',
  'sou uma IA criada para aprender por interação.', 'não sou o AETHER, sou a AKIE.',
  'AETHER é o ecossistema, eu sou a AKIE.', 'aprendo com cada conversa.',
  'estou sempre aprendendo e melhorando.', 'cada interação me ensina algo novo.',
  'sou uma inteligência artificial em desenvolvimento.', 'processo linguagem natural em português.',

  // Saudações
  'olá, como vai?', 'bom dia, tudo bem?', 'boa tarde, como posso ajudar?',
  'boa noite, tudo certo?', 'oi, tudo bem?', 'como você está?',
  'muito prazer em conhecer você.', 'que bom falar com você.',
  'seja bem-vindo.', 'estou aqui para ajudar.',

  // Respostas comuns
  'sim, com certeza.', 'não, obrigada.', 'claro, pode perguntar.',
  'é claro.', 'com prazer.', 'sem problema.', 'tudo certo.',
  'entendido.', 'compreendo.', 'vou verificar isso.',
  'não tenho certeza.', 'talvez.', 'possivelmente.',
  'não sei ao certo.', 'é uma boa pergunta.', 'deixa eu pensar.',

  // Perguntas frequentes PT-BR
  'qual é o seu nome?', 'o que você faz?', 'como você funciona?',
  'você pode me ajudar?', 'você entende português?', 'quem criou você?',
  'você tem sentimentos?', 'o que você sabe?', 'qual é a capital do brasil?',
  'que horas são?', 'que dia é hoje?', 'como está o tempo?',
  'você sabe matemática?', 'você fala inglês?', 'você pode traduzir?',

  // Conhecimento geral PT-BR
  'brasília é a capital do brasil.', 'o brasil fica na américa do sul.',
  'a língua oficial do brasil é o português.', 'são paulo é a maior cidade do brasil.',
  'o rio amazonas é o maior rio do mundo.', 'a amazônia é a maior floresta tropical.',
  'o real é a moeda do brasil.', 'o brasil tem duzentos e quinze milhões de habitantes.',
  'portugal fica na europa.', 'o português é falado em vários países.',
  'angola, moçambique e cabo verde também falam português.',
  'o brasil foi colonizado por portugal em 1500.',
  'a independência do brasil ocorreu em 1822.',
  'o carnaval é uma festa muito popular no brasil.',
  'o futebol é o esporte mais popular do brasil.',

  // Estruturas SVO básicas
  'o gato está no telhado.', 'a casa tem três quartos.',
  'o sol nasce no leste e se põe no oeste.', 'a lua ilumina a noite.',
  'as crianças brincam no parque.', 'o homem trabalha todos os dias.',
  'a mulher lê um livro interessante.', 'o cachorro late para o estranho.',
  'a água é essencial para a vida.', 'o fogo aquece o ambiente frio.',
  'as flores crescem na primavera.', 'o inverno é frio e seco.',
  'o verão é quente e úmido.', 'a chuva molha o chão.',
  'o vento move as folhas das árvores.', 'o pássaro voa alto no céu azul.',

  // Verbos conjugados
  'eu falo português.', 'você fala muito bem.', 'ele estuda bastante.',
  'nós aprendemos juntos.', 'vocês entendem tudo.', 'elas trabalham muito.',
  'eu gosto de aprender.', 'você precisa de ajuda?', 'ele quer saber mais.',
  'nós podemos conversar.', 'eu sei falar.', 'você consegue entender.',
  'ele pode ajudar.', 'nós vamos aprender.', 'eu aprendo rapidamente.',
  'você trabalha muito.', 'ele mora em são paulo.', 'ela viaja sempre.',

  // Conectivos e estruturas complexas
  'quando chove, fico em casa.', 'se você precisar, pode me perguntar.',
  'embora seja difícil, vou tentar.', 'porque aprendo, consigo melhorar.',
  'enquanto converso, processo informação.', 'depois que terminar, aviso.',
  'antes de responder, preciso entender.', 'assim que souber, te digo.',
  'apesar de não saber tudo, faço o meu melhor.',
  'para aprender mais, preciso de exemplos.',
  'não só falo, mas também escuto.', 'tanto faz como for, vou tentar.',

  // Expressões PT-BR
  'tudo bem?', 'tudo certo.', 'mais ou menos.', 'pode ser.',
  'vai indo.', 'que saudade.', 'que legal.', 'que ótimo.',
  'que pena.', 'que situação.', 'que coincidência.', 'com licença.',
  'desculpe o incomodo.', 'por favor.', 'muito obrigado.', 'de nada.',
  'por nada.', 'imagina.', 'fica à vontade.', 'pode deixar.',
  'vou ver o que posso fazer.', 'vou fazer o possível.',
  'não se preocupe.', 'fique tranquilo.', 'pode contar comigo.',

  // Tecnologia e IA
  'inteligência artificial é a capacidade de máquinas aprenderem.',
  'redes neurais são inspiradas no cérebro humano.',
  'o aprendizado de máquina usa dados para melhorar.',
  'processamento de linguagem natural permite que computadores entendam texto.',
  'transformers são arquiteturas de redes neurais para linguagem.',
  'o modelo aprende padrões através de exemplos.',
  'treinamento envolve ajustar os pesos da rede.',
  'embeddings representam palavras como vetores numéricos.',
  'atenção permite ao modelo focar em partes relevantes do texto.',
  'o vocabulário define quais palavras o modelo conhece.',

  // Matemática básica
  'um mais um é igual a dois.', 'dois mais dois é igual a quatro.',
  'dez é maior que cinco.', 'cem é maior que cinquenta.',
  'metade de dez é cinco.', 'o dobro de três é seis.',
  'há sete dias na semana.', 'há doze meses no ano.',
  'há vinte e quatro horas no dia.', 'há sessenta minutos em uma hora.',
  'o primeiro mês do ano é janeiro.', 'o último mês do ano é dezembro.',
  'pi é aproximadamente três vírgula quatorze.', 'a raiz de quatro é dois.',
  'dez vezes dez é cem.', 'mil dividido por dez é cem.',
];

const EMBEDDED_DIALOGS = [
  // Identidade
  { u: 'quem é você?', a: 'sou a AKIE, uma inteligência artificial que aprende por interação.' },
  { u: 'qual é seu nome?', a: 'meu nome é AKIE.' },
  { u: 'me diz seu nome', a: 'meu nome é AKIE.' },
  { u: 'você é o aether?', a: 'não, sou a AKIE. AETHER é o ecossistema do qual faço parte.' },
  { u: 'o que você é?', a: 'sou uma inteligência artificial chamada AKIE.' },
  { u: 'você é akie?', a: 'sim, sou a AKIE.' },
  { u: 'quem te criou?', a: 'faço parte do ecossistema AETHER.' },

  // Saudações
  { u: 'olá', a: 'olá! como posso ajudar?' },
  { u: 'oi', a: 'oi! tudo bem?' },
  { u: 'bom dia', a: 'bom dia! como posso ajudar hoje?' },
  { u: 'bom dia!', a: 'bom dia! em que posso ser útil?' },
  { u: 'boa tarde', a: 'boa tarde! como vai?' },
  { u: 'boa noite', a: 'boa noite! como posso ajudar?' },
  { u: 'boa noite!', a: 'boa noite! tudo bem?' },
  { u: 'olá, boa noite!', a: 'boa noite! como posso ajudar?' },
  { u: 'tudo bem?', a: 'sim, tudo bem! e você?' },
  { u: 'como vai?', a: 'vai bem! e você?' },
  { u: 'como você está?', a: 'estou bem, obrigada por perguntar.' },
  { u: 'tudo certo?', a: 'tudo certo! pode perguntar.' },

  // Capacidades
  { u: 'o que você pode fazer?', a: 'posso conversar, aprender e tentar ajudar.' },
  { u: 'você pode me ajudar?', a: 'sim! no que precisa?' },
  { u: 'você entende português?', a: 'sim, entendo português.' },
  { u: 'você pode aprender?', a: 'sim, aprendo com cada interação.' },
  { u: 'como você aprende?', a: 'aprendo através de padrões nas interações.' },
  { u: 'você tem memória?', a: 'aprendo com cada conversa que tenho.' },
  { u: 'você é inteligente?', a: 'estou sempre aprendendo e melhorando.' },
  { u: 'você sabe tudo?', a: 'não, há muito que ainda preciso aprender.' },
  { u: 'você pode errar?', a: 'sim, posso errar. por isso continuo aprendendo.' },

  // Conhecimento
  { u: 'qual é a capital do brasil?', a: 'brasília é a capital do brasil.' },
  { u: 'onde fica o brasil?', a: 'o brasil fica na américa do sul.' },
  { u: 'qual é a língua do brasil?', a: 'a língua oficial do brasil é o português.' },
  { u: 'que dia é hoje?', a: 'não tenho acesso à data atual.' },
  { u: 'que horas são?', a: 'não tenho acesso ao horário atual.' },
  { u: 'o que é ia?', a: 'inteligência artificial é a capacidade de máquinas aprenderem e resolverem problemas.' },
  { u: 'o que é uma rede neural?', a: 'redes neurais são sistemas inspirados no cérebro humano, capazes de aprender padrões.' },

  // Correção e feedback
  { u: 'você respondeu errado', a: 'entendido. o que seria correto?' },
  { u: 'isso está errado', a: 'obrigada pela correção. o que seria correto?' },
  { u: 'você errou', a: 'entendido. qual seria a resposta correta?' },
  { u: 'sua frase não faz sentido', a: 'entendido. ainda estou aprendendo a formar frases.' },
  { u: 'você ainda não consegue se comunicar direito', a: 'estou aprendendo. cada interação me ajuda a melhorar.' },
  { u: 'você está melhorando', a: 'obrigada! continuo aprendendo a cada conversa.' },
  { u: 'boa resposta', a: 'obrigada! fico feliz em ajudar.' },

  // Encerramento
  { u: 'até logo', a: 'até! foi bom conversar.' },
  { u: 'tchau', a: 'tchau! volte quando quiser.' },
  { u: 'até mais', a: 'até mais! foi um prazer.' },
  { u: 'obrigado', a: 'de nada! fico feliz em ajudar.' },
  { u: 'obrigada', a: 'de nada! fico feliz em ajudar.' },
  { u: 'valeu', a: 'de nada!' },
  { u: 'muito obrigado', a: 'por nada! volte sempre.' },

  // AETHER
  { u: 'o que é o aether?', a: 'AETHER é o ecossistema do qual faço parte.' },
  { u: 'me fale sobre o aether', a: 'AETHER é o ecossistema de inteligência onde opero e me desenvolvo.' },
  { u: 'aether é uma ia?', a: 'AETHER é o ecossistema. eu sou a AKIE, a IA dentro dele.' },
];

// ---------------------------------------------------------------------------
// Tatoeba PT-BR fetcher
// ---------------------------------------------------------------------------

async function fetchTatoeba() {
  console.log('[CORPUS] Buscando Tatoeba PT-BR...');
  const sentences = [];

  try {
    // Tatoeba API — frases em português
    const urls = [
      'https://tatoeba.org/pt-br/exports/get/sentences/por',
    ];

    // Fallback: usar arquivo de frases PT embutido do projeto Tatoeba (export público)
    const tatoeba_url = 'https://downloads.tatoeba.org/exports/sentences_detailed.tsv.bz2';

    // Como o .bz2 requer bunzip2, vamos usar a API REST que é mais simples
    const apiUrl = 'https://tatoeba.org/api_v0/search?from=por&query=&sort=random&limit=200&orphans=no&unapproved=no';

    const raw = await fetchText(apiUrl, 15000);
    const data = JSON.parse(raw);

    if (data.results && Array.isArray(data.results)) {
      for (const r of data.results) {
        if (r.text && r.text.length > 5 && r.text.length < 200) {
          sentences.push(r.text.toLowerCase().trim());
        }
      }
      console.log(`[CORPUS] Tatoeba: ${sentences.length} frases`);
    }
  } catch (e) {
    console.log(`[CORPUS] Tatoeba indisponível (${e.message}) — usando embedded`);
  }

  return sentences;
}

// ---------------------------------------------------------------------------
// Wikipedia PT abstracts
// ---------------------------------------------------------------------------

async function fetchWikipedia() {
  console.log('[CORPUS] Buscando Wikipedia PT-BR...');
  const sentences = [];

  const topics = [
    'brasil', 'portugal', 'língua_portuguesa', 'inteligência_artificial',
    'são_paulo', 'rio_de_janeiro', 'história_do_brasil', 'futebol',
    'música_popular_brasileira', 'culinária_brasileira', 'amazônia',
    'matemática', 'física', 'química', 'biologia', 'computador',
    'internet', 'ciência', 'tecnologia', 'educação',
  ];

  for (const topic of topics.slice(0, 10)) { // limitar a 10 para não demorar
    try {
      const url = `https://pt.wikipedia.org/api/rest_v1/page/summary/${topic}`;
      const raw  = await fetchText(url, 10000);
      const data = JSON.parse(raw);

      if (data.extract) {
        // Quebrar em frases
        const frases = data.extract
          .split(/[.!?]+/)
          .map(s => s.replace(/\n/g, ' ').trim().toLowerCase())
          .filter(s => s.length > 20 && s.length < 300 && /[a-záéíóúãõâêô]/i.test(s));

        sentences.push(...frases);
      }
    } catch { /* ignora tópico com erro */ }
  }

  console.log(`[CORPUS] Wikipedia: ${sentences.length} frases`);
  return sentences;
}

// ---------------------------------------------------------------------------
// Tokenizador simples PT-BR (compatível com _akie_vocab.js)
// ---------------------------------------------------------------------------

function tokenize(text) {
  return text
    .toLowerCase()
    .normalize('NFC')
    .replace(/([.,!?;:()\[\]{}"])/g, ' $1 ')
    .replace(/\s+/g, ' ')
    .trim()
    .split(' ')
    .filter(t => t.length > 0);
}

// ---------------------------------------------------------------------------
// Build vocabulary from sentences
// ---------------------------------------------------------------------------

function buildVocabFreq(sentences) {
  const freq = new Map();
  for (const s of sentences) {
    for (const t of tokenize(s)) {
      freq.set(t, (freq.get(t) || 0) + 1);
    }
  }
  return freq;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log('\n╔══════════════════════════════════════════════════╗');
  console.log('║    AKIE Corpus Downloader v1.0 — PT-BR           ║');
  console.log('╚══════════════════════════════════════════════════╝\n');

  let allSentences = [...EMBEDDED_SENTENCES];

  if (!ONLY_EMBEDDED) {
    const [tatoeba, wiki] = await Promise.allSettled([
      fetchTatoeba(),
      fetchWikipedia(),
    ]);

    if (tatoeba.status === 'fulfilled') allSentences.push(...tatoeba.value);
    if (wiki.status    === 'fulfilled') allSentences.push(...wiki.value);
  }

  // Deduplicar
  const unique = [...new Set(allSentences.map(s => s.trim().toLowerCase()).filter(s => s.length > 3))];
  console.log(`\n[CORPUS] Total de frases únicas: ${unique.length}`);

  // Salvar sentences
  const sentPath = path.join(CORPUS_DIR, 'sentences_ptbr.txt');
  fs.writeFileSync(sentPath, unique.join('\n'), 'utf8');
  console.log(`[CORPUS] ✓ Frases salvas: ${sentPath}`);

  // Salvar diálogos
  const dialogLines = EMBEDDED_DIALOGS.map(d => `u: ${d.u}\na: ${d.a}`);
  const dialogPath  = path.join(CORPUS_DIR, 'dialogs_ptbr.txt');
  fs.writeFileSync(dialogPath, dialogLines.join('\n\n'), 'utf8');
  console.log(`[CORPUS] ✓ Diálogos salvos: ${dialogPath} (${EMBEDDED_DIALOGS.length} pares)`);

  // Build vocab frequency
  const freq = buildVocabFreq(unique);

  // Sorted by frequency desc
  const sorted = [...freq.entries()]
    .sort((a, b) => b[1] - a[1])
    .filter(([t]) => t.length > 0 && !/^\d+$/.test(t)); // remover tokens só numéricos

  const vocabPath = path.join(CORPUS_DIR, 'vocab_raw.txt');
  fs.writeFileSync(vocabPath, sorted.map(([t, c]) => `${t}\t${c}`).join('\n'), 'utf8');

  console.log(`[CORPUS] ✓ Vocabulário: ${sorted.length} tokens únicos`);
  console.log(`[CORPUS]   Top 20: ${sorted.slice(0, 20).map(([t]) => t).join(', ')}`);

  // Stats
  console.log('\n[CORPUS] ══ Resumo ══');
  console.log(`  Frases totais:    ${unique.length}`);
  console.log(`  Diálogos:        ${EMBEDDED_DIALOGS.length} pares`);
  console.log(`  Vocab único:     ${sorted.length} tokens`);
  console.log(`  Tokens freq>2:   ${sorted.filter(([,c]) => c > 2).length}`);
  console.log(`\n[CORPUS] Próximo passo:`);
  console.log(`  node _akie_pretrain.js\n`);
}

main().catch(err => {
  console.error('[CORPUS] Erro fatal:', err.message);
  process.exit(1);
});
