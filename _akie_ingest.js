/**
 * _akie_ingest.js  —  AKIE Data Engine (Ingestão Autônoma) v3.0
 *
 * MUDANÇAS v3.0:
 *   - Wikipedia REMOVIDA permanentemente (poluía grafo com texto enciclopédico formal)
 *   - Foco 100% em Tatoeba PT/EN + banco de frases conversacionais nativas
 *   - isValidSentence afrouxada: mínimo 5 chars, aceita gírias/contrações de chat
 *   - Verbos expandidos com contrações coloquiais (tô, tá, vou, blz, vlw, tem...)
 *
 * Pipeline:
 *   carregarFila → extrairFrases → filtrar → tokenizar → treinarPares
 *
 * RESTRIÇÕES:
 *   - Zero texto bruto armazenado após processamento
 *   - Zero dependências externas (apenas Node.js nativo)
 */

'use strict';

const https  = require('https');
const http   = require('http');
const fs     = require('fs');
const path   = require('path');
const stream = require('stream');

const { tokenizeText, makeTrainingPairs } = require('./_akie_vocab');

// ---------------------------------------------------------------------------
// Configuração
// ---------------------------------------------------------------------------

const INGEST_DIR      = process.env.INGEST_DIR || '/data/datasets';
const BATCH_SIZE      = parseInt(process.env.INGEST_BATCH || '1000', 10);
const MIN_CHARS       = 5;   // Reduzido de 10 → aceita "oi!", "blz", etc.
const MAX_WORDS       = 30;  // Levemente aumentado para frases naturais
const MAX_QUEUE_SIZE  = 50_000;

// Arquivos de controle
const TATOEBA_CURSOR_FILE  = path.join(INGEST_DIR, 'tatoeba_cursor.json');
const CONV_CURSOR_FILE     = path.join(INGEST_DIR, 'conv_cursor.json');
const INGEST_STATS_FILE    = path.join(INGEST_DIR, 'ingest_stats.json');

// ---------------------------------------------------------------------------
// Banco de frases conversacionais nativas (PT-BR coloquial)
// Usado como fonte primária quando Tatoeba não está disponível,
// e também intercalado como reforço conversacional.
// ---------------------------------------------------------------------------

const CONVERSATIONAL_SEED = [
  // Saudações e cumprimentos
  'oi, tudo bem?',
  'olá, como vai?',
  'tudo certo por aí?',
  'e aí, sumido?',
  'boa tarde, como você está?',
  'bom dia, dormiu bem?',
  'boa noite, tá tudo bem?',
  'oi oi, como foi o dia?',
  'fala, como tá?',
  'oi, tô aqui sim',
  // Respostas de saudação
  'tô bem, obrigado',
  'tudo ótimo, e você?',
  'tá ótimo por aqui',
  'tô indo, valeu por perguntar',
  'tá tranquilo, e aí?',
  'tô bem sim, e você?',
  'tudo certo, e contigo?',
  'aqui tá bem, obrigado',
  'blz, e você?',
  'vlw, tô bem',
  // Expressões de concordância e discordância
  'com certeza, faz sentido',
  'não concordo muito com isso',
  'acho que você tem razão',
  'hmm, não sei não',
  'pode ser, mas depende',
  'exatamente isso que eu pensei',
  'verdade, faz todo sentido',
  'não sei se é bem assim',
  'faz sentido pra mim',
  'talvez, depende do caso',
  // Perguntas cotidianas
  'o que você acha disso?',
  'como assim você não sabe?',
  'qual é a sua opinião?',
  'você conseguiu resolver?',
  'precisa de ajuda com alguma coisa?',
  'o que você quer fazer hoje?',
  'tem alguma dúvida?',
  'como posso te ajudar?',
  'o que aconteceu?',
  'pode me explicar melhor?',
  // Respostas a perguntas
  'não sei ao certo',
  'deixa eu pensar um pouco',
  'vou verificar isso pra você',
  'sim, com certeza',
  'não, não é bem assim',
  'deixa eu te explicar melhor',
  'posso te ajudar com isso',
  'precisa de mais detalhes',
  'vou te ajudar agora',
  'claro, pode perguntar',
  // Expressões de dúvida
  'não entendi bem o que você quis dizer',
  'pode repetir, por favor?',
  'não ficou claro pra mim',
  'você pode explicar de outro jeito?',
  'o que exatamente você precisa?',
  'não tenho certeza do que entendi',
  'pode dar um exemplo?',
  'não sei ao certo o que você quer',
  // Conversas sobre o dia a dia
  'hoje foi um dia bem corrido',
  'tô com sono, fui dormir tarde',
  'preciso tomar um café agora',
  'tô com fome, vou comer algo',
  'fui trabalhar de manhã cedo',
  'tive reunião o dia todo',
  'hoje choveu muito aqui',
  'tô em casa hoje',
  'saí mais cedo do trabalho',
  'tô estudando essa semana',
  // Expressões emocionais
  'que ótima notícia',
  'que chato isso',
  'fiquei feliz quando soube',
  'que situação difícil',
  'tô animado com isso',
  'me preocupei com essa situação',
  'que alívio saber disso',
  'tô surpreso com essa novidade',
  'que tristeza isso',
  'fiquei contente em ouvir',
  // Pedidos e solicitações
  'pode me ajudar com uma coisa?',
  'preciso de uma dica rápida',
  'você sabe como fazer isso?',
  'tem como me explicar?',
  'pode me dar um exemplo?',
  'o que você recomenda?',
  'qual é a melhor opção?',
  'como você faria isso?',
  'você pode verificar pra mim?',
  'qual seria a melhor abordagem?',
  // Despedidas
  'até mais, foi bom conversar',
  'tchau, cuida-se',
  'até logo, valeu',
  'flw, qualquer coisa me fala',
  'até depois, boa sorte',
  'abraço, até amanhã',
  'boa noite, descanse bem',
  'até mais, obrigado pela ajuda',
  'tchau tchau',
  'flw, foi bom conversar',
  // Gírias e linguagem informal
  'cara, que situação estranha',
  'mano, não acredito nisso',
  'véi, tô chocado',
  'que demais, adorei a ideia',
  'sério mesmo que aconteceu isso?',
  'cara, que situação complicada',
  'mano, como assim?',
  'não tô acreditando nisso',
  'que coisa louca',
  'haha, que situação',
  // Tecnologia e digital
  'o aplicativo tá fora do ar',
  'meu celular tá com problema',
  'não tô conseguindo acessar o site',
  'a internet tá lenta aqui',
  'preciso atualizar o sistema',
  'o arquivo não abre',
  'tô tentando instalar mas não funciona',
  'tem como recuperar os dados?',
  'o computador travou de novo',
  'preciso de ajuda com o código',
  // Feedback e avaliação
  'achei muito útil essa informação',
  'isso me ajudou bastante',
  'não foi bem o que eu esperava',
  'superou minhas expectativas',
  'gostei muito da resposta',
  'ainda tenho dúvidas sobre isso',
  'ficou mais claro agora',
  'entendi perfeitamente',
  'ainda não ficou claro pra mim',
  'obrigado, isso resolveu meu problema',
];

// ---------------------------------------------------------------------------
// Fontes (apenas Tatoeba + Conversacional Nativo)
// ---------------------------------------------------------------------------

const SOURCES = {
  TATOEBA: {
    name:   'tatoeba',
    url:    'https://downloads.tatoeba.org/exports/sentences.csv',
    file:   path.join(INGEST_DIR, 'tatoeba_sentences.csv'),
    parsed: path.join(INGEST_DIR, 'tatoeba_sentences_pt.txt'),
    cursor: TATOEBA_CURSOR_FILE,
    lang:   'pt',
    maxAge: 7 * 24 * 60 * 60 * 1000,
  },
  CONVERSATIONAL: {
    name:   'conversational',
    url:    null, // sem download externo — banco local
    file:   null,
    parsed: path.join(INGEST_DIR, 'conversational_pt.txt'),
    cursor: CONV_CURSOR_FILE,
    lang:   'pt',
    maxAge: Infinity, // não expira
  },
};

// ---------------------------------------------------------------------------
// Estado interno
// ---------------------------------------------------------------------------

const ingestState = {
  initialized: false,
  stats: {
    total_extracted: 0,
    total_valid:     0,
    total_inserted:  0,
    last_run:        null,
    sources:         {},
  },
};

// ---------------------------------------------------------------------------
// UTILS — I/O e HTTP
// ---------------------------------------------------------------------------

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function readCursor(file) {
  try {
    if (fs.existsSync(file)) return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (e) { /* ignora */ }
  return { byteOffset: 0, linesRead: 0, done: false };
}

function saveCursor(file, cursor) {
  try { fs.writeFileSync(file, JSON.stringify(cursor)); }
  catch (e) { /* não crítico */ }
}

function saveStats() {
  try {
    fs.writeFileSync(INGEST_STATS_FILE, JSON.stringify(ingestState.stats, null, 2));
  } catch (e) { /* não crítico */ }
}

function loadStats() {
  try {
    if (fs.existsSync(INGEST_STATS_FILE)) {
      ingestState.stats = JSON.parse(fs.readFileSync(INGEST_STATS_FILE, 'utf8'));
    }
  } catch (e) { /* ignora */ }
}

function downloadFile(url, destPath) {
  return new Promise((resolve, reject) => {
    const proto = url.startsWith('https') ? https : http;
    const file  = fs.createWriteStream(destPath + '.tmp');

    const doRequest = (targetUrl) => {
      proto.get(targetUrl, (res) => {
        if (res.statusCode === 301 || res.statusCode === 302) {
          file.close();
          fs.unlinkSync(destPath + '.tmp');
          return doRequest(res.headers.location);
        }
        if (res.statusCode !== 200) {
          file.close();
          return reject(new Error(`HTTP ${res.statusCode} ao baixar ${targetUrl}`));
        }
        res.pipe(file);
        file.on('finish', () => {
          file.close(() => {
            fs.renameSync(destPath + '.tmp', destPath);
            resolve();
          });
        });
      }).on('error', (err) => {
        file.close();
        try { fs.unlinkSync(destPath + '.tmp'); } catch (e) { /* ignora */ }
        reject(err);
      });
    };

    doRequest(url);
  });
}

function needsDownload(filePath, maxAgeMs) {
  if (!filePath) return false;
  if (!fs.existsSync(filePath)) return true;
  const stat = fs.statSync(filePath);
  return (Date.now() - stat.mtimeMs) > maxAgeMs;
}

// ---------------------------------------------------------------------------
// FILTRO DE FRASES — v3.0 (afrouxado para chat coloquial PT-BR)
// ---------------------------------------------------------------------------

// Verbos e tokens conversacionais — inclui contrações e gírias
const VERB_PATTERNS_PT = /\b(é|são|foi|foram|ser|estar|tem|têm|ter|faz|fazer|pode|poder|deve|dever|vai|vão|ir|vem|vir|usa|usar|tô|tá|tô|to|ta|vou|num|blz|vlw|né|hm|oi|olá|opa|ei|pra|pro|assim|sabe|acho|gosto|quer|quero|preciso|tenho|consigo|dá|dava|ficou|fiquei|fui|vim|sei|sabia|fica|tava|estava|eram|somos|somos|gera|significa|representa|define|indica|mostra|inclui|contém|existe|ocorre|acontece|resulta|depende|precisa|ajuda|melhora|reduz|aumenta|permite|facilita|requer|possui|apresenta|realiza|desenvolve|produz|cria|forma|afeta|explica|descreve)\b/i;
const VERB_PATTERNS_EN = /\b(is|are|was|were|be|been|has|have|had|do|does|did|can|could|will|would|may|might|shall|should|must|need|use|uses|means|defines|includes|contains|allows|requires|provides|enables|creates|forms|helps|makes|gives|takes|shows|tells|becomes|remains|appears|seems|works|runs|goes|comes|gets|keeps|leads|brings|starts|stops|ends|changes|grows|moves|happens|occurs|exists|results|depends|represents|indicates|suggests|demonstrates|explains|describes)\b/i;

// Tokens coloquiais que por si só validam a frase (dispensam verbo formal)
const COLLOQUIAL_TOKENS = /\b(oi|olá|opa|ei|né|blz|vlw|flw|haha|rsrs|kkk|ok|oks|sim|não|nao|nope|yep|yeah|ops|nossa|uau|wow|poxa|puts|caramba|vish|ué|oxe|eita|pronto|claro|exato|certo|errado|legal|top|show|massa|dahora|maneiro|demais|irado)\b/i;

/**
 * isValidSentence v3.0 — afrouxada para chat coloquial
 */
function isValidSentence(sentence) {
  if (!sentence || typeof sentence !== 'string') return false;

  const trimmed = sentence.trim();

  // Mínimo de 5 chars (aceita "oi!", "blz", etc.)
  if (trimmed.length < MIN_CHARS) return false;
  if (trimmed.length > 300) return false;

  // Remover frases com caracteres problemáticos
  if (/[<>{}[\]|\\^`~]/.test(trimmed)) return false;
  if (/https?:\/\//.test(trimmed)) return false;

  // Aceitar tokens coloquiais diretamente (dispensam validação verbal)
  if (COLLOQUIAL_TOKENS.test(trimmed)) return true;

  // Verificar presença de letras (ratio mínimo de 50% — mais permissivo)
  const alphaRatio = (trimmed.match(/[a-záéíóúãõâêôàçü]/gi) || []).length / trimmed.length;
  if (alphaRatio < 0.5) return false;

  // Pelo menos 2 tokens (palavras)
  const words = trimmed.split(/\s+/).filter(w => w.length > 0);
  if (words.length < 2) return false;
  if (words.length > MAX_WORDS) return false;

  // Presença de verbo/marcador conversacional PT ou EN
  if (VERB_PATTERNS_PT.test(trimmed) || VERB_PATTERNS_EN.test(trimmed)) return true;

  // Frases interrogativas (terminam em ?) também são válidas
  if (trimmed.endsWith('?')) return true;

  // Frases com pelo menos 4 palavras podem ser aceitas mesmo sem verbo detectado
  if (words.length >= 4) return true;

  return false;
}

/**
 * Normaliza uma frase para ingestão.
 */
function normalizeSentence(sentence) {
  return sentence
    .trim()
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .replace(/["""''«»]/g, '"')
    .replace(/[–—]/g, '-');
}

// ---------------------------------------------------------------------------
// TATOEBA — Parser CSV
// ---------------------------------------------------------------------------

/**
 * Parseia sentences.csv e extrai frases PT e EN.
 * Formato: id\tlang\ttext
 */
async function parseTatoeba(source) {
  const { file, parsed } = source;

  if (!fs.existsSync(file)) {
    console.log('[INGEST] Tatoeba: arquivo não encontrado, pulando parse.');
    return 0;
  }

  console.log('[INGEST] Tatoeba: iniciando parse CSV...');

  const content = fs.readFileSync(file, 'utf8');
  const lines   = content.split('\n');
  const out     = fs.createWriteStream(parsed, { flags: 'w' });
  let   extracted = 0;

  for (const line of lines) {
    const parts = line.split('\t');
    if (parts.length < 3) continue;

    const lang = parts[1].trim().toLowerCase();
    if (!['por', 'pt', 'eng', 'en'].includes(lang)) continue;

    const text = parts.slice(2).join('\t').trim();
    if (isValidSentence(text)) {
      out.write(normalizeSentence(text) + '\n');
      extracted++;
    }

    if (extracted > 500_000) break;
  }

  await new Promise(resolve => out.end(resolve));
  console.log(`[INGEST] Tatoeba: ${extracted} frases válidas → ${parsed}`);
  return extracted;
}

// ---------------------------------------------------------------------------
// CONVERSACIONAL NATIVO — Gera/atualiza arquivo de frases locais
// ---------------------------------------------------------------------------

/**
 * Escreve o banco de frases conversacionais nativas em disco.
 * Chamado automaticamente se o arquivo não existir.
 */
async function ensureConversationalFile(source) {
  if (fs.existsSync(source.parsed)) return true;

  ensureDir(INGEST_DIR);
  const out = fs.createWriteStream(source.parsed, { flags: 'w' });

  for (const phrase of CONVERSATIONAL_SEED) {
    out.write(normalizeSentence(phrase) + '\n');
  }

  await new Promise(resolve => out.end(resolve));
  console.log(`[INGEST] Conversacional: ${CONVERSATIONAL_SEED.length} frases gravadas → ${source.parsed}`);
  return true;
}

// ---------------------------------------------------------------------------
// DIVISOR DE FRASES
// ---------------------------------------------------------------------------

function splitSentences(text) {
  if (!text) return [];
  return text
    .split(/(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÃÕÂÊÔ\w])/)
    .map(s => s.trim())
    .filter(s => s.length >= MIN_CHARS);
}

// ---------------------------------------------------------------------------
// LEITOR DE ARQUIVO EM BLOCOS (cursor-based)
// ---------------------------------------------------------------------------

function readBatch(filePath, cursor, batchSize) {
  if (!fs.existsSync(filePath)) return [];

  const stats = fs.statSync(filePath);

  if (cursor.byteOffset >= stats.size) {
    cursor.byteOffset = 0;
    cursor.linesRead  = 0;
    cursor.done       = false;
    console.log(`[INGEST] Cursor resetado para ${path.basename(filePath)} — reiniciando`);
  }

  const fd     = fs.openSync(filePath, 'r');
  const BUF    = Buffer.allocUnsafe(Math.min(512 * 1024, stats.size));
  let   offset = cursor.byteOffset;
  let   text   = '';
  const lines  = [];

  while (lines.length < batchSize) {
    const bytesRead = fs.readSync(fd, BUF, 0, BUF.length, offset);
    if (bytesRead === 0) break;

    text += BUF.slice(0, bytesRead).toString('utf8');
    offset += bytesRead;

    let newline;
    while ((newline = text.indexOf('\n')) !== -1 && lines.length < batchSize) {
      const line = text.slice(0, newline).trim();
      text = text.slice(newline + 1);
      if (line.length > 0) lines.push(line);
    }

    if (offset >= stats.size) break;
  }

  fs.closeSync(fd);

  const consumed = lines.join('\n').length + lines.length;
  cursor.byteOffset = Math.min(cursor.byteOffset + consumed, stats.size);
  cursor.linesRead += lines.length;

  return lines;
}

// ---------------------------------------------------------------------------
// CONVERSÃO FRASE → PARES DE TREINO
// ---------------------------------------------------------------------------

function sentencesToTrainingPairs(sentences, vocab) {
  const allPairs = [];

  for (const sentence of sentences) {
    const tokens = tokenizeText(sentence);
    if (tokens.length < 2) continue;

    vocab.addTokens(tokens);

    const ids = vocab.tokenize(sentence);
    if (ids.length < 2) continue;

    const pairs = makeTrainingPairs(ids);
    allPairs.push(...pairs);
  }

  return allPairs;
}

// ---------------------------------------------------------------------------
// VERIFICAÇÃO E DOWNLOAD DE FONTES
// ---------------------------------------------------------------------------

async function ensureSourceFile(source) {
  ensureDir(INGEST_DIR);

  // Fonte conversacional local — sem download
  if (!source.url) return true;

  if (needsDownload(source.file, source.maxAge)) {
    console.log(`[INGEST] Baixando ${source.name}...`);
    try {
      await downloadFile(source.url, source.file);
      const stat = fs.statSync(source.file);
      console.log(`[INGEST] ${source.name} baixado: ${(stat.size / 1024 / 1024).toFixed(1)}MB`);

      if (fs.existsSync(source.parsed)) fs.unlinkSync(source.parsed);
      saveCursor(source.cursor, { byteOffset: 0, linesRead: 0, done: false });
      return true;
    } catch (err) {
      console.log(`[INGEST] Falha no download de ${source.name}: ${err.message}`);
      return fs.existsSync(source.file) || fs.existsSync(source.parsed);
    }
  }

  return true;
}

async function ensureParsedFile(source) {
  // Fonte conversacional local
  if (!source.url) {
    return await ensureConversationalFile(source);
  }

  if (fs.existsSync(source.parsed)) return true;
  if (!fs.existsSync(source.file))  return false;

  if (source.name === 'tatoeba') {
    const count = await parseTatoeba(source);
    return count > 0;
  }

  return false;
}

// ---------------------------------------------------------------------------
// ENTRY POINT — runDataIngestion()
// ---------------------------------------------------------------------------

/**
 * Roda um ciclo de ingestão.
 * Alterna entre TATOEBA e CONVERSATIONAL em round-robin.
 * Retorna: { pairs: Array, vocabGrowth: number, sentences: number }
 */
async function runDataIngestion(vocab, options) {
  options = options || {};
  const batchSize = options.batchSize || BATCH_SIZE;

  ensureDir(INGEST_DIR);
  loadStats();

  if (!ingestState.initialized) {
    ingestState.initialized = true;
    console.log('[INGEST] Data Engine v3.0 iniciado (Wikipedia removida).');
    console.log(`[INGEST] Dir: ${INGEST_DIR} | Batch: ${batchSize} frases/ciclo`);
  }

  const allPairs    = [];
  const vocabBefore = vocab.size;
  let   totalSents  = 0;
  let   validSents  = 0;

  // Round-robin entre as fontes ativas
  const sourceKeys = Object.keys(SOURCES);
  const cycleCount = ingestState.stats.total_inserted || 0;
  const sourceKey  = sourceKeys[cycleCount % sourceKeys.length];
  const source     = SOURCES[sourceKey];

  console.log(`[INGEST] Fonte ativa: ${source.name}`);

  const available = await ensureSourceFile(source);
  if (!available) {
    console.log(`[INGEST] ${source.name}: não disponível, pulando ciclo.`);
    return { pairs: [], vocabGrowth: 0, sentences: 0 };
  }

  const parsed = await ensureParsedFile(source);
  if (!parsed) {
    console.log(`[INGEST] ${source.name}: parse ainda não disponível.`);
    return { pairs: [], vocabGrowth: 0, sentences: 0 };
  }

  const cursor    = readCursor(source.cursor);
  const sentences = readBatch(source.parsed, cursor, batchSize);
  saveCursor(source.cursor, cursor);

  totalSents = sentences.length;
  console.log(`[INGEST] frases extraídas: ${totalSents}`);

  if (totalSents === 0) {
    console.log(`[INGEST] ${source.name}: sem frases no batch atual.`);
    return { pairs: [], vocabGrowth: 0, sentences: 0 };
  }

  // Filtrar novamente após normalização
  const filtered = sentences.filter(isValidSentence);
  validSents = filtered.length;
  console.log(`[INGEST] frases válidas: ${validSents}`);

  const pairs = sentencesToTrainingPairs(filtered, vocab);
  allPairs.push(...pairs);

  const vocabGrowth = vocab.size - vocabBefore;

  ingestState.stats.total_extracted += totalSents;
  ingestState.stats.total_valid     += validSents;
  ingestState.stats.total_inserted  += pairs.length;
  ingestState.stats.last_run        = new Date().toISOString();

  if (!ingestState.stats.sources[source.name]) {
    ingestState.stats.sources[source.name] = {};
  }
  ingestState.stats.sources[source.name].last_batch  = validSents;
  ingestState.stats.sources[source.name].total_pairs =
    (ingestState.stats.sources[source.name].total_pairs || 0) + pairs.length;

  saveStats();

  console.log(`[INGEST] ${source.name}: ${validSents} frases → ${pairs.length} pares`);
  if (vocabGrowth > 0) {
    console.log(`[INGEST] vocabulário: +${vocabGrowth} tokens (total: ${vocab.size})`);
  }

  return {
    pairs:       allPairs,
    vocabGrowth: vocabGrowth,
    sentences:   validSents,
  };
}

// ---------------------------------------------------------------------------
// STATUS
// ---------------------------------------------------------------------------

function getIngestStatus() {
  loadStats();
  const status = { ...ingestState.stats };

  status.files = {};
  for (const [key, src] of Object.entries(SOURCES)) {
    const parsedExists = src.parsed ? fs.existsSync(src.parsed) : false;
    const rawExists    = src.file   ? fs.existsSync(src.file)   : false;
    const cursor       = readCursor(src.cursor);

    status.files[src.name] = {
      raw_downloaded:  rawExists,
      parsed_exists:   parsedExists,
      raw_size_mb:     rawExists ? (fs.statSync(src.file).size / 1024 / 1024).toFixed(1) : 0,
      parsed_lines:    cursor.linesRead,
      cursor_position: cursor.byteOffset,
    };
  }

  return status;
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

module.exports = {
  runDataIngestion,
  getIngestStatus,
  isValidSentence,
  normalizeSentence,
  splitSentences,
  INGEST_DIR,
  SOURCES,
};
