/**
 * _akie_ingest.js  —  AKIE Data Engine (Ingestão Autônoma)
 *
 * Pipeline permanente de aquisição de linguagem real em escala.
 * Roda integrado ao scheduler do worker.js como modo DATA_INGESTION.
 *
 * Fontes:
 *   1. Simple Wikipedia (XML dump, ~230MB comprimido)
 *   2. Tatoeba sentences.csv (frases simples PT/EN)
 *
 * Fluxo por ciclo:
 *   carregarFila → extrairFrases → filtrar → tokenizar → treinarPares
 *
 * RESTRIÇÕES:
 *   - Zero texto bruto armazenado após processamento
 *   - Zero dependências externas (apenas Node.js nativo)
 *   - Zero alterações em _aether.js ou NEXUS
 */

'use strict';

const https  = require('https');
const http   = require('http');
const fs     = require('fs');
const path   = require('path');
const zlib   = require('zlib');
const stream = require('stream');

const { tokenizeText, makeTrainingPairs } = require('./_akie_vocab');

// ---------------------------------------------------------------------------
// Configuração
// ---------------------------------------------------------------------------

const INGEST_DIR     = process.env.INGEST_DIR || '/data/datasets';
const BATCH_SIZE     = parseInt(process.env.INGEST_BATCH || '1000', 10);
const MIN_WORDS      = 3;
const MAX_WORDS      = 20;
const MAX_QUEUE_SIZE = 50_000; // frases em memória por vez

// Arquivos de controle (cursor de posição)
const WIKI_CURSOR_FILE   = path.join(INGEST_DIR, 'wiki_cursor.json');
const TATOEBA_CURSOR_FILE = path.join(INGEST_DIR, 'tatoeba_cursor.json');
const INGEST_STATS_FILE  = path.join(INGEST_DIR, 'ingest_stats.json');

// ---------------------------------------------------------------------------
// Fontes
// ---------------------------------------------------------------------------

const SOURCES = {
  WIKIPEDIA: {
    name:    'wikipedia',
    url:     'https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2',
    file:    path.join(INGEST_DIR, 'simplewiki.xml.bz2'),
    // Arquivo parseado (texto limpo, uma frase por linha)
    parsed:  path.join(INGEST_DIR, 'simplewiki_sentences.txt'),
    cursor:  WIKI_CURSOR_FILE,
    lang:    'pt',
    maxAge:  24 * 60 * 60 * 1000, // re-download após 24h
  },
  TATOEBA: {
    name:   'tatoeba',
    url:    'https://downloads.tatoeba.org/exports/sentences.csv',
    file:   path.join(INGEST_DIR, 'tatoeba_sentences.csv'),
    parsed: path.join(INGEST_DIR, 'tatoeba_sentences_pt.txt'),
    cursor: TATOEBA_CURSOR_FILE,
    lang:   'pt',
    maxAge: 7 * 24 * 60 * 60 * 1000, // re-download após 7 dias
  },
};

// ---------------------------------------------------------------------------
// Estado interno do módulo
// ---------------------------------------------------------------------------

const ingestState = {
  initialized:  false,
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

/**
 * Download com suporte a HTTP e HTTPS.
 * Sem axios. Sem dependências externas.
 * Retorna Promise<void> — salva em destPath.
 */
function downloadFile(url, destPath) {
  return new Promise((resolve, reject) => {
    const proto = url.startsWith('https') ? https : http;
    const file  = fs.createWriteStream(destPath + '.tmp');

    const doRequest = (targetUrl) => {
      proto.get(targetUrl, (res) => {
        // Seguir redirect
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

/**
 * Verifica se o arquivo precisa ser baixado novamente.
 */
function needsDownload(filePath, maxAgeMs) {
  if (!fs.existsSync(filePath)) return true;
  const stat = fs.statSync(filePath);
  return (Date.now() - stat.mtimeMs) > maxAgeMs;
}

// ---------------------------------------------------------------------------
// FILTRO DE FRASES
// ---------------------------------------------------------------------------

// Verbos e estruturas comuns em PT e EN para validação básica
const VERB_PATTERNS_PT = /\b(é|são|foi|foram|ser|estar|tem|têm|ter|faz|fazer|pode|poder|deve|dever|vai|vão|ir|vem|vir|usa|usar|permite|fazer|gera|gerar|significa|representa|define|indica|mostra|inclui|contém|pertence|existe|ocorre|acontece|resulta|depende|precisa|ajuda|melhora|reduz|aumenta|permite|facilita|requer|possui|contém|apresenta|realiza|desenvolve|produz|cria|forma|compõe|afeta|influencia|explica|descreve)\b/i;
const VERB_PATTERNS_EN = /\b(is|are|was|were|be|been|has|have|had|do|does|did|can|could|will|would|may|might|shall|should|must|need|use|uses|means|refers|defines|includes|contains|allows|requires|provides|enables|creates|forms|affects|helps|makes|gives|takes|shows|tells|says|becomes|remains|appears|seems|looks|feels|works|runs|goes|comes|gets|puts|sets|keeps|holds|turns|leads|brings|carries|starts|stops|ends|changes|grows|moves|happens|occurs|exists|results|depends|needs|represents|indicates|suggests|implies|demonstrates|explains|describes)\b/i;

/**
 * Verifica se uma frase é válida para ingestão.
 * Critérios: tamanho, conteúdo, presença de verbo.
 */
function isValidSentence(sentence) {
  if (!sentence || typeof sentence !== 'string') return false;

  const trimmed = sentence.trim();
  if (trimmed.length < 10) return false;
  if (trimmed.length > 200) return false;

  // Remover frases com caracteres problemáticos
  if (/[<>{}[\]|\\^`~]/.test(trimmed)) return false;
  if (/https?:\/\//.test(trimmed)) return false;
  if (/\d{4,}/.test(trimmed)) return false; // números longos (anos OK: 4 dígitos, mas evitar IDs)

  // Contar palavras
  const words = trimmed.split(/\s+/).filter(w => w.length > 0);
  if (words.length < MIN_WORDS) return false;
  if (words.length > MAX_WORDS) return false;

  // Verificar se tem estrutura verbal (PT ou EN)
  if (!VERB_PATTERNS_PT.test(trimmed) && !VERB_PATTERNS_EN.test(trimmed)) return false;

  // Rejeitar frases que são só números/símbolos
  const alphaRatio = (trimmed.match(/[a-záéíóúãõâêôàç]/gi) || []).length / trimmed.length;
  if (alphaRatio < 0.6) return false;

  return true;
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
// WIKIPEDIA — Parser de XML/BZ2
// ---------------------------------------------------------------------------

/**
 * Converte o dump XML comprimido em arquivo de frases (1 por linha).
 * Usa streaming — nunca carrega o XML inteiro em memória.
 *
 * Processo:
 *   bz2 stream → descomprimir (bzip2 manual não disponível no Node nativo)
 *   Alternativa: usar apenas gzip se disponível, ou pré-processar offline.
 *
 * NOTA: Node.js não tem bzip2 nativo. Usamos o binário `bzip2` do sistema
 * (disponível no Railway/Ubuntu) via child_process.spawn.
 */
async function parseWikipediaDump(source) {
  const { file, parsed } = source;

  if (!fs.existsSync(file)) {
    console.log('[INGEST] Wikipedia: arquivo não encontrado, pulando parse.');
    return 0;
  }

  console.log('[INGEST] Wikipedia: iniciando parse do dump XML...');

  const { spawn } = require('child_process');
  const parsedStream = fs.createWriteStream(parsed, { flags: 'w' });

  return new Promise((resolve, reject) => {
    // Descomprimir via bzip2 -d -c (stdout)
    const bzip2 = spawn('bzip2', ['-d', '-c', file]);
    let buffer     = '';
    let inText     = false;
    let extracted  = 0;
    let lineCount  = 0;

    bzip2.stdout.on('data', (chunk) => {
      buffer += chunk.toString('utf8', 0, Math.min(chunk.length, 65536));

      // Processar por blocos para controlar memória
      let newline;
      while ((newline = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);

        // Detectar início e fim de bloco de texto
        if (line.includes('<text ') || line.trim().startsWith('<text>')) {
          inText = true;
        }
        if (line.includes('</text>')) {
          inText = false;
        }

        if (!inText) continue;

        // Limpar markup wiki
        const clean = cleanWikiMarkup(line);
        if (!clean) continue;

        // Dividir em frases
        const sentences = splitSentences(clean);
        for (const s of sentences) {
          if (isValidSentence(s)) {
            parsedStream.write(normalizeSentence(s) + '\n');
            extracted++;
            lineCount++;
          }
        }

        // Limite de segurança: não parsear mais de 2M frases por vez
        if (extracted > 2_000_000) {
          bzip2.kill();
          break;
        }
      }
    });

    bzip2.on('close', () => {
      parsedStream.end(() => {
        console.log(`[INGEST] Wikipedia: ${extracted} frases válidas extraídas → ${parsed}`);
        resolve(extracted);
      });
    });

    bzip2.stderr.on('data', (d) => {
      // bzip2 escreve progresso no stderr — ignorar
    });

    bzip2.on('error', (err) => {
      parsedStream.end();
      // bzip2 não disponível — tentar fallback gzip
      console.log('[INGEST] Wikipedia: bzip2 não disponível, pulando. Use arquivo .gz ou pré-processe offline.');
      resolve(0);
    });
  });
}

/**
 * Remove markup wiki de uma linha de texto.
 * Remove: templates {{...}}, links [[...]], tags XML, cabeçalhos ==
 */
function cleanWikiMarkup(line) {
  let s = line;

  // Remover tags XML
  s = s.replace(/<[^>]+>/g, ' ');

  // Remover templates {{ ... }}
  s = s.replace(/\{\{[^}]*\}\}/g, ' ');

  // Converter links [[Texto|Display]] → Display ou Texto
  s = s.replace(/\[\[(?:[^\]|]*\|)?([^\]]*)\]\]/g, '$1');

  // Remover links externos [http://... texto]
  s = s.replace(/\[https?:\/\/[^\s\]]+\s*([^\]]*)\]/g, '$1');

  // Remover cabeçalhos ==
  s = s.replace(/={2,}[^=]+=*/g, '');

  // Remover referências
  s = s.replace(/&[a-z]+;/g, ' ');

  // Limpar espaços
  s = s.replace(/\s+/g, ' ').trim();

  return s.length > 5 ? s : '';
}

// ---------------------------------------------------------------------------
// TATOEBA — Parser CSV
// ---------------------------------------------------------------------------

/**
 * Parseia o sentences.csv do Tatoeba e extrai frases PT e EN.
 *
 * Formato: id\tlang\ttext
 * Ex: 1234\tpor\tO gato está na mesa.
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

  let extracted = 0;

  for (const line of lines) {
    const parts = line.split('\t');
    if (parts.length < 3) continue;

    const lang = parts[1].trim().toLowerCase();
    // Aceitar PT (por, pt) e EN (eng, en) 
    if (!['por', 'pt', 'eng', 'en'].includes(lang)) continue;

    const text = parts.slice(2).join('\t').trim();
    if (isValidSentence(text)) {
      out.write(normalizeSentence(text) + '\n');
      extracted++;
    }

    if (extracted > 500_000) break; // limite de segurança
  }

  await new Promise(resolve => out.end(resolve));
  console.log(`[INGEST] Tatoeba: ${extracted} frases válidas extraídas → ${parsed}`);
  return extracted;
}

// ---------------------------------------------------------------------------
// DIVISOR DE FRASES
// ---------------------------------------------------------------------------

/**
 * Divide um bloco de texto em frases individuais.
 * Estratégia conservadora: divide em .!? seguidos de espaço+maiúscula.
 */
function splitSentences(text) {
  if (!text) return [];

  return text
    // Ponto final / exclamação / interrogação seguido de espaço
    .split(/(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÃÕÂÊÔ\w])/)
    .map(s => s.trim())
    .filter(s => s.length > 10);
}

// ---------------------------------------------------------------------------
// LEITOR DE ARQUIVO EM BLOCOS (cursor-based)
// ---------------------------------------------------------------------------

/**
 * Lê até `batchSize` linhas a partir do cursor atual.
 * Atualiza o cursor após a leitura.
 * Retorna array de strings (frases).
 *
 * Usa `byteOffset` para evitar carregar o arquivo inteiro em memória.
 */
function readBatch(filePath, cursor, batchSize) {
  if (!fs.existsSync(filePath)) return [];

  const stats = fs.statSync(filePath);

  // Cursor chegou ao fim do arquivo — resetar para recomeçar
  if (cursor.byteOffset >= stats.size) {
    cursor.byteOffset = 0;
    cursor.linesRead  = 0;
    cursor.done       = false;
    console.log(`[INGEST] Cursor resetado para ${path.basename(filePath)} — reiniciando do início`);
  }

  const fd     = fs.openSync(filePath, 'r');
  const BUF    = Buffer.allocUnsafe(Math.min(512 * 1024, stats.size)); // 512KB por leitura
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

  // Atualizar cursor pela posição real dos bytes consumidos
  // Recalcular baseado nas linhas consumidas (aproximação conservadora)
  const consumed = lines.join('\n').length + lines.length; // +1 por \n em cada linha
  cursor.byteOffset = Math.min(cursor.byteOffset + consumed, stats.size);
  cursor.linesRead += lines.length;

  return lines;
}

// ---------------------------------------------------------------------------
// CONVERSÃO FRASE → PARES DE TREINO
// ---------------------------------------------------------------------------

/**
 * Converte array de frases em pares de treino.
 * Expande vocabulário com tokens novos.
 * Retorna pares {x, y} para trainBatch.
 */
function sentencesToTrainingPairs(sentences, vocab) {
  const allPairs = [];

  for (const sentence of sentences) {
    const tokens = tokenizeText(sentence);
    if (tokens.length < MIN_WORDS) continue;

    // Adicionar tokens novos ao vocabulário
    vocab.addTokens(tokens);

    // Tokenizar com IDs
    const ids = vocab.tokenize(sentence);
    if (ids.length < 3) continue;

    const pairs = makeTrainingPairs(ids);
    allPairs.push(...pairs);
  }

  return allPairs;
}

// ---------------------------------------------------------------------------
// VERIFICAÇÃO E DOWNLOAD DE FONTES
// ---------------------------------------------------------------------------

/**
 * Garante que o arquivo da fonte existe e está atualizado.
 * Faz download apenas se necessário.
 * Retorna true se o arquivo está disponível.
 */
async function ensureSourceFile(source) {
  ensureDir(INGEST_DIR);

  if (needsDownload(source.file, source.maxAge)) {
    console.log(`[INGEST] Baixando ${source.name}...`);
    console.log(`[INGEST] URL: ${source.url}`);

    try {
      await downloadFile(source.url, source.file);
      const stat = fs.statSync(source.file);
      console.log(`[INGEST] ${source.name} baixado: ${(stat.size / 1024 / 1024).toFixed(1)}MB`);

      // Re-parsear após novo download
      const parsedExists = fs.existsSync(source.parsed);
      if (parsedExists) fs.unlinkSync(source.parsed);

      // Resetar cursor
      saveCursor(source.cursor, { byteOffset: 0, linesRead: 0, done: false });

      return true;
    } catch (err) {
      console.log(`[INGEST] Falha no download de ${source.name}: ${err.message}`);
      // Se arquivo anterior existe, continuar com ele
      return fs.existsSync(source.file) || fs.existsSync(source.parsed);
    }
  }

  return true;
}

/**
 * Garante que o arquivo parseado existe.
 * Se não existe mas o raw existe, parsear.
 */
async function ensureParsedFile(source) {
  if (fs.existsSync(source.parsed)) return true;
  if (!fs.existsSync(source.file))  return false;

  if (source.name === 'wikipedia') {
    const count = await parseWikipediaDump(source);
    return count > 0;
  }

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
 * Roda um ciclo de ingestão de dados.
 * Chamado pelo scheduler no modo DATA_INGESTION.
 *
 * Retorna: { pairs: Array, vocabGrowth: number, sentences: number }
 */
async function runDataIngestion(vocab, options) {
  options = options || {};
  const batchSize = options.batchSize || BATCH_SIZE;

  ensureDir(INGEST_DIR);
  loadStats();

  if (!ingestState.initialized) {
    ingestState.initialized = true;
    console.log('[INGEST] Data Engine iniciado.');
    console.log(`[INGEST] Dir: ${INGEST_DIR} | Batch: ${batchSize} frases/ciclo`);
  }

  const allPairs     = [];
  const vocabBefore  = vocab.size;
  let   totalSents   = 0;
  let   validSents   = 0;

  // Selecionar fonte ativa (alterna entre fontes a cada ciclo)
  const sourceKeys = Object.keys(SOURCES);
  const cycleCount = ingestState.stats.total_inserted || 0;
  const sourceKey  = sourceKeys[cycleCount % sourceKeys.length];  // round-robin
  const source     = SOURCES[sourceKey];

  console.log(`[INGEST] Fonte ativa: ${source.name}`);

  // 1. Garantir arquivo disponível (download se necessário)
  const available = await ensureSourceFile(source);
  if (!available) {
    console.log(`[INGEST] ${source.name}: não disponível, pulando ciclo.`);
    return { pairs: [], vocabGrowth: 0, sentences: 0 };
  }

  // 2. Garantir arquivo parseado
  const parsed = await ensureParsedFile(source);
  if (!parsed) {
    console.log(`[INGEST] ${source.name}: parse ainda não disponível.`);
    return { pairs: [], vocabGrowth: 0, sentences: 0 };
  }

  // 3. Ler batch do cursor atual
  const cursor   = readCursor(source.cursor);
  const sentences = readBatch(source.parsed, cursor, batchSize);
  saveCursor(source.cursor, cursor);

  totalSents = sentences.length;
  console.log(`[INGEST] frases extraídas: ${totalSents}`);

  if (totalSents === 0) {
    console.log(`[INGEST] ${source.name}: sem frases no batch atual.`);
    return { pairs: [], vocabGrowth: 0, sentences: 0 };
  }

  // 4. Filtrar (isValidSentence já foi aplicado no parse, mas aplicar novamente
  //    para garantir qualidade após normalização)
  const filtered = sentences.filter(isValidSentence);
  validSents = filtered.length;
  console.log(`[INGEST] frases válidas: ${validSents}`);

  // 5. Converter em pares de treino + expandir vocabulário
  const pairs = sentencesToTrainingPairs(filtered, vocab);
  allPairs.push(...pairs);

  const vocabGrowth = vocab.size - vocabBefore;

  // 6. Atualizar stats
  ingestState.stats.total_extracted += totalSents;
  ingestState.stats.total_valid     += validSents;
  ingestState.stats.total_inserted  += pairs.length;
  ingestState.stats.last_run        = new Date().toISOString();
  ingestState.stats.sources[source.name] = ingestState.stats.sources[source.name] || {};
  ingestState.stats.sources[source.name].last_batch = validSents;
  ingestState.stats.sources[source.name].total_pairs = (ingestState.stats.sources[source.name].total_pairs || 0) + pairs.length;

  saveStats();

  console.log(`[INGEST] frases inseridas: ${validSents} → ${pairs.length} pares`);
  if (vocabGrowth > 0) {
    console.log(`[INGEST] vocabulário: +${vocabGrowth} tokens (total: ${vocab.size})`);
  }

  return {
    pairs:        allPairs,
    vocabGrowth:  vocabGrowth,
    sentences:    validSents,
  };
}

// ---------------------------------------------------------------------------
// STATUS
// ---------------------------------------------------------------------------

/**
 * Retorna status atual do data engine.
 * Usado pelo endpoint GET /status do worker.
 */
function getIngestStatus() {
  loadStats();
  const status = { ...ingestState.stats };

  // Adicionar info sobre arquivos locais
  status.files = {};
  for (const [key, src] of Object.entries(SOURCES)) {
    const rawExists    = fs.existsSync(src.file);
    const parsedExists = fs.existsSync(src.parsed);
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
  cleanWikiMarkup,
  INGEST_DIR,
  SOURCES,
};
