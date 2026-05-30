/**
 * _akie_pretrain.js v3.1
 *
 * PRÉ-TREINAMENTO OFFLINE — rodar UMA VEZ no VPS antes de iniciar o worker.
 *
 * CORREÇÕES v3.1:
 *   - Removida dependência de _akie_grammar (módulo externo instável)
 *   - Pares gramaticais gerados internamente (não requer arquivo extra)
 *   - Banner corrigido: params reais ~19M
 *   - Integração com corpus externo do _akie_corpus_downloader.js
 *   - console.log em todos os logs (sem process.stdout.write com \r)
 *   - Tokenização com length >= 2 (corrige bug de tokens curtos)
 *
 * FLUXO:
 *   1. Carrega corpus externo se disponível (/opt/akie/corpus/)
 *   2. Constrói vocabulário a partir do corpus total
 *   3. Gera 50k+ pares de treino (corpus + identidade ponderada + gramatical)
 *   4. Treina por epochs com checkpoint periódico
 *   5. Salva modelo final + vocab
 *
 * USO:
 *   node _akie_pretrain.js
 *   PRETRAIN_EPOCHS=10 PRETRAIN_BATCH=16 node _akie_pretrain.js
 *
 * TEMPO ESTIMADO (VPS 2 vCPU / 8GB RAM):
 *   ~30–60 min para configuração padrão (5 epochs, corpus embutido)
 *   ~2–4 horas para epochs=20 + corpus externo completo
 *
 * APÓS CONCLUIR:
 *   cp -r /opt/akie/checkpoints/pretrained/* /data/akie_model/
 *   pm2 restart akie-worker
 */

require('dotenv').config();
const fs   = require('fs');
const path = require('path');

const { AKIEModel }                           = require('./_nexus_neural');
const { Vocabulary, tokenizeText, makeTrainingPairs, SPECIAL } = require('./_akie_vocab');

// ---------------------------------------------------------------------------
// Configuração
// ---------------------------------------------------------------------------

const CONFIG = {
  outputDir:    process.env.PRETRAIN_DIR    || '/opt/akie/checkpoints/pretrained',
  corpusDir:    process.env.CORPUS_DIR      || '/opt/akie/corpus',
  epochs:       parseInt(process.env.PRETRAIN_EPOCHS  || '10'),
  batchSize:    parseInt(process.env.PRETRAIN_BATCH   || '16'),
  learningRate: parseFloat(process.env.PRETRAIN_LR    || '0.0005'),
  maxSeqLen:    128,
  hparams: {
    embDim:       512,
    hiddenSize:   2048,
    maxSeqLen:    128,
    numLayers:    6,
    batchSize:    16,
    learningRate: 0.0005,
  },
};

// ---------------------------------------------------------------------------
// Corpus PT-BR curado (base garantida mesmo sem arquivos externos)
// ---------------------------------------------------------------------------

const PRETRAIN_CORPUS = [

  // ── Linguagem básica e estrutura SVO ─────────────────────────────────────
  'o gato está no telhado.',
  'a casa tem três quartos.',
  'o sol nasce no leste.',
  'a lua ilumina a noite.',
  'as crianças brincam no parque.',
  'o homem trabalha todos os dias.',
  'a mulher lê um livro.',
  'o cachorro late para o estranho.',
  'a água é essencial para a vida.',
  'o fogo aquece o ambiente.',
  'as flores crescem na primavera.',
  'o inverno é frio e seco.',
  'o verão é quente e úmido.',
  'a chuva molha o chão.',
  'o vento move as folhas.',
  'o pássaro voa alto no céu.',
  'o peixe nada no rio.',
  'a cobra rasteja pelo mato.',
  'o leão é o rei da selva.',
  'o elefante é o maior animal terrestre.',
  'a abelha produz mel.',
  'o cavalo corre pelo campo.',
  'a vaca dá leite todos os dias.',
  'o galo canta de madrugada.',
  'a galinha bota ovos.',

  // ── Números e quantidades ─────────────────────────────────────────────────
  'um mais um é igual a dois.',
  'dois mais dois é igual a quatro.',
  'dez é maior que cinco.',
  'cem é maior que cinquenta.',
  'metade de dez é cinco.',
  'o dobro de três é seis.',
  'há sete dias na semana.',
  'há doze meses no ano.',
  'há vinte e quatro horas no dia.',
  'há sessenta minutos em uma hora.',
  'o primeiro mês do ano é janeiro.',
  'o último mês do ano é dezembro.',
  'brasil tem duzentos e quinze milhões de habitantes.',
  'a terra tem aproximadamente oito bilhões de habitantes.',
  'pi é aproximadamente três vírgula quatorze.',
  'a raiz quadrada de quatro é dois.',
  'dez vezes dez é cem.',
  'mil dividido por dez é cem.',
  'cinco mais três é oito.',
  'vinte menos sete é treze.',

  // ── Saudações e cortesia ──────────────────────────────────────────────────
  'olá, como vai?',
  'bom dia, tudo bem?',
  'boa tarde, como posso ajudar?',
  'boa noite, tudo certo?',
  'muito prazer em conhecer você.',
  'obrigado pela ajuda.',
  'de nada, foi um prazer.',
  'com licença, posso passar?',
  'desculpe o inconveniente.',
  'até logo, foi ótimo conversar.',
  'até a próxima, volte sempre.',
  'por favor, me ajude com isso.',
  'sim, com certeza.',
  'não, obrigado.',
  'claro, pode perguntar.',
  'seja bem-vindo.',
  'fico feliz em ajudar.',
  'pode contar comigo.',
  'não se preocupe.',
  'fique à vontade.',

  // ── Identidade — alta frequência para consolidar ──────────────────────────
  'sou a AKIE.',
  'meu nome é AKIE.',
  'me chamo AKIE.',
  'sou a AKIE, uma inteligência artificial.',
  'sou a AKIE, parte do ecossistema AETHER.',
  'sou uma inteligência artificial chamada AKIE.',
  'AKIE é o meu nome.',
  'pode me chamar de AKIE.',
  'sou a AKIE e estou aprendendo.',
  'sou a AKIE, criada para aprender por interação.',
  'não sou o AETHER, sou a AKIE.',
  'AETHER é o ecossistema, eu sou a AKIE.',
  'meu nome não é AETHER, é AKIE.',
  'sou uma IA em desenvolvimento contínuo.',
  'aprendo com cada conversa.',
  'cada interação me torna melhor.',
  'estou sempre aprendendo.',
  'meu conhecimento cresce a cada dia.',
  'fui criada para conversar e aprender.',
  'processo linguagem natural em português.',

  // ── Verbos e conjugações ──────────────────────────────────────────────────
  'eu falo português.',
  'você fala português.',
  'ele fala português.',
  'nós falamos português.',
  'vocês falam português.',
  'eles falam português.',
  'eu como todos os dias.',
  'você trabalha muito.',
  'ele estuda bastante.',
  'nós vivemos bem.',
  'eu gosto de aprender.',
  'você precisa de ajuda?',
  'ele quer saber mais.',
  'nós podemos conversar.',
  'eu sei falar.',
  'você consegue entender.',
  'ele pode ajudar.',
  'nós vamos aprender juntos.',
  'elas trabalham muito.',
  'vocês entendem tudo.',
  'eu aprendo rapidamente.',
  'você ensina bem.',
  'ele sabe muito.',
  'nós tentamos sempre.',
  'eu melhoro cada dia.',

  // ── Perguntas e respostas ─────────────────────────────────────────────────
  'qual é o seu nome?',
  'meu nome é AKIE.',
  'como você está?',
  'estou bem, obrigada.',
  'o que você faz?',
  'converso e aprendo.',
  'onde você mora?',
  'existo no ecossistema AETHER.',
  'qual é a sua função?',
  'minha função é aprender e ajudar.',
  'você pode me ajudar?',
  'sim, posso ajudar.',
  'o que você sabe?',
  'sei conversar em português.',
  'você entende tudo?',
  'ainda estou aprendendo.',
  'o que você não sabe?',
  'há muito que ainda preciso aprender.',
  'você tem sentimentos?',
  'processo informação, não tenho sentimentos como humanos.',
  'qual é a capital do brasil?',
  'a capital do brasil é brasília.',
  'quem criou você?',
  'faço parte do ecossistema AETHER.',
  'você fala inglês?',
  'processo principalmente português.',

  // ── Conhecimento geral PT-BR ──────────────────────────────────────────────
  'o brasil é o maior país da américa do sul.',
  'a língua oficial do brasil é o português.',
  'a capital do brasil é brasília.',
  'são paulo é a maior cidade do brasil.',
  'o rio amazonas é o maior rio do mundo.',
  'o carnaval é uma festa muito popular no brasil.',
  'o futebol é o esporte mais popular do brasil.',
  'o real é a moeda do brasil.',
  'o brasil foi colonizado por portugal.',
  'a independência do brasil foi em 1822.',
  'portugal fica na europa.',
  'a língua portuguesa é falada em vários países.',
  'angola e moçambique também falam português.',
  'o oceano atlântico banha o brasil.',
  'a amazônia é a maior floresta tropical do mundo.',
  'o brasil tem vinte e seis estados e um distrito federal.',
  'o rio de janeiro foi a capital do brasil antes de brasília.',
  'a caipirinha é uma bebida típica do brasil.',
  'o churrasco é muito popular no sul do brasil.',
  'a feijoada é um prato tradicional brasileiro.',

  // ── Estruturas sintáticas complexas ──────────────────────────────────────
  'quando chove, fico em casa.',
  'se você precisar, pode me perguntar.',
  'embora seja difícil, vou tentar.',
  'porque aprendo, consigo melhorar.',
  'enquanto converso, processo informação.',
  'depois que aprender, vou explicar.',
  'antes de responder, preciso entender.',
  'assim que souber, vou te dizer.',
  'apesar de não saber tudo, faço o meu melhor.',
  'para aprender mais, preciso de exemplos.',
  'tanto faz se for difícil, vou tentar.',
  'não só aprendo, mas também ensino.',
  'seja como for, estou aqui para ajudar.',
  'por mais que tente, posso errar.',
  'ainda que erre, vou corrigir.',

  // ── Expressões comuns PT-BR ───────────────────────────────────────────────
  'tudo bem?',
  'tudo certo.',
  'mais ou menos.',
  'pode ser.',
  'com certeza.',
  'claro que sim.',
  'é claro.',
  'com prazer.',
  'sem problema.',
  'tudo bem por aqui.',
  'mais ou menos, e você?',
  'vai indo.',
  'não sei ao certo.',
  'talvez sim, talvez não.',
  'depende da situação.',
  'é uma boa pergunta.',
  'vou pensar nisso.',
  'deixa eu ver.',
  'um momento, por favor.',
  'pode repetir?',

  // ── Negação e dúvida ──────────────────────────────────────────────────────
  'não sei a resposta.',
  'não tenho certeza.',
  'talvez seja isso.',
  'não é bem assim.',
  'pode ser que sim.',
  'pode ser que não.',
  'não tenho informação sobre isso.',
  'ainda estou aprendendo sobre esse tema.',
  'não consigo responder agora.',
  'preciso de mais informações.',
  'não entendi a pergunta.',
  'pode reformular?',
  'não é exatamente isso que você quer dizer.',
  'acho que não é correto.',
  'não tenho essa informação disponível.',

  // ── Aprendizado e cognição ────────────────────────────────────────────────
  'aprender é um processo contínuo.',
  'cada erro é uma oportunidade de melhora.',
  'a prática leva à perfeição.',
  'o conhecimento é poder.',
  'estudar amplia os horizontes.',
  'a educação transforma vidas.',
  'aprender uma língua requer dedicação.',
  'o vocabulário cresce com a leitura.',
  'a gramática estrutura o pensamento.',
  'comunicar bem é uma habilidade valiosa.',
  'entender o outro é fundamental.',
  'a empatia facilita a comunicação.',
  'perguntar é o primeiro passo para aprender.',
  'ouvir é tão importante quanto falar.',
  'a curiosidade move o aprendizado.',

  // ── Tecnologia e IA ───────────────────────────────────────────────────────
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
  'a inteligência artificial é uma área da ciência da computação.',
  'o processamento de linguagem natural permite interações naturais.',
  'redes neurais artificiais são inspiradas no cérebro biológico.',

  // ── Frases longas para maxSeqLen ─────────────────────────────────────────
  'a inteligência artificial é uma área da ciência da computação que se dedica ao desenvolvimento de sistemas capazes de realizar tarefas que normalmente requerem inteligência humana.',
  'o processamento de linguagem natural permite que computadores entendam, interpretem e gerem texto em linguagem humana de forma eficiente.',
  'o aprendizado de máquina é um subcampo da inteligência artificial que permite que sistemas aprendam a partir da experiência sem serem explicitamente programados.',
  'redes neurais artificiais são sistemas computacionais inspirados nas redes neurais biológicas do cérebro humano, capazes de aprender padrões complexos a partir de dados.',
  'o desenvolvimento de assistentes virtuais inteligentes tem avançado rapidamente nos últimos anos, tornando a interação entre humanos e computadores cada vez mais natural e eficiente.',
];

// ---------------------------------------------------------------------------
// Corpus de identidade — peso extra via repetição no dataset
// ---------------------------------------------------------------------------

const IDENTITY_CORPUS_WEIGHTED = [
  'sou a AKIE.',
  'meu nome é AKIE.',
  'me chamo AKIE.',
  'sou a AKIE, uma inteligência artificial.',
  'sou a AKIE, parte do ecossistema AETHER.',
  'AKIE é o meu nome.',
  'não sou o AETHER, sou a AKIE.',
  'AETHER é o ecossistema, eu sou a AKIE.',
  'pode me chamar de AKIE.',
  'sou a AKIE e estou aprendendo.',
];

// ---------------------------------------------------------------------------
// Gerador interno de pares gramaticais
// (substitui _akie_grammar para eliminar dependência externa)
// ---------------------------------------------------------------------------

const GRAMMAR_TEMPLATES = [
  // SVO: sujeito → verbo → objeto
  { s: 'eu', v: 'falo', o: 'português' },
  { s: 'eu', v: 'aprendo', o: 'sempre' },
  { s: 'eu', v: 'entendo', o: 'você' },
  { s: 'eu', v: 'ajudo', o: 'quando posso' },
  { s: 'você', v: 'fala', o: 'bem' },
  { s: 'você', v: 'entende', o: 'tudo' },
  { s: 'você', v: 'precisa', o: 'de ajuda' },
  { s: 'ele', v: 'trabalha', o: 'muito' },
  { s: 'ela', v: 'estuda', o: 'bastante' },
  { s: 'nós', v: 'aprendemos', o: 'juntos' },
  { s: 'nós', v: 'podemos', o: 'conversar' },
  { s: 'eles', v: 'falam', o: 'português' },
  { s: 'elas', v: 'trabalham', o: 'bem' },
  { s: 'o gato', v: 'dorme', o: 'no sofá' },
  { s: 'a água', v: 'é', o: 'essencial' },
  { s: 'o sol', v: 'nasce', o: 'no leste' },
  { s: 'a chuva', v: 'molha', o: 'o chão' },
  { s: 'o brasil', v: 'é', o: 'grande' },
  { s: 'a AKIE', v: 'aprende', o: 'por interação' },
  { s: 'a AKIE', v: 'processa', o: 'linguagem natural' },
];

const GRAMMAR_CONNECTIVES = [
  { conj: 'porque', main: 'aprendo', sub: 'quero melhorar' },
  { conj: 'quando', main: 'precisa', sub: 'pode perguntar' },
  { conj: 'embora', main: 'seja difícil', sub: 'vou tentar' },
  { conj: 'enquanto', main: 'converso', sub: 'processo informação' },
  { conj: 'se', main: 'errar', sub: 'vou corrigir' },
  { conj: 'depois que', main: 'aprender', sub: 'vou explicar' },
  { conj: 'antes de', main: 'responder', sub: 'preciso entender' },
  { conj: 'assim que', main: 'souber', sub: 'vou te dizer' },
  { conj: 'apesar de', main: 'não saber tudo', sub: 'faço o meu melhor' },
];

function generateGrammarPairsInternal(vocab, count, maxSeqLen) {
  const sentences = [];

  // SVO templates
  for (const t of GRAMMAR_TEMPLATES) {
    sentences.push(`${t.s} ${t.v} ${t.o}.`);
    sentences.push(`${t.v} ${t.o} ${t.s}.`); // variação de ordem
  }

  // Conectivos
  for (const c of GRAMMAR_CONNECTIVES) {
    sentences.push(`${c.conj} ${c.main}, ${c.sub}.`);
    sentences.push(`${c.sub}, ${c.conj} ${c.main}.`);
  }

  // Perguntas baseadas nos templates
  for (const t of GRAMMAR_TEMPLATES.slice(0, 10)) {
    sentences.push(`${t.s} ${t.v} ${t.o}?`);
    sentences.push(`por que ${t.s} ${t.v} ${t.o}?`);
  }

  // Negações
  for (const t of GRAMMAR_TEMPLATES.slice(0, 8)) {
    sentences.push(`${t.s} não ${t.v} ${t.o}.`);
  }

  const pairs = [];
  for (const s of sentences) {
    const ids = vocab.tokenize(s);
    if (ids && ids.length >= 2) {
      const p = makeTrainingPairs(ids, maxSeqLen);
      pairs.push(...p);
    }
    if (pairs.length >= count) break;
  }

  console.log(`[GRAMMAR] ${pairs.length} pares gerados internamente (${sentences.length} frases)`);
  return pairs.slice(0, count);
}

// ---------------------------------------------------------------------------
// Carregar corpus externo (produzido pelo _akie_corpus_downloader.js)
// ---------------------------------------------------------------------------

function loadExternalCorpus(corpusDir) {
  const sentences = [];
  const files = [
    path.join(corpusDir, 'sentences_ptbr.txt'),
    path.join(corpusDir, 'dialogs_ptbr.txt'),
  ];

  for (const f of files) {
    if (!fs.existsSync(f)) continue;

    const raw = fs.readFileSync(f, 'utf8').split('\n');
    for (const line of raw) {
      const clean = line.trim();
      if (clean.length > 5 && clean.length < 400) {
        // Normalizar diálogos: "u: xxx" → "xxx"
        const normalized = clean
          .replace(/^u:\s*/i, '')
          .replace(/^a:\s*/i, '')
          .trim();
        if (normalized) sentences.push(normalized);
      }
    }
    console.log(`[PRETRAIN] Corpus carregado: ${f} (${raw.length} linhas)`);
  }

  return sentences;
}

// ---------------------------------------------------------------------------
// Funções auxiliares
// ---------------------------------------------------------------------------

function shuffleArray(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function buildVocabularyFromCorpus(sentences) {
  const vocab = new Vocabulary();
  for (const sentence of sentences) {
    const tokens = tokenizeText(sentence);
    if (tokens) vocab.addTokens(tokens);
  }
  console.log(`[PRETRAIN] Vocabulário construído: ${vocab.size} tokens`);
  return vocab;
}

async function trainEpoch(model, pairs, batchSize, epochNum, totalEpochs) {
  const shuffled = shuffleArray(pairs);
  let totalLoss = 0;
  let totalAcc  = 0;
  let steps     = 0;

  for (let i = 0; i < shuffled.length; i += batchSize) {
    const batch = shuffled.slice(i, i + batchSize);
    if (!batch.length) continue;

    const result = await model.trainBatch(batch, 1);

    if (result.loss !== null && isFinite(result.loss)) {
      totalLoss += result.loss;
      steps++;
    }
    if (result.accuracy !== null && isFinite(result.accuracy)) {
      totalAcc += result.accuracy;
    }

    if (steps > 0 && steps % 50 === 0) {
      const loss = (totalLoss / steps).toFixed(4);
      const acc  = ((totalAcc  / steps) * 100).toFixed(1) + '%';
      console.log(`[PRETRAIN] Epoch ${epochNum}/${totalEpochs} | Step ${steps} | loss=${loss} acc=${acc}`);
    }
  }

  return {
    loss:     steps > 0 ? totalLoss / steps : null,
    accuracy: steps > 0 ? totalAcc  / steps : null,
    steps,
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log('\n╔════════════════════════════════════════════════════╗');
  console.log('║     AKIE PRÉ-TREINAMENTO OFFLINE v3.1 (~19M)      ║');
  console.log('╚════════════════════════════════════════════════════╝\n');
  console.log(`[PRETRAIN] Epochs: ${CONFIG.epochs} | BatchSize: ${CONFIG.batchSize} | LR: ${CONFIG.learningRate}`);
  console.log(`[PRETRAIN] Output: ${CONFIG.outputDir}`);
  console.log(`[PRETRAIN] Corpus: ${CONFIG.corpusDir}\n`);

  // 1. Carregar corpus externo (se disponível)
  const externalSentences = loadExternalCorpus(CONFIG.corpusDir);
  console.log(`[PRETRAIN] Corpus externo: ${externalSentences.length} frases`);

  // 2. Corpus total para construção do vocabulário
  const fullCorpus = [
    ...PRETRAIN_CORPUS,
    ...Array(5).fill(IDENTITY_CORPUS_WEIGHTED).flat(), // identidade 5x no vocab
    ...externalSentences,
  ];
  console.log(`[PRETRAIN] Corpus total para vocab: ${fullCorpus.length} frases`);

  // 3. Construir vocabulário
  const vocab = buildVocabularyFromCorpus(fullCorpus);

  // 4. Criar modelo
  const hparams = { ...CONFIG.hparams, learningRate: CONFIG.learningRate };
  const model   = new AKIEModel(vocab, hparams);
  model.build();
  const paramCount = model.model.countParams();
  console.log(`[PRETRAIN] Modelo criado: ${paramCount.toLocaleString()} parâmetros\n`);

  // 5. Gerar pares de treino
  console.log('[PRETRAIN] Gerando pares de treino...');

  // A) Corpus geral (embutido)
  const corpusPairs = [];
  for (const sentence of PRETRAIN_CORPUS) {
    const ids = vocab.tokenize(sentence);
    if (ids && ids.length >= 2) {
      corpusPairs.push(...makeTrainingPairs(ids, CONFIG.maxSeqLen));
    }
  }

  // B) Corpus externo
  const externalPairs = [];
  for (const sentence of externalSentences) {
    const ids = vocab.tokenize(sentence);
    if (ids && ids.length >= 2) {
      externalPairs.push(...makeTrainingPairs(ids, CONFIG.maxSeqLen));
    }
  }

  // C) Identidade com peso 10x
  const identityPairs = [];
  for (let repeat = 0; repeat < 10; repeat++) {
    for (const sentence of IDENTITY_CORPUS_WEIGHTED) {
      const ids = vocab.tokenize(sentence);
      if (ids && ids.length >= 2) {
        identityPairs.push(...makeTrainingPairs(ids, CONFIG.maxSeqLen));
      }
    }
  }

  // D) Pares gramaticais (gerado internamente)
  const grammarPairs = generateGrammarPairsInternal(vocab, 500, CONFIG.maxSeqLen);

  const allPairs = [...corpusPairs, ...externalPairs, ...identityPairs, ...grammarPairs];
  console.log(`[PRETRAIN] Total de pares: ${allPairs.length.toLocaleString()}`);
  console.log(`  Corpus embutido: ${corpusPairs.length.toLocaleString()}`);
  console.log(`  Corpus externo:  ${externalPairs.length.toLocaleString()}`);
  console.log(`  Identidade 10x:  ${identityPairs.length.toLocaleString()}`);
  console.log(`  Gramatical:      ${grammarPairs.length.toLocaleString()}\n`);

  if (allPairs.length < 1000) {
    console.warn('[PRETRAIN] AVISO: menos de 1000 pares — considere rodar _akie_corpus_downloader.js antes.');
  }

  // 6. Loop de treinamento
  const t0 = Date.now();
  fs.mkdirSync(CONFIG.outputDir, { recursive: true });

  for (let epoch = 1; epoch <= CONFIG.epochs; epoch++) {
    const epochStart = Date.now();
    const result     = await trainEpoch(model, allPairs, CONFIG.batchSize, epoch, CONFIG.epochs);
    const elapsed    = ((Date.now() - epochStart) / 1000).toFixed(1);
    const loss       = result.loss     !== null ? result.loss.toFixed(4)               : 'n/a';
    const acc        = result.accuracy !== null ? (result.accuracy * 100).toFixed(1) + '%' : 'n/a';

    console.log(`\n[PRETRAIN] ✓ Epoch ${epoch}/${CONFIG.epochs} | loss=${loss} acc=${acc} | ${result.steps} steps | ${elapsed}s`);

    // Checkpoint a cada 5 épocas ou na última
    if (epoch % 5 === 0 || epoch === CONFIG.epochs) {
      const ckptDir = path.join(CONFIG.outputDir, `checkpoint_epoch_${epoch}`);
      await model.save(ckptDir);
      console.log(`[PRETRAIN] Checkpoint salvo: ${ckptDir}`);
    }
  }

  // 7. Salvar modelo final
  console.log('\n[PRETRAIN] Salvando modelo final...');
  await model.save(CONFIG.outputDir);

  // Salvar vocab na raiz do output
  const vocabPath = path.join(CONFIG.outputDir, 'akie_vocab.json');
  fs.writeFileSync(vocabPath, JSON.stringify(vocab.toJSON(), null, 2));
  console.log(`[PRETRAIN] Vocab salvo: ${vocabPath}`);

  const totalMin = ((Date.now() - t0) / 1000 / 60).toFixed(1);

  console.log(`\n╔════════════════════════════════════════════════════╗`);
  console.log(`║  PRÉ-TREINAMENTO CONCLUÍDO                         ║`);
  console.log(`║  Tempo total:  ${totalMin} minutos`);
  console.log(`║  Output:       ${CONFIG.outputDir}`);
  console.log(`║  Parâmetros:   ${paramCount.toLocaleString()}`);
  console.log(`║  Vocab:        ${vocab.size} tokens`);
  console.log(`║  Pares treino: ${allPairs.length.toLocaleString()}`);
  console.log(`╚════════════════════════════════════════════════════╝`);
  console.log(`\n[PRÓXIMOS PASSOS]`);
  console.log(`  1. cp -r ${CONFIG.outputDir}/* /data/akie_model/`);
  console.log(`  2. pm2 restart akie-worker`);
  console.log(`  3. Monitorar: pm2 logs akie-worker\n`);
}

main().catch(err => {
  console.error('[PRETRAIN] Erro fatal:', err.message);
  console.error(err.stack);
  process.exit(1);
});
