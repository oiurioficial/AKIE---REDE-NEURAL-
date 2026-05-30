/**
 * _akie_pretrain.js v4.0
 * Path: /opt/akie/app/
 *
 * CORREÇÕES v4.0:
 *   - Path correto: /opt/akie/app/ (não /home/claude/)
 *   - generateSyntheticConversations (nome real do export)
 *   - generateGrammarPairs com assinatura correta (vocab, targetCount, maxSeqLen)
 *   - Vocab construído do corpus completo antes de instanciar o modelo
 *   - Guard ids.length >= 2 em todos os pares
 *   - Checkpoint a cada 2 epochs (não só no 5)
 *   - nohup-safe: sem \r nos logs principais
 *
 * ORDEM DE EXECUÇÃO:
 *   1. node _akie_corpus_downloader.js   (uma vez — ~10min)
 *   2. node _akie_pretrain.js
 *
 * USO:
 *   node _akie_pretrain.js
 *   PRETRAIN_EPOCHS=10 PRETRAIN_BATCH=16 node _akie_pretrain.js
 */

const fs   = require('fs');
const path = require('path');

const APP_DIR    = '/opt/akie/app';
const CORPUS_DIR = '/opt/akie/corpus';
const OUT_DIR    = '/opt/akie/checkpoints/pretrained';
const LOG_DIR    = '/opt/akie/logs';

const { AKIEModel, HPARAMS }                                   = require(path.join(APP_DIR, '_nexus_neural'));
const { Vocabulary, tokenizeText, makeTrainingPairs, SPECIAL } = require(path.join(APP_DIR, '_akie_vocab'));

// Módulos opcionais
let generateSyntheticConversations = null;
let generateGrammarPairs           = null;

try {
  const s = require(path.join(APP_DIR, '_akie_synthetic'));
  generateSyntheticConversations = s.generateSyntheticConversations || null;
  console.log('[PRETRAIN] _akie_synthetic.js carregado:', !!generateSyntheticConversations);
} catch(e) { console.log('[PRETRAIN] _akie_synthetic.js indisponivel:', e.message); }

try {
  const g = require(path.join(APP_DIR, '_akie_grammar'));
  generateGrammarPairs = g.generateGrammarPairs || null;
  console.log('[PRETRAIN] _akie_grammar.js carregado:', !!generateGrammarPairs);
} catch(e) { console.log('[PRETRAIN] _akie_grammar.js indisponivel:', e.message); }

// Config
const EPOCHS      = parseInt(process.env.PRETRAIN_EPOCHS  || '10');
const BATCH_SIZE  = parseInt(process.env.PRETRAIN_BATCH   || '16');
const LR          = parseFloat(process.env.PRETRAIN_LR    || '0.0005');
const MAX_SEQ_LEN = 128;

// Corpus embutido — garantido mesmo sem downloader
const CORPUS_EMBEDDED = [
  // Identidade — peso 10x (repetido abaixo)
  'sou a akie.','meu nome é akie.','me chamo akie.',
  'sou a akie, uma inteligência artificial.',
  'sou a akie, parte do ecossistema aether.',
  'akie é o meu nome.','não sou o aether, sou a akie.',
  'aether é o ecossistema, eu sou a akie.',
  'pode me chamar de akie.','sou a akie e estou aprendendo.',
  'sou uma inteligência artificial chamada akie.',
  'aprendo com cada conversa.','cada interação me torna melhor.',
  'estou sempre aprendendo.','processo linguagem natural em português.',
  // Estrutura básica
  'o gato está no telhado.','a casa tem três quartos.',
  'o sol nasce no leste.','a lua ilumina a noite.',
  'as crianças brincam no parque.','o homem trabalha todos os dias.',
  'a mulher lê um livro.','o cachorro late para o estranho.',
  'a água é essencial para a vida.','o fogo aquece o ambiente.',
  'as flores crescem na primavera.','o inverno é frio e seco.',
  'o verão é quente e úmido.','a chuva molha o chão.',
  'o vento move as folhas.','o pássaro voa alto no céu.',
  'o peixe nada no rio.','o leão é o rei da selva.',
  'a abelha produz mel.','o cavalo corre pelo campo.',
  // Números
  'um mais um é igual a dois.','dois mais dois é igual a quatro.',
  'há sete dias na semana.','há doze meses no ano.',
  'há vinte e quatro horas no dia.',
  'o primeiro mês do ano é janeiro.',
  'o último mês do ano é dezembro.',
  // Saudações
  'olá, como vai?','bom dia, tudo bem?','boa tarde, como posso ajudar?',
  'boa noite, tudo certo?','muito prazer em conhecer você.',
  'obrigado pela ajuda.','de nada, foi um prazer.',
  'por favor, me ajude com isso.','sim, com certeza.',
  'não, obrigado.','claro, pode perguntar.','seja bem-vindo.',
  'fico feliz em ajudar.','pode contar comigo.',
  'não se preocupe.','fique à vontade.',
  // Brasil
  'o brasil é o maior país da américa do sul.',
  'a língua oficial do brasil é o português.',
  'a capital do brasil é brasília.',
  'são paulo é a maior cidade do brasil.',
  'o rio amazonas é o maior rio do mundo em volume.',
  'o carnaval é uma festa muito popular no brasil.',
  'o futebol é o esporte mais popular do brasil.',
  'o real é a moeda do brasil.',
  'o brasil foi colonizado por portugal.',
  'a independência do brasil foi em 1822.',
  'portugal fica na europa ocidental.',
  'a língua portuguesa é falada em vários países.',
  'angola e moçambique também falam português.',
  'a amazônia é a maior floresta tropical do mundo.',
  'o brasil tem vinte e seis estados e um distrito federal.',
  // Verbos conjugados
  'eu falo português.','você fala português.','ele fala português.',
  'nós falamos português.','vocês falam português.','eles falam português.',
  'eu gosto de aprender.','você precisa de ajuda?','ele quer saber mais.',
  'nós podemos conversar.','eu sei falar.','você consegue entender.',
  'nós vamos aprender juntos.','eu estou estudando.','ela está trabalhando.',
  // Perguntas e respostas
  'qual é o seu nome?','meu nome é akie.',
  'como você está?','estou bem, obrigada.',
  'o que você faz?','converso e aprendo.',
  'qual é a sua função?','minha função é aprender e ajudar.',
  'você pode me ajudar?','sim, posso ajudar.',
  'o que você sabe?','sei conversar em português.',
  'você entende tudo?','ainda estou aprendendo.',
  'o que você não sabe?','há muito que ainda preciso aprender.',
  'qual é a capital do brasil?','a capital do brasil é brasília.',
  // Estruturas complexas
  'quando chove, fico em casa.',
  'se você precisar, pode me perguntar.',
  'embora seja difícil, vou tentar.',
  'porque aprendo, consigo melhorar.',
  'enquanto converso, processo informação.',
  'antes de responder, preciso entender.',
  'assim que souber, vou te dizer.',
  'apesar de não saber tudo, faço o meu melhor.',
  'para aprender mais, preciso de exemplos.',
  'não só aprendo, mas também ensino.',
  // Negação e dúvida
  'não sei a resposta.','não tenho certeza.','talvez seja isso.',
  'não é bem assim.','pode ser que sim.','pode ser que não.',
  'não tenho informação sobre isso.',
  'ainda estou aprendendo sobre esse tema.',
  'preciso de mais informações.','não entendi a pergunta.',
  'pode reformular a pergunta?',
  // Tecnologia
  'inteligência artificial é a capacidade de máquinas aprenderem.',
  'redes neurais são inspiradas no cérebro humano.',
  'o aprendizado de máquina usa dados para melhorar.',
  'processamento de linguagem natural permite que computadores entendam texto.',
  'o modelo aprende padrões através de exemplos de treinamento.',
  'treinamento envolve ajustar os pesos da rede neural.',
  'embeddings representam palavras como vetores numéricos densos.',
  'o vocabulário define quais palavras o modelo reconhece.',
  'transformers são arquiteturas poderosas para processamento de linguagem.',
  'atenção é o mecanismo central dos modelos transformer.',
  // Frases longas
  'a inteligência artificial é uma área da ciência da computação dedicada ao desenvolvimento de sistemas capazes de realizar tarefas que normalmente requerem inteligência humana.',
  'o processamento de linguagem natural permite que computadores entendam, interpretem e gerem texto em linguagem humana de forma eficiente e precisa.',
  'redes neurais artificiais são sistemas computacionais inspirados nas redes neurais biológicas do cérebro humano, capazes de aprender padrões complexos a partir de dados.',
  'o brasil é um país de dimensões continentais com grande diversidade cultural, linguística e biológica distribuída entre seus vinte e seis estados.',
  'a amazônia abriga a maior biodiversidade do planeta e desempenha papel fundamental na regulação do clima global através do ciclo hidrológico.',
];

// Identidade com peso 10x
const IDENTITY = [
  'sou a akie.','meu nome é akie.','me chamo akie.',
  'sou a akie, uma inteligência artificial.',
  'sou a akie, parte do ecossistema aether.',
  'akie é o meu nome.','não sou o aether, sou a akie.',
  'aether é o ecossistema, eu sou a akie.',
  'pode me chamar de akie.','sou a akie e estou aprendendo.',
];

function loadCorpusFile(fpath) {
  if (!fs.existsSync(fpath)) return [];
  return fs.readFileSync(fpath, 'utf8')
    .split('\n')
    .map(l => l.trim().replace(/^[ua]:\s*/i, ''))
    .filter(l => l.length > 5 && l.length < 500);
}

function makePairs(sentences, vocab, maxLen) {
  const pairs = [];
  for (const s of sentences) {
    const ids = vocab.tokenize(s);
    if (!ids || ids.length < 2) continue;
    pairs.push(...makeTrainingPairs(ids, maxLen));
  }
  return pairs;
}

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length-1; i > 0; i--) {
    const j = Math.floor(Math.random()*(i+1));
    [a[i],a[j]] = [a[j],a[i]];
  }
  return a;
}

async function trainEpoch(model, pairs, batchSize, epoch, total) {
  const data = shuffle(pairs);
  let loss = 0, acc = 0, steps = 0;
  for (let i = 0; i < data.length; i += batchSize) {
    const batch = data.slice(i, i+batchSize);
    if (!batch.length) continue;
    const r = await model.trainBatch(batch, 1);
    if (r.loss !== null && isFinite(r.loss)) { loss += r.loss; steps++; }
    if (r.accuracy !== null && isFinite(r.accuracy)) acc += r.accuracy;
    if (steps > 0 && steps % 100 === 0) {
      console.log(`[PRETRAIN] Epoch ${epoch}/${total} step=${steps} loss=${(loss/steps).toFixed(4)} acc=${((acc/steps)*100).toFixed(1)}%`);
    }
  }
  return { loss: steps>0 ? loss/steps : null, acc: steps>0 ? acc/steps : null, steps };
}

async function main() {
  console.log('\n[PRETRAIN v4.0] ============================================');
  console.log('[PRETRAIN v4.0] AKIE Pre-treinamento — reinicio do zero');
  console.log(`[PRETRAIN v4.0] Epochs=${EPOCHS} Batch=${BATCH_SIZE} LR=${LR}`);
  console.log('[PRETRAIN v4.0] ============================================\n');

  fs.mkdirSync(OUT_DIR,  { recursive: true });
  fs.mkdirSync(LOG_DIR,  { recursive: true });

  // 1. Carregar corpus externo
  const extSentences = loadCorpusFile(path.join(CORPUS_DIR, 'sentences_ptbr.txt'));
  const extDialogs   = loadCorpusFile(path.join(CORPUS_DIR, 'dialogs_ptbr.txt'));
  console.log(`[PRETRAIN v4.0] Corpus externo: ${extSentences.length} frases + ${extDialogs.length} diálogos`);

  // 2. Corpus total para construir vocab
  const allSentences = [
    ...CORPUS_EMBEDDED,
    ...Array(9).fill(IDENTITY).flat(), // identidade 10x total
    ...extSentences,
    ...extDialogs,
  ];
  console.log(`[PRETRAIN v4.0] Total corpus: ${allSentences.length} frases`);

  // 3. Construir vocab
  const vocab = new Vocabulary();
  for (const s of allSentences) {
    const tokens = tokenizeText(s);
    if (tokens) vocab.addTokens(tokens);
  }
  console.log(`[PRETRAIN v4.0] Vocabulário: ${vocab.size} tokens`);

  if (vocab.size < 1000) {
    console.warn('[PRETRAIN v4.0] AVISO: vocab < 1000 — rode _akie_corpus_downloader.js primeiro');
    console.warn('[PRETRAIN v4.0] Continuando com corpus embutido...');
  }

  // 4. Instanciar modelo com vocab real
  const hparams = {
    ...HPARAMS,
    vocabSize:    vocab.size,
    learningRate: LR,
    maxSeqLen:    MAX_SEQ_LEN,
    batchSize:    BATCH_SIZE,
  };
  const model = new AKIEModel(vocab, hparams);
  model.build();
  const paramCount = model.model.countParams();
  console.log(`[PRETRAIN v4.0] Modelo: ${paramCount.toLocaleString()} parâmetros\n`);

  // 5. Gerar pares de treino
  console.log('[PRETRAIN v4.0] Gerando pares de treino...');

  const pairsEmbedded = makePairs(CORPUS_EMBEDDED, vocab, MAX_SEQ_LEN);
  const pairsIdentity = makePairs(Array(9).fill(IDENTITY).flat(), vocab, MAX_SEQ_LEN);
  const pairsExternal = makePairs([...extSentences, ...extDialogs], vocab, MAX_SEQ_LEN);

  let pairsSynthetic = [];
  if (generateSyntheticConversations) {
    try {
      pairsSynthetic = await generateSyntheticConversations(vocab, 200, MAX_SEQ_LEN);
      if (!Array.isArray(pairsSynthetic)) pairsSynthetic = [];
      console.log(`[PRETRAIN v4.0] Synthetic: ${pairsSynthetic.length} pares`);
    } catch(e) { console.log(`[PRETRAIN v4.0] Synthetic erro: ${e.message}`); }
  }

  let pairsGrammar = [];
  if (generateGrammarPairs) {
    try {
      pairsGrammar = generateGrammarPairs(vocab, 500, MAX_SEQ_LEN);
      if (!Array.isArray(pairsGrammar)) pairsGrammar = [];
      console.log(`[PRETRAIN v4.0] Grammar: ${pairsGrammar.length} pares`);
    } catch(e) { console.log(`[PRETRAIN v4.0] Grammar erro: ${e.message}`); }
  }

  const allPairs = [
    ...pairsEmbedded,
    ...pairsIdentity,
    ...pairsExternal,
    ...pairsSynthetic,
    ...pairsGrammar,
  ];

  console.log(`\n[PRETRAIN v4.0] Pares de treino:`);
  console.log(`  Embutido:  ${pairsEmbedded.length.toLocaleString()}`);
  console.log(`  Identidade: ${pairsIdentity.length.toLocaleString()}`);
  console.log(`  Externo:   ${pairsExternal.length.toLocaleString()}`);
  console.log(`  Synthetic: ${pairsSynthetic.length.toLocaleString()}`);
  console.log(`  Grammar:   ${pairsGrammar.length.toLocaleString()}`);
  console.log(`  TOTAL:     ${allPairs.length.toLocaleString()}\n`);

  if (allPairs.length < 500) {
    console.error('[PRETRAIN v4.0] ERRO: menos de 500 pares — verifique _akie_vocab.js e corpus');
    process.exit(1);
  }

  // 6. Treinar
  const t0 = Date.now();

  for (let epoch = 1; epoch <= EPOCHS; epoch++) {
    const t1 = Date.now();
    const result = await trainEpoch(model, allPairs, BATCH_SIZE, epoch, EPOCHS);
    const elapsed = ((Date.now()-t1)/1000).toFixed(1);
    const lossStr = result.loss  !== null ? result.loss.toFixed(4)               : 'n/a';
    const accStr  = result.acc   !== null ? (result.acc*100).toFixed(1)+'%'      : 'n/a';

    console.log(`\n[PRETRAIN v4.0] Epoch ${epoch}/${EPOCHS} | loss=${lossStr} acc=${accStr} | steps=${result.steps} | ${elapsed}s`);

    // Checkpoint a cada 2 epochs e no final
    if (epoch % 2 === 0 || epoch === EPOCHS) {
      const ckpt = path.join(OUT_DIR, `checkpoint_epoch_${epoch}`);
      await model.save(ckpt);
      console.log(`[PRETRAIN v4.0] Checkpoint: ${ckpt}`);
    }
  }

  // 7. Salvar modelo final + vocab
  console.log('\n[PRETRAIN v4.0] Salvando modelo final...');
  await model.save(OUT_DIR);
  fs.writeFileSync(
    path.join(OUT_DIR, 'akie_vocab.json'),
    JSON.stringify(vocab.toJSON(), null, 2)
  );

  const totalMin = ((Date.now()-t0)/1000/60).toFixed(1);
  console.log(`\n[PRETRAIN v4.0] ============================================`);
  console.log(`[PRETRAIN v4.0] CONCLUIDO em ${totalMin} minutos`);
  console.log(`[PRETRAIN v4.0] Params:  ${paramCount.toLocaleString()}`);
  console.log(`[PRETRAIN v4.0] Vocab:   ${vocab.size} tokens`);
  console.log(`[PRETRAIN v4.0] Pares:   ${allPairs.length.toLocaleString()}`);
  console.log(`[PRETRAIN v4.0] Output:  ${OUT_DIR}`);
  console.log(`[PRETRAIN v4.0] ============================================`);
  console.log(`\n[PRETRAIN v4.0] PROXIMOS PASSOS:`);
  console.log(`  cp -r ${OUT_DIR}/* /opt/akie/model/`);
  console.log(`  pm2 restart akie-worker\n`);
}

main().catch(e => {
  console.error('\n[PRETRAIN v4.0] ERRO FATAL:', e.message);
  console.error(e.stack);
  process.exit(1);
});
