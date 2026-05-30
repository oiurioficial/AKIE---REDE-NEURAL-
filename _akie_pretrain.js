/**
 * _akie_pretrain.js v1.0
 *
 * PRÉ-TREINAMENTO OFFLINE — rodar UMA VEZ no VPS antes de iniciar o worker.
 *
 * OBJETIVO:
 *   Estabelecer uma base linguística sólida em PT-BR antes do aprendizado
 *   contínuo. Equivale ao "pre-training" de LLMs — sem isso, o modelo
 *   aprende linguagem e conteúdo ao mesmo tempo, o que é ineficiente.
 *
 * O QUE FAZ:
 *   1. Constrói vocabulário a partir de corpus PT-BR curado (~5000 tokens)
 *   2. Treina o modelo em múltiplas épocas sobre corpus de frases corretas
 *   3. Inclui corpus de identidade com peso reforçado
 *   4. Salva checkpoint completo e portável
 *
 * USO:
 *   node _akie_pretrain.js
 *   # ou com configuração:
 *   PRETRAIN_EPOCHS=10 PRETRAIN_BATCH=16 node _akie_pretrain.js
 *
 * TEMPO ESTIMADO (VPS 2 vCPU / 8GB RAM):
 *   ~30-60 minutos para configuração padrão
 *   ~2-4 horas para configuração extendida (epochs=20, corpus completo)
 *
 * APÓS CONCLUIR:
 *   Copiar /data/akie_pretrained/ para /data/akie_model/ e iniciar o worker.
 */

require('dotenv').config();
const fs   = require('fs');
const path = require('path');

const { AKIEModel }            = require('./_nexus_neural');
const { Vocabulary, tokenizeText, makeTrainingPairs, SPECIAL } = require('./_akie_vocab');
const { generateGrammarPairs } = require('./_akie_grammar');

// ---------------------------------------------------------------------------
// Configuração
// ---------------------------------------------------------------------------

const CONFIG = {
  outputDir:   process.env.PRETRAIN_DIR    || '/data/akie_pretrained',
  epochs:      parseInt(process.env.PRETRAIN_EPOCHS  || '10'),
  batchSize:   parseInt(process.env.PRETRAIN_BATCH   || '32'),
  learningRate: parseFloat(process.env.PRETRAIN_LR   || '0.0005'),
  maxSeqLen:   128,
  hparams: {
    embDim:       512,
    hiddenSize:   2048,
    maxSeqLen:    128,
    numLayers:    6,
    batchSize:    32,
    learningRate: 0.0005,
  },
};

// ---------------------------------------------------------------------------
// Corpus PT-BR curado para pré-treinamento
// Estrutura: frases declarativas corretas, diversas, sem ruído
// ---------------------------------------------------------------------------

const PRETRAIN_CORPUS = [

  // ── Linguagem básica e estrutura ─────────────────────────────────────────
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

  // ── Identidade e IA — alto peso ──────────────────────────────────────────
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

  // ── Frases sobre aprendizado ──────────────────────────────────────────────
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

  // ── Frases longas e complexas para maxSeqLen=64 ───────────────────────────
  'a inteligência artificial é uma área da ciência da computação que se dedica ao desenvolvimento de sistemas capazes de realizar tarefas que normalmente requerem inteligência humana.',
  'o processamento de linguagem natural permite que computadores entendam, interpretem e gerem texto em linguagem humana de forma eficiente e precisa.',
  'o aprendizado de máquina é um subcampo da inteligência artificial que permite que sistemas aprendam e melhorem automaticamente a partir da experiência.',
  'redes neurais artificiais são sistemas computacionais inspirados nas redes neurais biológicas do cérebro humano, capazes de aprender padrões complexos.',
  'o desenvolvimento de assistentes virtuais inteligentes tem avançado rapidamente nos últimos anos, tornando a interação humano-computador cada vez mais natural.',
];

// ---------------------------------------------------------------------------
// Corpus de identidade com peso extra (repetido para reforço)
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

function buildVocabularyFromCorpus(corpus) {
  const vocab = new Vocabulary();
  for (const sentence of corpus) {
    const tokens = tokenizeText(sentence);
    vocab.addTokens(tokens);
  }
  console.log(`[PRETRAIN] Vocabulário construído: ${vocab.size} tokens`);
  return vocab;
}

async function trainEpoch(model, pairs, batchSize, epochNum) {
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

    if (steps % 50 === 0) {
      const loss = steps > 0 ? (totalLoss / steps).toFixed(4) : 'n/a';
      const acc  = steps > 0 ? ((totalAcc  / steps) * 100).toFixed(1) + '%' : 'n/a';
      process.stdout.write(`\r[PRETRAIN] Epoch ${epochNum} | Step ${steps} | loss=${loss} acc=${acc}    `);
    }
  }

  const avgLoss = steps > 0 ? totalLoss / steps : null;
  const avgAcc  = steps > 0 ? totalAcc  / steps : null;
  return { loss: avgLoss, accuracy: avgAcc };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log('\n╔════════════════════════════════════════════════════╗');
  console.log('║        AKIE PRÉ-TREINAMENTO OFFLINE v3.0 (~30M params)          ║');
  console.log('╚════════════════════════════════════════════════════╝\n');
  console.log(`[PRETRAIN] Epochs: ${CONFIG.epochs} | BatchSize: ${CONFIG.batchSize} | LR: ${CONFIG.learningRate}`);
  console.log(`[PRETRAIN] Output: ${CONFIG.outputDir}\n`);

  // 1. Construir vocabulário completo a partir do corpus
  const fullCorpus = [
    ...PRETRAIN_CORPUS,
    ...Array(5).fill(IDENTITY_CORPUS_WEIGHTED).flat(), // identidade 5x no vocab
  ];
  const vocab = buildVocabularyFromCorpus(fullCorpus);

  // 2. Criar modelo
  const hparams = { ...CONFIG.hparams, learningRate: CONFIG.learningRate };
  const model   = new AKIEModel(vocab, hparams);
  model.build();
  console.log(`[PRETRAIN] Modelo criado: ${model.model.countParams().toLocaleString()} parâmetros\n`);

  // 3. Gerar pares de treino
  console.log('[PRETRAIN] Gerando pares de treino...');

  // A) Corpus geral
  const corpusPairs = [];
  for (const sentence of PRETRAIN_CORPUS) {
    const ids   = vocab.tokenize(sentence);
    const pairs = makeTrainingPairs(ids, CONFIG.maxSeqLen);
    corpusPairs.push(...pairs);
  }

  // B) Identidade com peso 10x
  const identityPairs = [];
  for (let repeat = 0; repeat < 10; repeat++) {
    for (const sentence of IDENTITY_CORPUS_WEIGHTED) {
      const ids   = vocab.tokenize(sentence);
      const pairs = makeTrainingPairs(ids, CONFIG.maxSeqLen);
      identityPairs.push(...pairs);
    }
  }

  // C) Pares gramaticais validados
  const grammarPairs = generateGrammarPairs(vocab, 500, CONFIG.maxSeqLen);

  const allPairs = [...corpusPairs, ...identityPairs, ...grammarPairs];
  console.log(`[PRETRAIN] Total de pares: ${allPairs.length.toLocaleString()}`);
  console.log(`  Corpus geral:   ${corpusPairs.length.toLocaleString()}`);
  console.log(`  Identidade 10x: ${identityPairs.length.toLocaleString()}`);
  console.log(`  Gramatical:     ${grammarPairs.length.toLocaleString()}\n`);

  // 4. Loop de treinamento
  const t0 = Date.now();

  for (let epoch = 1; epoch <= CONFIG.epochs; epoch++) {
    const epochStart = Date.now();
    const result     = await trainEpoch(model, allPairs, CONFIG.batchSize, epoch);
    const elapsed    = ((Date.now() - epochStart) / 1000).toFixed(1);
    const loss       = result.loss     ? result.loss.toFixed(4)              : 'n/a';
    const acc        = result.accuracy ? (result.accuracy * 100).toFixed(1) + '%' : 'n/a';

    console.log(`\n[PRETRAIN] ✓ Epoch ${epoch}/${CONFIG.epochs} | loss=${loss} acc=${acc} | ${elapsed}s`);

    // Salvar checkpoint a cada 5 épocas
    if (epoch % 5 === 0 || epoch === CONFIG.epochs) {
      const ckptDir = path.join(CONFIG.outputDir, `checkpoint_epoch_${epoch}`);
      await model.save(ckptDir);
      console.log(`[PRETRAIN] Checkpoint salvo: ${ckptDir}`);
    }
  }

  // 5. Salvar modelo final
  console.log('\n[PRETRAIN] Salvando modelo final...');
  await model.save(CONFIG.outputDir);

  // Salvar vocab separadamente na raiz do output
  const vocabPath = path.join(CONFIG.outputDir, 'akie_vocab.json');
  fs.writeFileSync(vocabPath, JSON.stringify(vocab.toJSON(), null, 2));

  const totalTime = ((Date.now() - t0) / 1000 / 60).toFixed(1);
  console.log(`\n╔════════════════════════════════════════════════════╗`);
  console.log(`║  PRÉ-TREINAMENTO CONCLUÍDO em ${totalTime} minutos`);
  console.log(`║  Modelo salvo em: ${CONFIG.outputDir}`);
  console.log(`║  Parâmetros: ${model.model.countParams().toLocaleString()}`);
  console.log(`║  Vocab: ${vocab.size} tokens`);
  console.log(`║  Steps: ${model.trainSteps.toLocaleString()}`);
  console.log(`╚════════════════════════════════════════════════════╝`);
  console.log(`\n[PRÓXIMO PASSO]`);
  console.log(`  cp -r ${CONFIG.outputDir}/* /data/akie_model/`);
  console.log(`  # Depois iniciar o worker normalmente\n`);
}

main().catch(err => {
  console.error('[PRETRAIN] Erro fatal:', err);
  process.exit(1);
});
