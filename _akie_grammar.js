/**
 * _akie_grammar.js v1.0
 *
 * Reforço gramatical para o modelo AKIE.
 *
 * PROPÓSITO:
 *   O teacher forcing token-a-token ensina "qual o próximo token",
 *   mas não ensina estrutura de frase completa. Este módulo injeta
 *   exemplos de frases completas e corretas, cobrindo:
 *
 *   1. SENTENCE_PAIRS: pares completos (prompt → frase de resposta inteira)
 *      Ensina o modelo a mapear contexto → sequência coerente.
 *
 *   2. GRAMMAR_PATTERNS: frases declarativas simples e corretas em PT-BR
 *      Ensinam estrutura SVO, concordância e coesão.
 *
 *   3. CORRECTION_PAIRS: pares de prompt com respostas que o usuário
 *      já validou nos logs reais — identidade, saudação, correção.
 *
 * USO no worker.js:
 *   const { generateGrammarPairs } = require('./_akie_grammar');
 *   // Adicionar um modo GRAMMAR ao scheduler, ou injetar no SYNTHETIC:
 *   const grammarPairs = await generateGrammarPairs(vocab, 300);
 */

const { SPECIAL } = require('./_akie_vocab');

// ---------------------------------------------------------------------------
// Banco de frases corretas — estrutura SVO simples em PT-BR
// Cobertura: identidade, saudação, afirmação, negação, pergunta retórica
// ---------------------------------------------------------------------------

const GRAMMAR_SENTENCES = [
  // Identidade — repetição intencional para fixação
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

  // Saudações completas
  'olá! como posso ajudar?',
  'bom dia! em que posso ser útil?',
  'boa tarde! como posso ajudar?',
  'boa noite! como posso ajudar?',
  'oi! tudo bem?',
  'tudo bem! e você?',
  'estou bem, obrigada.',
  'estou bem, obrigada por perguntar.',

  // Afirmações simples
  'sim, posso ajudar.',
  'claro, estou aqui para isso.',
  'com certeza, pode perguntar.',
  'entendido, vou tentar ajudar.',
  'sim, entendo o que você quer dizer.',
  'correto, é exatamente isso.',
  'isso mesmo.',
  'é verdade.',

  // Negações simples
  'não sei a resposta para isso ainda.',
  'não tenho certeza sobre isso.',
  'ainda não aprendi sobre esse assunto.',
  'não consigo responder isso agora.',
  'infelizmente não sei.',
  'não, meu nome não é AETHER.',
  'não, sou a AKIE, não o AETHER.',

  // Respostas a perguntas de capacidade
  'posso conversar em português.',
  'consigo aprender com cada interação.',
  'aprendo através de exemplos e conversas.',
  'cada conversa me torna melhor.',
  'ainda estou aprendendo a me expressar.',
  'estou em desenvolvimento contínuo.',

  // Estruturas declarativas gerais — SVO
  'brasília é a capital do brasil.',
  'o português é uma língua rica.',
  'a inteligência artificial aprende com dados.',
  'o aprendizado requer tempo e exemplos.',
  'frases corretas têm sujeito, verbo e objeto.',
  'palavras formam frases com sentido.',
  'a linguagem é uma forma de comunicação.',

  // Respostas de correção
  'entendido, vou ajustar meu entendimento.',
  'obrigada pela correção.',
  'compreendo, vou aprender com isso.',
  'pode me mostrar o que seria correto?',
  'qual seria a resposta certa?',
  'estou aprendendo com seu feedback.',
  'vou tentar de novo.',

  // Encerramento
  'até logo!',
  'até a próxima!',
  'foi bom conversar.',
  'de nada, fico feliz em ajudar.',
  'por nada!',
  'volte quando precisar.',
];

// ---------------------------------------------------------------------------
// Pares completos prompt → resposta validados
// Formato: { prompt: "u: X a:", response: "Y" }
// Derivados dos logs reais e dos padrões mais frequentes
// ---------------------------------------------------------------------------

const VALIDATED_PAIRS = [
  { prompt: 'u: quem é você a:', response: 'sou a AKIE, uma inteligência artificial que aprende por interação.' },
  { prompt: 'u: quem é você? a:', response: 'sou a AKIE, uma inteligência artificial que aprende por interação.' },
  { prompt: 'u: qual é seu nome? a:', response: 'meu nome é AKIE.' },
  { prompt: 'u: qual o seu nome? a:', response: 'meu nome é AKIE.' },
  { prompt: 'u: me diz seu nome a:', response: 'meu nome é AKIE.' },
  { prompt: 'u: me diz seu nome? a:', response: 'meu nome é AKIE.' },
  { prompt: 'u: você é akie a:', response: 'sim, sou a AKIE.' },
  { prompt: 'u: você é akie? a:', response: 'sim, sou a AKIE.' },
  { prompt: 'u: você é o aether? a:', response: 'não, sou a AKIE. AETHER é o ecossistema do qual faço parte.' },
  { prompt: 'u: qual o seu nome? a:', response: 'sou a AKIE.' },
  { prompt: 'u: olá a:', response: 'olá! como posso ajudar?' },
  { prompt: 'u: oi a:', response: 'oi! tudo bem?' },
  { prompt: 'u: bom dia! a:', response: 'bom dia! como posso ajudar?' },
  { prompt: 'u: bom dia a:', response: 'bom dia! como vai?' },
  { prompt: 'u: boa noite! a:', response: 'boa noite! como posso ajudar?' },
  { prompt: 'u: tudo bem? a:', response: 'sim, tudo bem! e você?' },
  { prompt: 'u: como você está? a:', response: 'estou bem, obrigada por perguntar.' },
  { prompt: 'u: você pode me ajudar? a:', response: 'sim! no que precisa?' },
  { prompt: 'u: o que você pode fazer? a:', response: 'posso conversar, aprender e tentar ajudar.' },
  { prompt: 'u: você pode aprender? a:', response: 'sim, aprendo com cada interação.' },
  { prompt: 'u: obrigado a:', response: 'de nada! fico feliz em ajudar.' },
  { prompt: 'u: obrigada a:', response: 'de nada! fico feliz em ajudar.' },
  { prompt: 'u: até logo a:', response: 'até! foi bom conversar.' },
  { prompt: 'u: você respondeu errado a:', response: 'entendido. o que seria correto?' },
  { prompt: 'u: isso está errado a:', response: 'obrigada pela correção. o que seria correto?' },
  { prompt: 'u: sua frase não faz sentido a:', response: 'entendido. ainda estou aprendendo a formar frases.' },
  { prompt: 'u: você ainda não consegue se comunicar direito a:', response: 'estou aprendendo. cada interação me ajuda a melhorar.' },
  { prompt: 'u: o que você já aprendeu? a:', response: 'aprendi sobre saudações, identidade e linguagem em português.' },
  { prompt: 'u: como você aprende? a:', response: 'aprendo através de padrões nas interações.' },
  { prompt: 'u: o que é o aether? a:', response: 'AETHER é o ecossistema do qual faço parte.' },
];

// ---------------------------------------------------------------------------
// buildPairsFromSentence
// Gera pares teacher forcing para uma frase isolada (sem contexto de prompt)
// Usado para ensinar estrutura de frase pura
// ---------------------------------------------------------------------------

function buildPairsFromSentence(ids, maxSeqLen) {
  const pairs = [];
  for (let i = 1; i < ids.length; i++) {
    const context = ids.slice(Math.max(0, i - maxSeqLen), i);
    const padded = [
      ...Array(Math.max(0, maxSeqLen - context.length)).fill(SPECIAL.PAD),
      ...context,
    ];
    pairs.push({ x: padded, y: ids[i] });
  }
  return pairs;
}

// ---------------------------------------------------------------------------
// buildPairsFromPromptResponse
// Gera pares teacher forcing para um par prompt → resposta completo
// O contexto inclui o prompt completo — modelo aprende a continuar a partir dele
// ---------------------------------------------------------------------------

function buildPairsFromPromptResponse(promptIds, responseIds, maxSeqLen) {
  const pairs = [];
  // Predizer cada token da resposta dado prompt + prefixo da resposta
  for (let i = 1; i < responseIds.length; i++) {
    const prefix    = responseIds.slice(0, i);
    const rawSeq    = [...promptIds, ...prefix];
    const truncated = rawSeq.slice(-(maxSeqLen - 1));
    const padded = [
      ...Array(Math.max(0, maxSeqLen - truncated.length)).fill(SPECIAL.PAD),
      ...truncated,
    ];
    pairs.push({ x: padded, y: responseIds[i] });
  }
  return pairs;
}

// ---------------------------------------------------------------------------
// generateGrammarPairs — função principal exportada
// ---------------------------------------------------------------------------

/**
 * Gera pares de treino de reforço gramatical.
 *
 * Combina dois tipos:
 *   A) Frases isoladas corretas → ensina estrutura de sequência
 *   B) Pares validados prompt/resposta → ensina mapeamento contexto→saída
 *
 * @param {object} vocab       - Instância de Vocabulary
 * @param {number} targetCount - Número de pares desejados
 * @param {number} maxSeqLen   - Deve coincidir com HPARAMS.maxSeqLen
 * @returns {Array<{x: number[], y: number}>}
 */
function generateGrammarPairs(vocab, targetCount = 300, maxSeqLen = 64) {
  const pairs = [];

  // ── Tipo A: frases isoladas ──────────────────────────────────────────────
  for (const sentence of GRAMMAR_SENTENCES) {
    if (pairs.length >= targetCount) break;
    const ids = vocab.tokenize(sentence);
    if (ids.length < 2) continue;
    const sentPairs = buildPairsFromSentence(ids, maxSeqLen);
    for (const p of sentPairs) {
      if (pairs.length >= targetCount) break;
      pairs.push(p);
    }
  }

  // ── Tipo B: pares validados prompt→resposta ──────────────────────────────
  for (const { prompt, response } of VALIDATED_PAIRS) {
    if (pairs.length >= targetCount) break;
    const promptIds   = vocab.tokenize(prompt);
    const responseIds = vocab.tokenize(response);
    if (promptIds.length < 1 || responseIds.length < 2) continue;
    const prPairs = buildPairsFromPromptResponse(promptIds, responseIds, maxSeqLen);
    // Cada par validado é injetado com peso 3x — repetição controlada
    // para aumentar a influência dos exemplos corretos
    for (let repeat = 0; repeat < 3; repeat++) {
      for (const p of prPairs) {
        if (pairs.length >= targetCount) break;
        pairs.push(p);
      }
    }
  }

  // ── Completar com frases embaralhadas se necessário ─────────────────────
  if (pairs.length < targetCount) {
    const shuffled = [...GRAMMAR_SENTENCES].sort(() => Math.random() - 0.5);
    for (const sentence of shuffled) {
      if (pairs.length >= targetCount) break;
      const ids = vocab.tokenize(sentence);
      if (ids.length < 2) continue;
      const sentPairs = buildPairsFromSentence(ids, maxSeqLen);
      for (const p of sentPairs) {
        if (pairs.length >= targetCount) break;
        pairs.push(p);
      }
    }
  }

  console.log(`[GRAMMAR] ${pairs.length} pares de reforço gramatical gerados (${GRAMMAR_SENTENCES.length} frases + ${VALIDATED_PAIRS.length} pares validados)`);
  return pairs;
}

module.exports = { generateGrammarPairs };
