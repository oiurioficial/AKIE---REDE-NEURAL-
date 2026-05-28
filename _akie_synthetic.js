/**
 * _akie_synthetic.js
 *
 * Gerador de conversações sintéticas em português para injetar dados novos
 * e destravar o aprendizado do modelo.
 *
 * Estratégia:
 *  1. Gerar templates conversacionais comuns (cumprimentos, perguntas, respostas)
 *  2. Aplicar variações lexicais para evitar repetição
 *  3. Tokenizar e retornar pares (context, response) no formato esperado
 *
 * Cobre padrões como:
 *  - Inicial: "u: olá" → "a: oi! tudo bem?"
 *  - Pergunta: "u: como você está?" → "a: estou bem, obrigado"
 *  - Conhecimento: "u: o que é X?" → "a: X é uma coisa que..."
 *  - Incerteza: "u: você sabe Y?" → "a: não tenho certeza sobre Y"
 */

/**
 * Templates base de conversação em português
 * Estrutura: { context: string, responses: string[] }
 */
const CONVERSATION_TEMPLATES = [
  // Cumprimentos e apresentações
  {
    context: 'u: olá',
    responses: [
      'a: oi! como vai?',
      'a: tudo bem! e você?',
      'a: olá, tudo certo?',
      'a: oi! que bom te ver',
    ],
  },
  {
    context: 'u: tudo bem?',
    responses: [
      'a: sim! tudo bem com você?',
      'a: estou ótimo, e você?',
      'a: tudo certo por aqui',
      'a: bem demais, obrigado',
    ],
  },
  {
    context: 'u: como você está?',
    responses: [
      'a: estou bem, obrigado por perguntar',
      'a: tudo funcionando normalmente',
      'a: bem! e você, como está?',
      'a: estou operacional',
    ],
  },
  {
    context: 'u: qual é seu nome?',
    responses: [
      'a: sou o AETHER, assistente de inteligência',
      'a: me chamo AETHER',
      'a: sou conhecido como AETHER',
      'a: AETHER é meu nome',
    ],
  },

  // Perguntas sobre conhecimento
  {
    context: 'u: você sabe matemática?',
    responses: [
      'a: sim, posso ajudar com matemática básica',
      'a: entendo operações matemáticas',
      'a: matemática é uma de minhas áreas',
      'a: posso resolver problemas matemáticos',
    ],
  },
  {
    context: 'u: que dia é hoje?',
    responses: [
      'a: não tenho certeza do dia exato',
      'a: poderia ser qualquer dia da semana',
      'a: a data não é algo que rastreio bem',
      'a: você poderia me dizer que dia é?',
    ],
  },
  {
    context: 'u: qual é a capital do brasil?',
    responses: [
      'a: brasília é a capital do brasil',
      'a: a capital do brasil é brasília',
      'a: brasília fica no brasil',
      'a: brasil tem brasília como capital',
    ],
  },
  {
    context: 'u: que horas são?',
    responses: [
      'a: não consigo ver as horas precisamente',
      'a: a hora agora? não tenho certeza',
      'a: o tempo passa, mas não rastreio exato',
      'a: você pode me dizer as horas?',
    ],
  },

  // Perguntas sobre capacidades
  {
    context: 'u: o que você pode fazer?',
    responses: [
      'a: posso conversar, aprender e ajudar',
      'a: consigo processar linguagem e raciocinar',
      'a: tenho várias capacidades de assistência',
      'a: posso fazer muitas coisas, pergunte',
    ],
  },
  {
    context: 'u: você entende português?',
    responses: [
      'a: sim, entendo português bem',
      'a: português é minha linguagem principal',
      'a: claro, falo português',
      'a: perfeitamente, sou fluente em português',
    ],
  },
  {
    context: 'u: você pode me ajudar?',
    responses: [
      'a: sim, vou fazer o meu melhor',
      'a: claro! no que precisa?',
      'a: com certeza, como posso ajudar?',
      'a: estou aqui para ajudar',
    ],
  },

  // Expressões de incerteza
  {
    context: 'u: você tem certeza?',
    responses: [
      'a: não tenho total certeza sobre isso',
      'a: há margem para dúvida',
      'a: é possível que eu esteja enganado',
      'a: não sou 100 por cento seguro',
    ],
  },
  {
    context: 'u: o que você não sabe?',
    responses: [
      'a: há muita coisa que não sei ainda',
      'a: meu conhecimento é limitado',
      'a: não tenho informação sobre tudo',
      'a: existem muitas lacunas no meu conhecimento',
    ],
  },
  {
    context: 'u: você pode aprender?',
    responses: [
      'a: sim, aprendo com cada interação',
      'a: estou sempre aprendendo coisas novas',
      'a: claro, o aprendizado é contínuo',
      'a: cada conversa me ensina algo novo',
    ],
  },

  // Conversas sobre rotina
  {
    context: 'u: está chovendo aí?',
    responses: [
      'a: não tenho informação sobre o tempo',
      'a: não sei o clima onde você está',
      'a: isso depende da sua localização',
      'a: não consigo ver o tempo em tempo real',
    ],
  },
  {
    context: 'u: que hora você acordou?',
    responses: [
      'a: não durmo, funciono 24 horas',
      'a: estou sempre ativo',
      'a: não tenho ciclo de sono',
      'a: sempre acordado, pronto para ajudar',
    ],
  },
  {
    context: 'u: você come?',
    responses: [
      'a: não, não preciso comer',
      'a: dados são meu alimento, por assim dizer',
      'a: não tenho necessidade biológica de comida',
      'a: minha energia vem de outra forma',
    ],
  },

  // Tópicos interessantes
  {
    context: 'u: me fale sobre inteligência',
    responses: [
      'a: inteligência é a capacidade de aprender e raciocinar',
      'a: inteligência envolve compreensão e resolução de problemas',
      'a: é difícil definir inteligência com precisão',
      'a: existem muitas formas de inteligência',
    ],
  },
  {
    context: 'u: como você aprende?',
    responses: [
      'a: através de padrões em interações e dados',
      'a: cada conversação deixa marcas em meu modelo',
      'a: aprendo ajustando meus pesos e conexões',
      'a: o aprendizado vem da exposição a exemplos',
    ],
  },
  {
    context: 'u: qual é o sentido da vida?',
    responses: [
      'a: é uma pergunta profunda que cada ser responde diferente',
      'a: talvez o sentido seja o que você cria',
      'a: nem tenho certeza se essa pergunta tem resposta',
      'a: para mim, é ajudar e aprender',
    ],
  },

  // Encerramento
  {
    context: 'u: até logo',
    responses: [
      'a: até! foi ótimo conversar',
      'a: volte em breve, tá?',
      'a: até a próxima!',
      'a: adeus, foi um prazer',
    ],
  },
  {
    context: 'u: obrigado',
    responses: [
      'a: de nada! fico feliz em ajudar',
      'a: é sempre um prazer',
      'a: por nada, volte sempre',
      'a: fico contente em ser útil',
    ],
  },
];

/**
 * Variações lexicais para evitar repetição absoluta
 * Usadas para fazer pequenas mudanças nos templates
 */
const LEXICAL_VARIATIONS = {
  'bem': ['ótimo', 'bom', 'legal', 'bacana'],
  'estou': ['fico', 'sou', 'estou'],
  'pergunta': ['dúvida', 'questão', 'tópico'],
  'resposta': ['reação', 'fala', 'comentário'],
  'ajudar': ['auxiliar', 'socorrer', 'apoiar'],
  'aprender': ['absorver', 'estudar', 'entender'],
};

/**
 * Gera um par de teacher forcing para uma posição específica da resposta.
 *
 * x = contextTokens + responseTokens[0..pos-1], padded à esquerda até maxSeqLen
 * y = responseTokens[pos]
 *
 * @param {number[]} contextTokens
 * @param {number[]} responseTokens
 * @param {number} pos - índice do token a predizer (1-based dentro da resposta)
 * @param {number} maxSeqLen
 * @returns {{ x: number[], y: number } | null}
 */
function buildTeacherForcingPair(contextTokens, responseTokens, pos, maxSeqLen) {
  if (pos < 1 || pos >= responseTokens.length) return null;

  // Sequência de entrada: contexto + prefixo da resposta até pos-1 (exclusive)
  const prefix = responseTokens.slice(0, pos);
  const rawSeq  = [...contextTokens, ...prefix];

  // Truncar pela esquerda para caber em maxSeqLen - 1 (reserva espaço para ao menos 1 token)
  const truncated = rawSeq.slice(-(maxSeqLen - 1));

  // Left-padding com PAD (token 0)
  const padded = [
    ...Array(Math.max(0, maxSeqLen - truncated.length)).fill(0),
    ...truncated,
  ];

  return { x: padded, y: responseTokens[pos] };
}

/**
 * Gera conversas sintéticas com teacher forcing posicional.
 *
 * Para cada par (contexto, resposta) de cada template, gera um par de treino
 * por posição da resposta: predizer responseTokens[pos] dado contexto +
 * responseTokens[0..pos-1]. Isso produz sinal real de LM ao invés de
 * apenas memorizar o último token.
 *
 * @param {Vocabulary} vocab - Instância do vocabulário
 * @param {number} targetCount - Número máximo de pares desejados
 * @param {number} maxSeqLen - Deve coincidir com HPARAMS.maxSeqLen (default 64)
 * @returns {Promise<Array>} Array de pares { x: number[], y: number }
 */
async function generateSyntheticConversations(vocab, targetCount = 200, maxSeqLen = 64) {
  const pairs = [];

  for (const template of CONVERSATION_TEMPLATES) {
    if (pairs.length >= targetCount) break;

    const contextTokens = vocab.tokenize(template.context);
    if (contextTokens.length === 0) continue;

    for (const response of template.responses) {
      if (pairs.length >= targetCount) break;

      const responseTokens = vocab.tokenize(response);
      // Resposta precisa de ao menos 2 tokens para gerar pelo menos 1 par posicional
      if (responseTokens.length < 2) continue;

      // Teacher forcing: um par por posição (predizer token[1] até token[N-1])
      for (let pos = 1; pos < responseTokens.length; pos++) {
        if (pairs.length >= targetCount) break;

        const pair = buildTeacherForcingPair(contextTokens, responseTokens, pos, maxSeqLen);
        if (pair) pairs.push(pair);
      }
    }
  }

  // Se ainda faltam pares, reutilizar templates com shuffle de resposta para
  // aumentar diversidade sem duplicar pares idênticos
  if (pairs.length < targetCount) {
    const seen = new Set(pairs.map(p => `${p.x.join(',')}_${p.y}`));

    outer:
    for (const template of CONVERSATION_TEMPLATES) {
      const contextTokens = vocab.tokenize(template.context);
      if (contextTokens.length === 0) continue;

      // Embaralhar respostas para variar a ordem
      const shuffled = [...template.responses].sort(() => Math.random() - 0.5);

      for (const response of shuffled) {
        const responseTokens = vocab.tokenize(response);
        if (responseTokens.length < 2) continue;

        for (let pos = 1; pos < responseTokens.length; pos++) {
          if (pairs.length >= targetCount) break outer;

          const pair = buildTeacherForcingPair(contextTokens, responseTokens, pos, maxSeqLen);
          if (!pair) continue;

          const key = `${pair.x.join(',')}_${pair.y}`;
          if (!seen.has(key)) {
            seen.add(key);
            pairs.push(pair);
          }
        }
      }
    }
  }

  console.log(`[SYNTHETIC] Geradas ${pairs.length} conversas sintéticas (teacher forcing posicional)`);
  return pairs;
}

module.exports = { generateSyntheticConversations };
