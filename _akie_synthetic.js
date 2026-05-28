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
 * Gera conversas sintéticas aplicando pequenas variações
 *
 * @param {Vocabulary} vocab - Instância do vocabulário
 * @param {number} targetCount - Número de pares desejados
 * @returns {Promise<Array>} Array de pares {x: int[], y: int}
 */
async function generateSyntheticConversations(vocab, targetCount = 200) {
  const pairs = [];

  // Primeiro: usar templates diretos
  for (const template of CONVERSATION_TEMPLATES) {
    for (const response of template.responses) {
      if (pairs.length >= targetCount) break;

      const contextTokens = vocab.tokenize(template.context);
      const responseTokens = vocab.tokenize(response);

      if (contextTokens.length > 0 && responseTokens.length > 0) {
        // Pegar último token da resposta como label
        const lastToken = responseTokens[responseTokens.length - 1];

        // Montar sequência: contexto preenchido até maxSeqLen - 1
        const maxSeqLen = 64; // Deve coincidir com HPARAMS.maxSeqLen
        const inputSeq = contextTokens.slice(-Math.max(0, maxSeqLen - 1));

        // Padding
        const padded = [
          ...Array(Math.max(0, maxSeqLen - inputSeq.length)).fill(0), // PAD token
          ...inputSeq,
        ];

        pairs.push({
          x: padded,
          y: lastToken,
        });
      }
    }

    if (pairs.length >= targetCount) break;
  }

  // Segundo: gerar variações dos templates (multiplicar dados)
  if (pairs.length < targetCount) {
    const remainingNeeded = targetCount - pairs.length;
    let added = 0;

    for (const template of CONVERSATION_TEMPLATES) {
      if (added >= remainingNeeded) break;

      for (let i = 0; i < Math.min(2, template.responses.length); i++) {
        if (added >= remainingNeeded) break;

        // Aplicar pequena variação no contexto (se houver)
        let contextVar = template.context;
        const words = contextVar.split(/\s+/);
        if (words.length > 2 && Math.random() < 0.3) {
          // Chance de fazer pequena mudança
          const randIdx = Math.floor(Math.random() * (words.length - 1));
          // Apenas exemplo simples: não fazemos variação elaborada aqui
          // Em produção, poderia usar um dicionário de sinônimos
        }

        const responseIdx = i % template.responses.length;
        const response = template.responses[responseIdx];

        const contextTokens = vocab.tokenize(contextVar);
        const responseTokens = vocab.tokenize(response);

        if (contextTokens.length > 0 && responseTokens.length > 0) {
          const lastToken = responseTokens[responseTokens.length - 1];
          const maxSeqLen = 64;
          const inputSeq = contextTokens.slice(-Math.max(0, maxSeqLen - 1));

          const padded = [
            ...Array(Math.max(0, maxSeqLen - inputSeq.length)).fill(0),
            ...inputSeq,
          ];

          // Verificar se este par é diferente dos já adicionados (evitar duplicação)
          const isDuplicate = pairs.some(p =>
            JSON.stringify(p.x) === JSON.stringify(padded) && p.y === lastToken
          );

          if (!isDuplicate) {
            pairs.push({ x: padded, y: lastToken });
            added++;
          }
        }
      }
    }
  }

  console.log(`[SYNTHETIC] Geradas ${pairs.length} conversas sintéticas`);
  return pairs;
}

module.exports = { generateSyntheticConversations };
