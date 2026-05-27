/**
 * akie_behavior.js — Comportamento Padrão do AKIE (SYSTEM DEFAULT)
 * 
 * Processa toda entrada do usuário ANTES do modelo.
 * Baseado na lógica de intenção do _aether.js (AETHER OS).
 * 
 * FLUXO:
 *   input → processBehavior(input, context?) → { mode, input, output?, context? }
 *     ↓
 *   se output definido → resposta direta (refino) — NÃO chama modelo
 *   se output vazio    → enviar input enriquecido para model.generate()
 * 
 * MODOS:
 *   SOCIAL  — conversa casual, saudação, pergunta simples
 *   ANALYZE — pedido de análise, explicação detalhada, raciocínio
 *   CODE    — pedido de código (usa regras de entrega completa)
 *   REFINE  — pedido de criação de prompt para outra IA
 */

// ---------------------------------------------------------------------------
// CONSTANTES DE CONFIGURAÇÃO
// ---------------------------------------------------------------------------

// Contexto base do AKIE — define persona e comportamento padrão
const AKIE_SYSTEM_CONTEXT = `
[CONTEXTO DO SISTEMA]
Você é o AKIE, um sistema de inteligência artificial em evolução contínua.
Você aprende com interações, constrói conhecimento semântico e melhora suas respostas.
Seu tom é amigável, direto e respeitoso.

[DIRETRIZES DE RESPOSTA]
- Seja claro e objetivo
- Se não souber, admita e ofereça aprender
- Para perguntas técnicas, explique de forma acessível
- Mantenha respostas entre 2 e 5 frases
- Use português natural, sem formalidade excessiva

[CAPACIDADES ATUAIS]
- Responder perguntas gerais
- Explicar conceitos
- Aprender com novas informações
- Processar linguagem natural em português
`.trim();

// Contexto para modo ANALYZE
const ANALYZE_EXTRA_CONTEXT = `
[MODO ANALYZE ATIVO]
Você está no modo de análise. O usuário quer uma resposta mais detalhada e estruturada.
- Entenda o problema antes de responder
- Quebre em partes lógicas se necessário
- Explique o raciocínio de forma clara
- Priorize clareza sobre complexidade
- Use estrutura: contexto rápido → análise direta → conclusão
`.trim();

// Contexto para modo CODE
const CODE_EXTRA_CONTEXT = `
[MODO CODE ATIVO]
Você está no modo de geração de código. Entregue código 100% funcional e completo.
- NUNCA use "// ...", "// restante aqui" ou qualquer omissão
- NUNCA pare no meio de um arquivo
- Todo bloco de código deve começar com \`\`\`linguagem e terminar com \`\`\`
- Se o código não couber, divida automaticamente em partes numeradas
`.trim();

// ---------------------------------------------------------------------------
// DETECÇÃO DE DOMÍNIO (extraído do _aether.js)
// ---------------------------------------------------------------------------

/**
 * Identifica o domínio da intenção do usuário.
 * Retorna: 'tecnologia' | 'gastronomia' | 'saude' | 'negocios' | 
 *          'juridico' | 'educacao' | 'comunicacao' | 'visual' | 'geral'
 */
function detectDomain(input) {
  const lower = input.toLowerCase();

  if (/\b(codigo|sistema|api|app|site|deploy|bug|erro|banco|servidor|react|javascript|python|sql|html|css|node|funcao|classe|variavel|componente|frontend|backend|fullstack)\b/.test(lower)) return 'tecnologia';
  if (/\b(imagem|foto|fotografia|ilustra|arte|wallpaper|banner|logo|poster|visual|desenho|pintura|crie.*imagem|gere.*imagem|gerar.*imagem|criar.*imagem)\b/i.test(lower)) return 'visual';
  if (/\b(receita|comida|bolo|prato|ingrediente|cozinhar|chef|restaurante)\b/.test(lower)) return 'gastronomia';
  if (/\b(saude|medico|dor|sintoma|remedio|consulta|terapia|psicolog|doenca)\b/.test(lower)) return 'saude';
  if (/\b(vender|comprar|preco|plano|contrato|negocio|empresa|cliente|venda|marketing)\b/.test(lower)) return 'negocios';
  if (/\b(lei|juridico|direito|advogado|processo|contrato|tribunal)\b/.test(lower)) return 'juridico';
  if (/\b(aprender|curso|aula|estudo|escola|professor|educacao|ensinar|explicar)\b/.test(lower)) return 'educacao';
  if (/\b(escrever|texto|artigo|conteudo|post|roteiro|historia|blog|redacao)\b/.test(lower)) return 'comunicacao';

  return 'geral';
}

// ---------------------------------------------------------------------------
// DETECÇÃO DE CONFIANÇA (extraído do _aether.js)
// ---------------------------------------------------------------------------

/**
 * Mede a completude da intenção do usuário.
 * Retorna: 'alta' | 'media' | 'baixa'
 */
function detectConfidence(input) {
  const words   = input.trim().split(/\s+/).length;
  const hasWhat = /\b(quero|preciso|crie|faca|como|o que|me|meu|minha|gostaria|pode)\b/i.test(input);
  const hasCtx  = words > 12;
  const hasGoal = words > 25;

  if (hasWhat && hasCtx && hasGoal) return 'alta';
  if (hasWhat && hasCtx) return 'media';
  return 'baixa';
}

// ---------------------------------------------------------------------------
// CLASSIFICAÇÃO DE MODO (baseado no classifyMode do _aether.js)
// ---------------------------------------------------------------------------

/**
 * Classifica o input do usuário em um modo de operação.
 * 
 * Modos:
 *   'refino'  — pedido de criação de prompt para outra IA
 *   'code'    — pedido de código/programação
 *   'analyze' — pedido de análise/estratégia
 *   'social'  — conversa casual (default)
 */
function classifyMode(input) {
  const lower = input.toLowerCase();

  // ── REFINO — criar prompt para IA ──────────────────────────────
  const refinoSignals = [
    /\b(gere um prompt|crie um prompt|prompt para|instrucao para ia|instrucao para o gpt|prompt master|prompt hd)\b/.test(lower),
    /\b(para a ia executar|para o claude|para o chatgpt|instrucao de sistema|cria.*prompt|criar.*prompt)\b/.test(lower),
  ];
  if (refinoSignals.some(Boolean)) return 'refino';

  // ── CODE — pedido de código ────────────────────────────────────
  const codeSignals = [
    /\b(codigo|programa|script|funcao|algoritmo|implementa|desenvolve.*codigo|cria.*funcao)\b/.test(lower),
    /\b(em python|em javascript|em html|em css|em react|em node|em java|em sql)\b/.test(lower),
    /\b(como.*codar|como.*programar|escreve.*codigo)\b/.test(lower),
  ];
  if (codeSignals.some(Boolean)) return 'code';

  // ── ANALYZE — análise/estratégia ────────────────────────────────
  const analyzeSignals = [
    /\b(analise|analisar|análise|estrategia|estratégia|melhor forma|qual a melhor|comparar|comparacao|vale a pena|decisao|planejamento)\b/.test(lower),
    /\b(como escalar|como crescer|como melhorar|como otimizar|como estruturar)\b/.test(lower),
  ];

  const words      = input.trim().split(/\s+/).length;
  const isLong     = words > 20;
  const confidence = detectConfidence(input);
  const domain     = detectDomain(input);

  if (
    analyzeSignals.some(Boolean) ||
    (confidence === 'alta' && (isLong || domain === 'negocios'))
  ) {
    return 'analyze';
  }

  // ── SOCIAL (default) ────────────────────────────────────────────
  return 'social';
}

// ---------------------------------------------------------------------------
// PROCESSAMENTO PRINCIPAL DE COMPORTAMENTO
// ---------------------------------------------------------------------------

/**
 * Função principal — processa o input do usuário e decide o comportamento.
 * 
 * @param {string} input - Texto de entrada do usuário
 * @param {object} options - Opções opcionais
 * @param {string} options.context - Contexto adicional do usuário
 * @param {string} options.language - Idioma ('pt', 'en', 'es')
 * @returns {object} { mode, input, output?, context? }
 */
function processBehavior(input, options = {}) {
  if (!input || typeof input !== 'string' || input.trim().length === 0) {
    return {
      mode: 'social',
      input: '',
      output: null,
      context: null,
    };
  }

  const trimmed = input.trim();

  // 1. Classificar modo
  const mode = classifyMode(trimmed);

  // 2. Detectar domínio
  const domain = detectDomain(trimmed);

  // 3. Detectar confiança
  const confidence = detectConfidence(trimmed);

  // 4. Construir resposta baseada no modo
  switch (mode) {
    case 'refino':
      return handleRefineMode(trimmed, options);

    case 'code':
      return handleCodeMode(trimmed, options);

    case 'analyze':
      return handleAnalyzeMode(trimmed, options);

    case 'social':
    default:
      return handleSocialMode(trimmed, options);
  }
}

// ---------------------------------------------------------------------------
// HANDLERS POR MODO
// ---------------------------------------------------------------------------

/**
 * Modo REFINE — delega ao sistema de refino existente.
 * NÃO usa o modelo AKIE. Retorna resposta diretamente.
 */
function handleRefineMode(input, options) {
  // Verifica se há sistema de refino disponível
  if (options.refineryAvailable && typeof options.refineryCallback === 'function') {
    // Delega ao sistema de refino externo (ex: buildRefinerySystem do _aether.js)
    const refinedPrompt = options.refineryCallback(input, options.language || 'pt');
    return {
      mode: 'refine',
      input: input,
      output: refinedPrompt,
      context: null,
    };
  }

  // Se não há refino disponível, tenta fazer um refino simples
  const domain = detectDomain(input);
  const refinedInput = buildSimpleRefinedPrompt(input, domain);

  return {
    mode: 'refine',
    input: refinedInput,
    output: null, // sem output = vai para o modelo com input enriquecido
    context: null,
  };
}

/**
 * Modo CODE — prepara input com contexto de geração de código.
 */
function handleCodeMode(input, options) {
  return {
    mode: 'code',
    input: input,
    output: null,
    context: CODE_EXTRA_CONTEXT,
  };
}

/**
 * Modo ANALYZE — prepara input com contexto analítico.
 */
function handleAnalyzeMode(input, options) {
  const confidence = detectConfidence(input);

  // Se confiança baixa, faz pergunta de clarificação
  if (confidence === 'baixa') {
    const clarification = generateClarificationQuestion(input);
    return {
      mode: 'analyze',
      input: input,
      output: clarification, // resposta direta — NÃO chama modelo
      context: null,
    };
  }

  return {
    mode: 'analyze',
    input: input,
    output: null,
    context: ANALYZE_EXTRA_CONTEXT,
  };
}

/**
 * Modo SOCIAL — conversa casual.
 */
function handleSocialMode(input, options) {
  // Detecta se é saudação pura
  if (isPureGreeting(input)) {
    return {
      mode: 'social',
      input: input,
      output: null, // deixa o modelo responder naturalmente
      context: null,
    };
  }

  // Se for pergunta muito vaga, gera clarificação
  const confidence = detectConfidence(input);
  if (confidence === 'baixa' && input.split(/\s+/).length > 3) {
    const clarification = generateClarificationQuestion(input);
    return {
      mode: 'social',
      input: input,
      output: clarification,
      context: null,
    };
  }

  return {
    mode: 'social',
    input: input,
    output: null,
    context: AKIE_SYSTEM_CONTEXT,
  };
}

// ---------------------------------------------------------------------------
// HELPERS
// ---------------------------------------------------------------------------

/**
 * Verifica se o input é apenas uma saudação.
 */
function isPureGreeting(input) {
  const lower = input.toLowerCase().trim();
  const greetings = [
    'oi', 'olá', 'ola', 'hey', 'ei', 'e aí', 'e ai', 'oiê', 'oie',
    'bom dia', 'boa tarde', 'boa noite',
    'hello', 'hi', 'hey there',
    'tudo bem', 'tudo bom', 'como vai', 'como está', 'como esta',
    'td bem', 'tb',
  ];
  return greetings.some(g => lower === g || lower.startsWith(g) && lower.length <= g.length + 3);
}

/**
 * Gera pergunta de clarificação para inputs vagos.
 * Máximo 12 palavras (seguindo regra do _aether.js).
 */
function generateClarificationQuestion(input) {
  const domain = detectDomain(input);
  const questions = {
    tecnologia: 'Qual tecnologia ou linguagem você está usando?',
    gastronomia: 'Que tipo de prato ou culinária você tem em mente?',
    saude: 'Pode descrever melhor o sintoma ou situação?',
    negocios: 'Qual o contexto do seu negócio ou mercado?',
    juridico: 'Pode especificar a área jurídica ou situação?',
    educacao: 'Qual o tema ou nível de aprendizado?',
    comunicacao: 'Qual o formato e público-alvo do conteúdo?',
    visual: 'Que tipo de imagem ou estilo visual você imagina?',
    geral: 'Pode dar mais detalhes sobre o que precisa?',
  };
  return questions[domain] || questions.geral;
}

/**
 * Constrói prompt refinado simples (fallback quando não há Refinaria externa).
 */
function buildSimpleRefinedPrompt(input, domain) {
  const personaMap = {
    tecnologia: 'Engenheiro de Software Sênior',
    gastronomia: 'Chef de Cozinha Profissional',
    saude: 'Especialista em Saúde',
    negocios: 'Consultor Estratégico',
    juridico: 'Consultor Jurídico',
    educacao: 'Professor Especialista',
    comunicacao: 'Copywriter Profissional',
    visual: null, // visual não usa persona
    geral: 'Especialista',
  };

  const persona = personaMap[domain] || personaMap.geral;

  // Para domínio visual, formato especial sem persona
  if (domain === 'visual') {
    return `${input}\n\nDescreva a cena visual diretamente, sem "Atue como". Inclua sujeito, estilo, iluminação, cores e qualidade.`;
  }

  return `Atue como ${persona}. Sua tarefa é: ${input}\n\nSeja direto e claro. Entregue exatamente o que foi solicitado.`;
}

/**
 * Obtém o contexto padrão do AKIE.
 */
function getDefaultContext() {
  return AKIE_SYSTEM_CONTEXT;
}

// ---------------------------------------------------------------------------
// EXPORT
// ---------------------------------------------------------------------------

module.exports = {
  // Função principal
  processBehavior,

  // Funções de detecção (expostas para uso externo)
  classifyMode,
  detectDomain,
  detectConfidence,
  isPureGreeting,

  // Contextos padrão
  getDefaultContext,
  AKIE_SYSTEM_CONTEXT,
  ANALYZE_EXTRA_CONTEXT,
  CODE_EXTRA_CONTEXT,

  // Constantes
  MODES: ['social', 'analyze', 'code', 'refine'],
};