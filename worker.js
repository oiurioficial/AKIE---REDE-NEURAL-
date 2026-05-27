/**
 * worker.js  —  AKIE Training Worker + HTTP Endpoint
 *
 * Roda 24h no Railway. Nunca para.
 * Expõe endpoint HTTP para o NEXUS chamar geração em tempo real.
 *
 *  ┌─────────────────────────────────────────────────────────────┐
 *  │  MODO 1 — INTERACTIVE  (usuário interagiu)                  │
 *  │  MODO 2 — CONSOLIDATION (replay do grafo semântico)         │
 *  │  MODO 3 — EXPANSION    (web crawl, lacunas do grafo)        │
 *  │  MODO 4 — SELF_PLAY    (modelo gera e treina nos melhores)  │
 *  └─────────────────────────────────────────────────────────────┘
 *
 *  CORREÇÕES v2:
 *    - Coleção: nexus_nodes → nexus_graph (alinhado com o motor)
 *    - Campo:   confirmed   → confidence === 'confirmed'
 *    - Adicionado: servidor HTTP na porta 3000
 *    - Adicionado: endpoint POST /generate para o NEXUS
 *    - Adicionado: endpoint GET  /status para monitoramento
 */

require('dotenv').config();
const admin  = require('firebase-admin');
const http   = require('http');
const path   = require('path');
const fs     = require('fs');

const { AKIEModel }              = require('./_nexus_neural');
const { Vocabulary, tokenizeText } = require('./_akie_vocab');
// ---------------------------------------------------------------------------
// Bootstrap de conhecimento — inline (sem dependência de arquivo externo)
// ---------------------------------------------------------------------------

const BOOTSTRAP_EPISODES = [
  { input: 'olá', output: 'olá como posso ajudar você hoje' },
  { input: 'oi', output: 'oi tudo bem com você' },
  { input: 'bom dia', output: 'bom dia espero que seu dia seja ótimo' },
  { input: 'boa tarde', output: 'boa tarde como está você' },
  { input: 'boa noite', output: 'boa noite posso ajudar com algo' },
  { input: 'tudo bem', output: 'tudo bem sim e com você' },
  { input: 'como vai você', output: 'estou bem pronto para ajudar' },
  { input: 'obrigado', output: 'de nada fico feliz em ajudar' },
  { input: 'obrigada', output: 'de nada sempre que precisar' },
  { input: 'até logo', output: 'até logo foi um prazer conversar' },
  { input: 'tchau', output: 'tchau volte quando precisar' },
  { input: 'preciso de ajuda', output: 'estou aqui para ajudar pode falar' },
  { input: 'quem é você', output: 'sou o akie um sistema de inteligência em aprendizado contínuo' },
  { input: 'o que você é', output: 'sou uma inteligência artificial que aprende com interações' },
  { input: 'o que você faz', output: 'aprendo com conversas e melhoro minhas respostas ao longo do tempo' },
  { input: 'você consegue aprender', output: 'sim aprendo continuamente com cada interação' },
  { input: 'você entende português', output: 'sim minha língua principal é o português' },
  { input: 'como você funciona', output: 'processo linguagem extraio padrões e construo conhecimento gradualmente' },
  { input: 'quanto é dois mais dois', output: 'dois mais dois é igual a quatro' },
  { input: 'quanto é cinco mais três', output: 'cinco mais três é igual a oito' },
  { input: 'quanto é dez menos quatro', output: 'dez menos quatro é igual a seis' },
  { input: 'quanto é três vezes quatro', output: 'três vezes quatro é igual a doze' },
  { input: 'quantos dias tem uma semana', output: 'uma semana tem sete dias' },
  { input: 'quantos meses tem um ano', output: 'um ano tem doze meses' },
  { input: 'quantas horas tem um dia', output: 'um dia tem vinte e quatro horas' },
  { input: 'qual é a capital do brasil', output: 'a capital do brasil é brasília' },
  { input: 'qual é o maior oceano', output: 'o oceano pacífico é o maior do mundo' },
  { input: 'quantos continentes existem', output: 'existem sete continentes no mundo' },
  { input: 'o sol é uma estrela', output: 'sim o sol é uma estrela de médio porte' },
  { input: 'o que é gravidade', output: 'gravidade é a força que atrai objetos em direção a massas maiores' },
  { input: 'o que é oxigênio', output: 'oxigênio é um elemento químico essencial para a respiração dos seres vivos' },
  { input: 'o que é água', output: 'água é uma molécula formada por dois átomos de hidrogênio e um de oxigênio' },
  { input: 'o que é fotossíntese', output: 'fotossíntese é o processo pelo qual plantas convertem luz solar em energia' },
  { input: 'o que é inteligência artificial', output: 'inteligência artificial é a capacidade de máquinas realizarem tarefas que exigem inteligência humana' },
  { input: 'o que é machine learning', output: 'machine learning é um método onde sistemas aprendem com dados sem serem explicitamente programados' },
  { input: 'o que é uma rede neural', output: 'uma rede neural é um sistema computacional inspirado no funcionamento do cérebro humano' },
  { input: 'o que é um algoritmo', output: 'um algoritmo é uma sequência de passos definidos para resolver um problema' },
  { input: 'o que é programação', output: 'programação é a arte de criar instruções para computadores executarem tarefas' },
  { input: 'o que é internet', output: 'internet é uma rede global que conecta bilhões de computadores e dispositivos' },
  { input: 'o que é banco de dados', output: 'banco de dados é um sistema organizado para armazenar e recuperar informações' },
  { input: 'o que é api', output: 'api é uma interface que permite que diferentes sistemas se comuniquem entre si' },
  { input: 'o que é variável', output: 'variável é um espaço na memória que armazena um valor que pode mudar' },
  { input: 'o que é função', output: 'função é um bloco de código reutilizável que executa uma tarefa específica' },
  { input: 'o que é recursão', output: 'recursão é quando uma função chama a si mesma para resolver um problema menor' },
  { input: 'o que é uma palavra', output: 'uma palavra é a menor unidade com significado independente na linguagem' },
  { input: 'o que é uma frase', output: 'uma frase é um conjunto de palavras que expressa um pensamento completo' },
  { input: 'o que é sinônimo', output: 'sinônimo é uma palavra com significado igual ou muito próximo de outra' },
  { input: 'o que é antônimo', output: 'antônimo é uma palavra com significado oposto a outra' },
  { input: 'o que é gramática', output: 'gramática é o conjunto de regras que organiza o uso correto da língua' },
  { input: 'como resolver um problema', output: 'primeiro entenda o problema depois analise alternativas e escolha a melhor solução' },
  { input: 'o que é hipótese', output: 'hipótese é uma suposição provisória que pode ser testada e verificada' },
  { input: 'o que é conclusão', output: 'conclusão é o resultado lógico obtido após análise de informações' },
  { input: 'o que é evidência', output: 'evidência é uma informação que apoia ou refuta uma afirmação' },
  { input: 'o que é análise', output: 'análise é o processo de examinar algo em detalhes para compreendê-lo melhor' },
  { input: 'o que é aprendizado', output: 'aprendizado é o processo de adquirir novos conhecimentos ou habilidades' },
  { input: 'o que é memória', output: 'memória é a capacidade de armazenar e recuperar informações e experiências' },
  { input: 'o que é conhecimento', output: 'conhecimento é o conjunto de informações compreendidas e assimiladas' },
  { input: 'o que é inteligência', output: 'inteligência é a capacidade de aprender adaptar-se e resolver problemas' },
  { input: 'o que é criatividade', output: 'criatividade é a capacidade de gerar ideias novas e originais' },
  { input: 'o que é experiência', output: 'experiência é o conhecimento adquirido através da prática e vivência' },
  { input: 'o que é erro', output: 'erro é um desvio do resultado esperado que oferece oportunidade de aprendizado' },
  { input: 'o que é clima', output: 'clima é o padrão de condições atmosféricas de uma região ao longo do tempo' },
  { input: 'o que é temperatura', output: 'temperatura é a medida de calor ou frio de um corpo ou ambiente' },
  { input: 'o que é floresta', output: 'floresta é uma área densamente coberta por árvores e vegetação variada' },
  { input: 'o que é ecosistema', output: 'ecossistema é o conjunto de seres vivos e o ambiente em que interagem' },
  { input: 'o que é saúde', output: 'saúde é o estado de completo bem-estar físico mental e social' },
  { input: 'por que dormir é importante', output: 'o sono restaura o corpo consolida memórias e regula funções vitais' },
  { input: 'por que beber água é importante', output: 'a água regula funções do organismo hidrata células e elimina toxinas' },
  { input: 'o que é nutrição', output: 'nutrição é o processo pelo qual o organismo obtém e usa nutrientes dos alimentos' },
  { input: 'o que é metabolismo', output: 'metabolismo é o conjunto de reações químicas que mantêm o organismo vivo' },
  { input: 'o que é tempo', output: 'tempo é a dimensão que mede a duração e sequência dos eventos' },
  { input: 'o que é espaço', output: 'espaço é o ambiente tridimensional onde todos os objetos existem e se movem' },
  { input: 'o que é velocidade', output: 'velocidade é a medida de quanto espaço um objeto percorre por unidade de tempo' },
  { input: 'o que é felicidade', output: 'felicidade é um estado emocional positivo de satisfação e bem-estar' },
  { input: 'o que é empatia', output: 'empatia é a capacidade de compreender e compartilhar os sentimentos de outra pessoa' },
  { input: 'o que é motivação', output: 'motivação é a força interna que impulsiona uma pessoa a agir em direção a objetivos' },
  { input: 'o que é padrão', output: 'padrão é uma regularidade ou sequência que se repete de forma reconhecível' },
  { input: 'o que é estrutura', output: 'estrutura é a forma como as partes de algo estão organizadas e relacionadas' },
  { input: 'o que é sistema', output: 'sistema é um conjunto de elementos interconectados que funcionam como um todo' },
  { input: 'o que é processo', output: 'processo é uma série de ações ou etapas para alcançar um resultado' },
  { input: 'o que é relação', output: 'relação é uma conexão ou vínculo entre dois ou mais elementos' },
  { input: 'o que é contexto', output: 'contexto é o conjunto de circunstâncias que envolvem e influenciam algo' },
  { input: 'o que é significado', output: 'significado é o conteúdo ou sentido que uma palavra símbolo ou evento transmite' },
  { input: 'o que é abstração', output: 'abstração é o processo de identificar características essenciais ignorando detalhes irrelevantes' },
  { input: 'o que é representação', output: 'representação é a forma como algo é simbolizado ou descrito em outro meio' },
  { input: 'você está funcionando', output: 'sim estou em operação e processando sua mensagem' },
  { input: 'você está aprendendo agora', output: 'sim cada interação contribui para meu aprendizado contínuo' },
  { input: 'você vai melhorar', output: 'sim o objetivo é melhorar continuamente com cada experiência' },
  { input: 'posso ensinar você algo', output: 'sim suas interações são fundamentais para meu aprendizado' },
  { input: 'quais são as estações do ano', output: 'as estações do ano são primavera verão outono e inverno' },
  { input: 'quais são os dias da semana', output: 'os dias da semana são segunda terça quarta quinta sexta sábado e domingo' },
  { input: 'quais são as vogais', output: 'as vogais do português são a e i o u' },
  { input: 'quais são os sentidos humanos', output: 'os sentidos humanos são visão audição olfato paladar e tato' },
  { input: 'quais são as operações matemáticas básicas', output: 'as operações básicas são adição subtração multiplicação e divisão' },
  { input: 'quais são os planetas do sistema solar', output: 'mercúrio vênus terra marte júpiter saturno urano e netuno' },
  { input: 'por que o céu é azul', output: 'o céu é azul porque a atmosfera dispersa a luz azul com mais intensidade' },
  { input: 'por que as plantas são verdes', output: 'as plantas são verdes por causa da clorofila que absorve outras cores da luz' },
  { input: 'por que a maçã cai', output: 'a maçã cai porque a gravidade atrai todos os objetos em direção ao centro da terra' },
  { input: 'por que precisamos respirar', output: 'respiramos para obter oxigênio que as células precisam para produzir energia' },
  { input: 'por que dormimos', output: 'dormimos para que o corpo e o cérebro se recuperem e se reorganizem' },
  { input: 'qual a diferença entre dado e informação', output: 'dado é o fato bruto informação é o dado processado e contextualizado' },
  { input: 'qual a diferença entre conhecimento e sabedoria', output: 'conhecimento é saber os fatos sabedoria é saber aplicá-los bem' },
  { input: 'qual a diferença entre sintaxe e semântica', output: 'sintaxe é a estrutura da linguagem semântica é o significado' },
  { input: 'qual a diferença entre hardware e software', output: 'hardware é a parte física do computador software é o conjunto de programas' },
  { input: 'qual a diferença entre causa e efeito', output: 'causa é o que origina algo efeito é o resultado produzido pela causa' },
  { input: 'o que é realidade', output: 'realidade é o conjunto de tudo que existe independentemente de ser percebido' },
  { input: 'o que é verdade', output: 'verdade é a correspondência entre uma afirmação e os fatos reais' },
  { input: 'o que é consciência', output: 'consciência é a capacidade de estar ciente de si mesmo e do ambiente' },
  { input: 'o que é pensamento', output: 'pensamento é o processo mental de processar informações e gerar ideias' },
  { input: 'o que é linguagem', output: 'linguagem é um sistema de símbolos usados para comunicar pensamentos e ideias' },
  { input: 'o que é cultura', output: 'cultura é o conjunto de valores práticas e criações compartilhados por um grupo' },
  { input: 'o que é ética', output: 'ética é o conjunto de princípios que orientam o comportamento correto' },
  { input: 'o que é liberdade', output: 'liberdade é a capacidade de agir segundo a própria vontade dentro de limites éticos' },
  { input: 'como funciona o aprendizado de máquina', output: 'o sistema recebe dados identifica padrões e ajusta seus parâmetros para melhorar' },
  { input: 'o que é treinamento de modelo', output: 'treinamento é o processo de ajustar os parâmetros de um modelo com base em dados' },
  { input: 'o que é vocabulário em nlp', output: 'vocabulário é o conjunto de tokens que o modelo reconhece e processa' },
  { input: 'o que é tokenização', output: 'tokenização é o processo de dividir texto em unidades menores chamadas tokens' },
  { input: 'o que é embedding', output: 'embedding é uma representação numérica densa de tokens em um espaço vetorial' },
  { input: 'o que é temperatura em ia', output: 'temperatura controla a aleatoriedade na geração de texto pelo modelo' },
  { input: 'o que é inferência', output: 'inferência é o processo de usar um modelo treinado para fazer previsões' },
  { input: 'o que é loss', output: 'loss é a medida de quão longe as previsões do modelo estão dos valores corretos' },
  { input: 'não entendi', output: 'me desculpe posso tentar explicar de outra forma' },
  { input: 'pode explicar melhor', output: 'sim vou tentar ser mais claro na explicação' },
  { input: 'não sei', output: 'tudo bem não saber isso é o ponto de partida para aprender' },
  { input: 'isso é verdade', output: 'pode ser preciso verificar as informações antes de confirmar' },
  { input: 'você tem certeza', output: 'faço meu melhor para ser preciso mas posso estar errado' },
  { input: 'interessante', output: 'sim é um assunto com muitas dimensões a explorar' },
];

const BOOTSTRAP_GRAPH_NODES = [
  { id: 'aprendizado', label: 'aprendizado', confidence: 'confirmed', relations: [
    { type: 'gera', target: 'conhecimento', weight: 0.9 },
    { type: 'requer', target: 'pratica', weight: 0.8 },
    { type: 'usa', target: 'memória', weight: 0.85 },
  ], contexts: ['educação', 'inteligência artificial'], verbs: ['aprender', 'adquirir', 'desenvolver'] },
  { id: 'conhecimento', label: 'conhecimento', confidence: 'confirmed', relations: [
    { type: 'gera', target: 'compreensão', weight: 0.9 },
    { type: 'requer', target: 'aprendizado', weight: 0.8 },
    { type: 'permite', target: 'decisão', weight: 0.85 },
  ], contexts: ['educação', 'ciência'], verbs: ['conhecer', 'saber', 'compreender'] },
  { id: 'linguagem', label: 'linguagem', confidence: 'confirmed', relations: [
    { type: 'permite', target: 'comunicação', weight: 0.95 },
    { type: 'usa', target: 'palavras', weight: 0.9 },
    { type: 'expressa', target: 'pensamento', weight: 0.85 },
  ], contexts: ['comunicação', 'cognição'], verbs: ['falar', 'escrever', 'comunicar'] },
  { id: 'inteligência', label: 'inteligência', confidence: 'confirmed', relations: [
    { type: 'permite', target: 'aprendizado', weight: 0.9 },
    { type: 'gera', target: 'solução', weight: 0.85 },
    { type: 'usa', target: 'raciocínio', weight: 0.9 },
  ], contexts: ['cognição', 'tecnologia'], verbs: ['raciocinar', 'adaptar', 'resolver'] },
  { id: 'sistema', label: 'sistema', confidence: 'confirmed', relations: [
    { type: 'compoe', target: 'componentes', weight: 0.9 },
    { type: 'gera', target: 'resultado', weight: 0.85 },
    { type: 'requer', target: 'estrutura', weight: 0.8 },
  ], contexts: ['tecnologia', 'organização'], verbs: ['processar', 'integrar', 'operar'] },
  { id: 'dado', label: 'dado', confidence: 'confirmed', relations: [
    { type: 'gera', target: 'informação', weight: 0.9 },
    { type: 'requer', target: 'processamento', weight: 0.8 },
    { type: 'permite', target: 'análise', weight: 0.85 },
  ], contexts: ['computação', 'ciência'], verbs: ['processar', 'analisar', 'armazenar'] },
  { id: 'padrão', label: 'padrão', confidence: 'confirmed', relations: [
    { type: 'emerge_de', target: 'dados', weight: 0.85 },
    { type: 'permite', target: 'previsão', weight: 0.8 },
    { type: 'gera', target: 'conhecimento', weight: 0.75 },
  ], contexts: ['análise', 'aprendizado'], verbs: ['identificar', 'reconhecer', 'extrair'] },
  { id: 'rede_neural', label: 'rede neural', confidence: 'confirmed', relations: [
    { type: 'usa', target: 'dados', weight: 0.9 },
    { type: 'gera', target: 'previsão', weight: 0.85 },
    { type: 'requer', target: 'treinamento', weight: 0.9 },
  ], contexts: ['inteligência artificial', 'machine learning'], verbs: ['aprender', 'prever', 'processar'] },
  { id: 'vocabulário', label: 'vocabulário', confidence: 'confirmed', relations: [
    { type: 'compoe', target: 'linguagem', weight: 0.9 },
    { type: 'permite', target: 'comunicação', weight: 0.85 },
    { type: 'gera', target: 'expressão', weight: 0.8 },
  ], contexts: ['linguagem', 'nlp'], verbs: ['expandir', 'aprender', 'usar'] },
  { id: 'memória', label: 'memória', confidence: 'confirmed', relations: [
    { type: 'permite', target: 'aprendizado', weight: 0.85 },
    { type: 'gera', target: 'reconhecimento', weight: 0.8 },
  ], contexts: ['cognição', 'computação'], verbs: ['lembrar', 'armazenar', 'recuperar'] },
];

const COMMON_WORDS_PT = [
  'o','a','os','as','um','uma','uns','umas',
  'eu','tu','ele','ela','nós','eles','elas',
  'meu','minha','seu','sua','nosso','nossa',
  'este','esta','esse','essa','aquele','aquela','isto','isso','aquilo',
  'que','quem','qual','quais','de','do','da','dos','das',
  'em','no','na','nos','nas','por','pelo','pela','pelos','pelas',
  'para','com','sem','sobre','sob','entre','até','desde','após','antes',
  'e','ou','mas','porém','pois','porque','se','quando','onde','como',
  'embora','ainda','já','também','ser','estar','ter','haver','fazer',
  'ir','vir','dar','ver','saber','poder','querer','dever','precisar',
  'falar','dizer','pedir','responder','perguntar','pensar','sentir',
  'conhecer','entender','aprender','criar','construir','desenvolver',
  'usar','aplicar','começar','terminar','continuar','parar','mudar',
  'ajudar','mostrar','explicar','definir','identificar',
  'grande','pequeno','novo','velho','bom','mau','ruim',
  'primeiro','último','único','geral','específico','importante',
  'necessário','possível','difícil','fácil','rápido','lento',
  'forte','fraco','alto','baixo','real','virtual','digital',
  'físico','mental','social','humano','natural','artificial',
  'correto','incorreto','completo','incompleto','simples','complexo',
  'pessoa','lugar','tempo','modo','parte','tipo','forma','vida',
  'mundo','terra','país','cidade','casa','trabalho','estudo',
  'problema','solução','início','fim','resultado','objetivo',
  'palavra','frase','texto','número','valor','medida','quantidade',
  'máquina','computador','programa','ferramenta','linha','tabela',
  'modelo','exemplo','regra','princípio','teoria','prática',
  'causa','efeito','razão','motivo','não','sim','muito','pouco',
  'mais','menos','bem','mal','sempre','nunca','aqui','ali','lá',
  'agora','antes','depois','logo','então','assim','também','ainda',
  'apenas','certamente','provavelmente','realmente',
];

async function runBootstrap(db, vocab) {
  // Verificar sentinela — idempotente
  try {
    const sentinel = await db.collection('akie_worker_status').doc('bootstrap').get();
    if (sentinel.exists && sentinel.data().completed === true) {
      console.log('[BOOTSTRAP] Já executado anteriormente. Pulando.');
      return false;
    }
  } catch(e) { /* continua mesmo sem sentinela */ }

  const [episodesSnap, graphSnap] = await Promise.all([
    db.collection('nexus_episodes').limit(1).get(),
    db.collection('nexus_graph').limit(1).get(),
  ]);

  const episodesEmpty = episodesSnap.empty;
  const graphEmpty    = graphSnap.empty;

  if (!episodesEmpty && !graphEmpty) {
    console.log('[BOOTSTRAP] Dados existentes. Marcando completo.');
    await db.collection('akie_worker_status').doc('bootstrap').set({
      completed: true, completed_at: new Date().toISOString(),
      inserted_episodes: 0, inserted_nodes: 0,
    });
    return false;
  }

  console.log('[BOOTSTRAP] iniciado');

  let insertedEpisodes = 0;
  let insertedNodes    = 0;

  if (episodesEmpty) {
    const CHUNK = 400;
    for (let i = 0; i < BOOTSTRAP_EPISODES.length; i += CHUNK) {
      const batch = db.batch();
      const chunk = BOOTSTRAP_EPISODES.slice(i, i + CHUNK);
      for (const ep of chunk) {
        batch.set(db.collection('nexus_episodes').doc(), {
          input: ep.input, output: ep.output,
          layer: 'bootstrap', feedback: 'positive',
          processed: false, created_at: new Date().toISOString(),
        });
      }
      await batch.commit();
      insertedEpisodes += chunk.length;
    }
    console.log('[BOOTSTRAP] dataset inserido: ' + insertedEpisodes + ' exemplos');
  }

  if (graphEmpty) {
    const batch = db.batch();
    for (const node of BOOTSTRAP_GRAPH_NODES) {
      batch.set(db.collection('nexus_graph').doc(node.id), {
        ...node, created_at: new Date().toISOString(), usage_count: 0,
      });
    }
    await batch.commit();
    insertedNodes = BOOTSTRAP_GRAPH_NODES.length;
    console.log('[BOOTSTRAP] grafo inicial: ' + insertedNodes + ' nós inseridos');
  }

  vocab.addTokens(COMMON_WORDS_PT);
  for (const ep of BOOTSTRAP_EPISODES) {
    vocab.addTokens(ep.input.split(' ').filter(t => t.length > 0));
    vocab.addTokens(ep.output.split(' ').filter(t => t.length > 0));
  }
  for (const node of BOOTSTRAP_GRAPH_NODES) {
    vocab.addTokens(node.label.split(' '));
    vocab.addTokens(node.verbs || []);
    vocab.addTokens(node.contexts || []);
    for (const rel of (node.relations || [])) {
      vocab.addTokens(rel.target.replace(/_/g, ' ').split(' '));
    }
  }

  console.log('[BOOTSTRAP] vocabulário inicial: ' + vocab.size + ' tokens');

  await db.collection('akie_worker_status').doc('bootstrap').set({
    completed: true, completed_at: new Date().toISOString(),
    inserted_episodes: insertedEpisodes, inserted_nodes: insertedNodes,
    vocab_size: vocab.size,
  });

  return true;
}
const {
  fetchUserEpisodes,
  episodesToPairs,
  fetchGraphSentences,
  graphSentencesToPairs,
  fetchWebPatterns,
  selfPlayPairs,
} = require('./_akie_dataset');

// ---------------------------------------------------------------------------
// Configuração
// ---------------------------------------------------------------------------

const CONFIG = {
  intervalMs:          60_000,
  modelDir:            process.env.MODEL_DIR || '/data/akie_model',
  httpPort:            parseInt(process.env.PORT || '3000', 10),
  saveEveryN:          5,
  consolidationAfter:  1,
  expansionAfter:      3,
  selfPlayAfter:       2,
};

const MODE = {
  INTERACTIVE:   'INTERACTIVE',
  CONSOLIDATION: 'CONSOLIDATION',
  EXPANSION:     'EXPANSION',
  SELF_PLAY:     'SELF_PLAY',
};

// ---------------------------------------------------------------------------
// Estado global
// ---------------------------------------------------------------------------

const state = {
  cycle:             0,
  idleCycles:        0,
  lastSaveCycle:     0,
  totalPairsTrained: 0,
  model:             null,
  vocab:             null,
  db:                null,
  modeHistory:       [],
  metrics: {
    lastLoss:        null,
    lastAccuracy:    null,
    trainCycles:     0,
  },
};

// ---------------------------------------------------------------------------
// Servidor HTTP — endpoint para o NEXUS chamar geração
// ---------------------------------------------------------------------------

function startHttpServer() {
  const server = http.createServer(async (req, res) => {
    const headers = {
      'Content-Type':                'application/json',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    };

    if (req.method === 'OPTIONS') {
      res.writeHead(204, headers);
      res.end();
      return;
    }

    // ── GET /status ────────────────────────────────────────────
    if (req.method === 'GET' && req.url === '/status') {
      const stats = state.model ? state.model.getStats() : { ready: false };
      res.writeHead(200, headers);
      res.end(JSON.stringify({
        ok:          true,
        cycle:       state.cycle,
        idle_cycles: state.idleCycles,
        metrics:     state.metrics,
        model:       stats,
        mode_history: state.modeHistory.slice(-5),
      }));
      return;
    }

    // ── POST /generate ─────────────────────────────────────────
    // Chamado pelo NEXUS quando quer geração neural
    // Body: { prompt: string, max_tokens?: number, temperature?: number }
    if (req.method === 'POST' && req.url === '/generate') {
      let body = '';
      req.on('data', chunk => { body += chunk; });
      req.on('end', () => {
        try {
          const { prompt, max_tokens, temperature } = JSON.parse(body || '{}');

          if (!prompt) {
            res.writeHead(400, headers);
            res.end(JSON.stringify({ error: 'Campo prompt ausente.' }));
            return;
          }

          if (!state.model || !state.model.ready) {
            res.writeHead(503, headers);
            res.end(JSON.stringify({ error: 'Modelo não está pronto ainda.' }));
            return;
          }

          const generated = state.model.generate(
            prompt,
            max_tokens  || 30,
            temperature || 0.7
          );

          // Marcar episódio para treino futuro (fire-and-forget)
          saveGenerationEpisode(prompt, generated).catch(() => {});

          res.writeHead(200, headers);
          res.end(JSON.stringify({
            ok:        true,
            prompt:    prompt,
            generated: generated || '',
            steps:     state.model.trainSteps,
          }));

        } catch (err) {
          res.writeHead(500, headers);
          res.end(JSON.stringify({ error: err.message }));
        }
      });
      return;
    }

    // ── POST /feedback ─────────────────────────────────────────
    // O NEXUS envia feedback sobre uma geração (positivo/negativo)
    // Body: { episode_id: string, feedback: 'positive' | 'negative' }
    if (req.method === 'POST' && req.url === '/feedback') {
      let body = '';
      req.on('data', chunk => { body += chunk; });
      req.on('end', async () => {
        try {
          const { episode_id, feedback } = JSON.parse(body || '{}');
          if (episode_id && feedback && state.db) {
            await state.db.collection('nexus_episodes')
              .doc(episode_id)
              .update({ feedback, feedback_at: new Date().toISOString() });
          }
          res.writeHead(200, headers);
          res.end(JSON.stringify({ ok: true }));
        } catch (err) {
          res.writeHead(500, headers);
          res.end(JSON.stringify({ error: err.message }));
        }
      });
      return;
    }

    res.writeHead(404, headers);
    res.end(JSON.stringify({ error: 'Rota não encontrada.' }));
  });

  server.listen(CONFIG.httpPort, () => {
    console.log(`[HTTP] Servidor rodando na porta ${CONFIG.httpPort}`);
    console.log(`[HTTP] Endpoints: GET /status | POST /generate | POST /feedback`);
  });

  return server;
}

// Salva geração como episódio para treino futuro
async function saveGenerationEpisode(prompt, generated) {
  if (!state.db || !generated) return;
  await state.db.collection('nexus_episodes').add({
    input:        prompt,
    output:       generated,
    layer:        'akie_generation',
    feedback:     null,
    processed:    false,
    created_at:   new Date().toISOString(),
  });
}

// ---------------------------------------------------------------------------
// Inicialização
// ---------------------------------------------------------------------------

async function init() {
  console.log('════════════════════════════════════════');
  console.log('  AKIE Training Worker v2 — iniciando');
  console.log(`  ${new Date().toISOString()}`);
  console.log('════════════════════════════════════════');

  // Firebase
  const raw = process.env.FIREBASE_SERVICE_ACCOUNT;
  if (!raw) throw new Error('FIREBASE_SERVICE_ACCOUNT não definido.');
  const serviceAccount = JSON.parse(raw);

  if (!admin.apps.length) {
    admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
  }
  state.db = admin.firestore();
  console.log('[INIT] Firebase conectado. Coleção ativa: nexus_graph');

  // Vocabulário
  state.vocab = await loadOrCreateVocab();

  // Bootstrap — garante que o sistema nunca inicie com grafo vazio
  // Idempotente: só roda uma vez na vida do Firestore
  const bootstrapped = await runBootstrap(state.db, state.vocab);
  if (bootstrapped) {
    // Salvar vocab expandido pelo bootstrap antes de construir o modelo
    await saveVocab(state.vocab);
  }

  console.log(`[INIT] Vocabulário: ${state.vocab.size} tokens`);

  // Modelo
  state.model = new AKIEModel(state.vocab);
  const loaded = await state.model.load(CONFIG.modelDir);

  if (!loaded) {
    console.log('[INIT] Nenhum modelo encontrado. Construindo novo...');
    await primeVocabulary(state.vocab, state.db);
    // Re-executar bootstrap de vocab caso primeVocabulary tenha adicionado tokens
    state.model = new AKIEModel(state.vocab);
    state.model.build();
    await state.model.save(CONFIG.modelDir);
  } else {
    // Modelo carregado — verificar se vocab cresceu desde o último save
    await maybeRebuildForVocabGrowth();
  }

  await state.db.collection('akie_worker_status').doc('current').set({
    started_at:  new Date().toISOString(),
    status:      'running',
    model_stats: state.model.getStats(),
  });

  console.log('[INIT] Worker pronto.\n');
}

/**
 * Alimenta vocabulário com tokens do grafo NEXUS antes do primeiro build.
 * CORRIGIDO: usa nexus_graph (não nexus_nodes).
 */
async function primeVocabulary(vocab, db) {
  console.log('[INIT] Carregando vocabulário base do grafo nexus_graph...');
  const snap = await db.collection('nexus_graph').limit(500).get();
  const sentences = [];

  snap.forEach(doc => {
    const d = doc.data();
    if (d.label) sentences.push(d.label);
    if (d.id)    sentences.push(d.id.replace(/_/g, ' '));
    (d.relations || []).forEach(r => {
      if (r.target) sentences.push(r.target.replace(/_/g, ' '));
      if (r.type)   sentences.push(r.type.replace(/_/g, ' '));
    });
    (d.contexts || []).forEach(c => sentences.push(c));
    (d.verbs    || []).forEach(v => sentences.push(v));
  });

  for (const s of sentences) {
    vocab.addTokens(tokenizeText(s));
  }
  await saveVocab(vocab);
  console.log(`[INIT] Vocabulário primário: ${vocab.size} tokens`);
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

async function scheduler() {
  state.cycle++;
  const tag = `[C${String(state.cycle).padStart(4, '0')}]`;

  try {
    // Prioridade 1: episódios reais de usuário
    const episodes = await fetchUserEpisodes(state.db, 100);
    if (episodes.length > 0) {
      state.idleCycles = 0;
      await runMode(MODE.INTERACTIVE, { episodes, tag });
      return;
    }

    state.idleCycles++;
    const n = state.idleCycles;

    if (n % CONFIG.selfPlayAfter === 0 && state.model.trainSteps > 50) {
      await runMode(MODE.SELF_PLAY, { tag });
    } else if (n % CONFIG.expansionAfter === 0) {
      await runMode(MODE.EXPANSION, { tag });
    } else {
      await runMode(MODE.CONSOLIDATION, { tag });
    }

  } catch (err) {
    console.error(`${tag} ERRO:`, err.message);
  }
}

// ---------------------------------------------------------------------------
// Executor de modo
// ---------------------------------------------------------------------------

async function runMode(mode, ctx = {}) {
  const { tag = '', episodes = [] } = ctx;
  const t0 = Date.now();
  let pairs = [];
  let desc  = '';

  console.log(`${tag} Modo: ${mode}`);

  switch (mode) {
    case MODE.INTERACTIVE: {
      pairs = episodesToPairs(episodes, state.vocab);
      desc  = `${episodes.length} episódios → ${pairs.length} pares`;
      await maybeExpandVocab(episodes);
      break;
    }
    case MODE.CONSOLIDATION: {
      const sentences = await fetchGraphSentences(state.db, 300);
      if (!sentences.length) {
        console.log(`${tag} Grafo vazio — pulando consolidação`);
        return;
      }
      pairs = graphSentencesToPairs(sentences, state.vocab);
      desc  = `${sentences.length} frases do grafo → ${pairs.length} pares`;
      break;
    }
    case MODE.EXPANSION: {
      const vocabBefore = state.vocab.size;
      pairs = await fetchWebPatterns(state.db, state.vocab, 3);
      desc  = `web crawl → ${pairs.length} pares`;
      // Sempre verificar crescimento de vocab, mesmo sem pairs
      // O fetchWebPatterns adiciona tokens ao vocab independentemente de gerar pares
      if (state.vocab.size > vocabBefore) {
        await saveVocab(state.vocab);
        await maybeRebuildForVocabGrowth();
      }
      break;
    }
    case MODE.SELF_PLAY: {
      pairs = await selfPlayPairs(state.model, state.db, state.vocab, 15);
      desc  = `self-play → ${pairs.length} pares`;
      break;
    }
  }

  if (pairs.length > 0) {
    shuffleArray(pairs);
    const result = await state.model.trainBatch(pairs, 3);
    state.metrics.lastLoss     = result.loss;
    state.metrics.lastAccuracy = result.accuracy;
    state.metrics.trainCycles++;
    state.totalPairsTrained += pairs.length;

    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    const loss    = result.loss    != null ? result.loss.toFixed(4) : 'n/a';
    const acc     = result.accuracy != null ? (result.accuracy * 100).toFixed(1) + '%' : 'n/a';
    console.log(`${tag} ✓ ${desc} | loss=${loss} acc=${acc} | ${elapsed}s`);

    if (state.cycle - state.lastSaveCycle >= CONFIG.saveEveryN) {
      await state.model.save(CONFIG.modelDir);
      await saveVocab(state.vocab);
      state.lastSaveCycle = state.cycle;
      await reportStatus(mode);
    }
  } else {
    console.log(`${tag} Nenhum dado disponível para ${mode}`);
  }

  state.modeHistory.push(mode);
  if (state.modeHistory.length > 20) state.modeHistory.shift();
}

// ---------------------------------------------------------------------------
// Vocabulário e persistência
// ---------------------------------------------------------------------------

async function maybeExpandVocab(episodes) {
  const before = state.vocab.size;
  for (const ep of episodes) {
    if (ep.input)  state.vocab.addTokens(tokenizeText(ep.input));
    if (ep.output) state.vocab.addTokens(tokenizeText(ep.output));
  }
  if (state.vocab.size > before) {
    console.log(`[VOCAB] Cresceu: ${before} → ${state.vocab.size}`);
    await maybeRebuildForVocabGrowth();
  }
}

async function maybeRebuildForVocabGrowth() {
  const curr        = state.vocab.size;
  const embSize     = state.model.getStats().embeddingVocabSize; // tamanho REAL da camada
  if (curr > embSize) {
    console.log(`[VOCAB] Expandindo camada de embedding: ${embSize} → ${curr}`);
    await state.model.expandVocabulary(curr);
    state.model.vocab = state.vocab;
  }
}

async function loadOrCreateVocab() {
  const p = path.join(CONFIG.modelDir, 'akie_vocab.json');
  try {
    if (fs.existsSync(p)) return Vocabulary.fromJSON(JSON.parse(fs.readFileSync(p, 'utf8')));
  } catch (e) { /* ignora */ }

  try {
    const doc = await state.db.collection('akie_worker_status').doc('vocabulary').get();
    if (doc.exists) return Vocabulary.fromJSON(doc.data());
  } catch (e) { /* ignora */ }

  return new Vocabulary();
}

async function saveVocab(vocab) {
  try {
    fs.mkdirSync(CONFIG.modelDir, { recursive: true });
    fs.writeFileSync(path.join(CONFIG.modelDir, 'akie_vocab.json'), JSON.stringify(vocab.toJSON(), null, 2));
  } catch (e) {
    try {
      await state.db.collection('akie_worker_status').doc('vocabulary').set(vocab.toJSON());
    } catch (e2) {
      console.error('[VOCAB] Falha ao salvar:', e2.message);
    }
  }
}

async function reportStatus(mode) {
  try {
    await state.db.collection('akie_worker_status').doc('current').set({
      updated_at:  new Date().toISOString(),
      status:      'running',
      cycle:       state.cycle,
      idle_cycles: state.idleCycles,
      last_mode:   mode,
      mode_history: state.modeHistory,
      total_pairs: state.totalPairsTrained,
      metrics:     state.metrics,
      model_stats: state.model.getStats(),
    }, { merge: true });
  } catch (e) { /* não crítico */ }
}

// ---------------------------------------------------------------------------
// Utilitários
// ---------------------------------------------------------------------------

function shuffleArray(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

// ---------------------------------------------------------------------------
// Watchdog
// ---------------------------------------------------------------------------

let lastHeartbeat = Date.now();

function startWatchdog() {
  setInterval(() => {
    if (Date.now() - lastHeartbeat > 5 * 60 * 1000) {
      console.error('[WATCHDOG] Loop travado. Forçando ciclo...');
      scheduler().catch(console.error);
      lastHeartbeat = Date.now();
    }
  }, 30_000);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

async function main() {
  await init();
  startHttpServer();
  startWatchdog();

  await scheduler();
  lastHeartbeat = Date.now();

  setInterval(async () => {
    await scheduler();
    lastHeartbeat = Date.now();
  }, CONFIG.intervalMs);

  process.on('SIGTERM', async () => {
    console.log('\n[WORKER] SIGTERM — salvando...');
    try {
      await state.model.save(CONFIG.modelDir);
      await saveVocab(state.vocab);
      await state.db.collection('akie_worker_status').doc('current').update({
        status: 'stopped', stopped_at: new Date().toISOString(),
      });
    } catch (e) { /* ignora */ }
    process.exit(0);
  });
}

main().catch(err => {
  console.error('[WORKER] Erro fatal:', err);
  process.exit(1);
});
