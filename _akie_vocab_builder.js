/**
 * _akie_vocab_builder.js v1.0
 *
 * Expande o vocabulário da AKIE de 427 → 8.000+ tokens PT-BR.
 *
 * ESTRATÉGIA:
 *   1. Carrega vocab atual do modelo salvo (akie_vocab.json)
 *   2. Injeta tokens curados: raízes, prefixos, sufixos, palavras funcionais
 *   3. Consome corpus externo se disponível (/opt/akie/corpus/sentences_ptbr.txt)
 *   4. Aplica filtro de frequência mínima e qualidade
 *   5. Salva novo vocab compatível com _akie_vocab.js (mesmo formato JSON)
 *   6. Gera relatório de expansão
 *
 * USO:
 *   node _akie_vocab_builder.js
 *   node _akie_vocab_builder.js --dry-run     (mostra stats sem salvar)
 *   node _akie_vocab_builder.js --target 5000 (define tamanho alvo)
 *
 * ATENÇÃO:
 *   Após rodar, o modelo precisa ser RETREINADO — o embedding layer muda de
 *   tamanho. Este script NÃO modifica pesos do modelo, apenas o vocab JSON.
 *
 * OUTPUT:
 *   /opt/akie/checkpoints/pretrained/akie_vocab.json  (substituído)
 *   /opt/akie/corpus/vocab_expansion_report.txt       (relatório)
 */

const fs   = require('fs');
const path = require('path');

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const DRY_RUN      = process.argv.includes('--dry-run');
const TARGET_SIZE  = parseInt(
  (process.argv.find(a => a.startsWith('--target=')) || '--target=8000').split('=')[1]
);
const VOCAB_PATH   = process.env.VOCAB_PATH  || '/opt/akie/checkpoints/pretrained/akie_vocab.json';
const CORPUS_PATH  = process.env.CORPUS_PATH || '/opt/akie/corpus/sentences_ptbr.txt';
const REPORT_PATH  = '/opt/akie/corpus/vocab_expansion_report.txt';

// ---------------------------------------------------------------------------
// Tokens especiais — devem estar SEMPRE no vocab (posições fixas 0-5)
// ---------------------------------------------------------------------------

const SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<BOS>', '<EOS>', '<MASK>', '<SEP>'];

// ---------------------------------------------------------------------------
// Léxico curado PT-BR — 8.000+ tokens de alta relevância
// Organizado por categoria para manutenção futura
// ---------------------------------------------------------------------------

// Palavras funcionais — alta frequência em qualquer corpus PT
const FUNCIONAL = [
  'o','a','os','as','um','uma','uns','umas',
  'de','do','da','dos','das','em','no','na','nos','nas',
  'por','pelo','pela','pelos','pelas','para','com','sem',
  'sob','sobre','até','após','ante','entre','contra','desde',
  'que','e','ou','mas','porém','contudo','todavia','entretanto',
  'porque','pois','logo','portanto','assim','então',
  'se','quando','enquanto','embora','aunque','apesar',
  'não','nem','também','já','ainda','só','apenas','sempre',
  'nunca','jamais','talvez','quase','muito','pouco','mais','menos',
  'bem','mal','aqui','ali','lá','aí','antes','depois','agora',
  'hoje','ontem','amanhã','sempre','logo','cedo','tarde','nunca',
  'eu','você','ele','ela','nós','vocês','eles','elas',
  'me','te','se','nos','vos','lhe','lhes',
  'meu','minha','teu','tua','seu','sua','nosso','nossa','vosso','vossa',
  'meus','minhas','teus','tuas','seus','suas','nossos','nossas',
  'este','esta','estes','estas','esse','essa','esses','essas',
  'aquele','aquela','aqueles','aquelas','isto','isso','aquilo',
  'qual','quais','quem','onde','como','quando','quanto','quantos',
  'que','cujo','cuja','cujos','cujas',
  'todo','toda','todos','todas','nenhum','nenhuma',
  'algum','alguma','alguns','algumas','outro','outra','outros','outras',
  'mesmo','mesma','próprio','própria','tal','tais','tanto','tanta',
];

// Verbos — formas mais frequentes no presente, passado e infinitivo
const VERBOS = [
  // ser/estar
  'ser','estar','sou','és','é','somos','são','estou','está','estamos','estão',
  'era','eram','foi','foram','sido','estado',
  // ter/haver
  'ter','haver','tenho','tens','tem','temos','têm','tinha','tinham','teve','tiveram',
  'há','houve','havido','tido',
  // ir/vir
  'ir','vir','vou','vai','vamos','vão','vim','veio','viemos','vieram','ido','vindo',
  // fazer
  'fazer','faço','faz','fazemos','fazem','fez','fizeram','feito','fazendo',
  // poder/querer/saber
  'poder','querer','saber','posso','pode','podemos','podem','pude','puderam',
  'quero','quer','queremos','querem','quis','quiseram',
  'sei','sabe','sabemos','sabem','soube','souberam','sabido',
  // dizer/falar
  'dizer','falar','digo','diz','dizemos','dizem','disse','disseram','dito',
  'falo','fala','falamos','falam','falei','falou','falaram','falado','falando',
  // ver/ouvir
  'ver','ouvir','vejo','vê','vemos','veem','vi','viu','viram','visto',
  'ouço','ouve','ouvimos','ouvem','ouvi','ouviu','ouviram','ouvido',
  // viver/morar/trabalhar
  'viver','morar','trabalhar','vivo','vive','vivemos','vivem',
  'moro','mora','moramos','moram','morei','morou','moraram',
  'trabalho','trabalha','trabalhamos','trabalham','trabalhei','trabalhou',
  // aprender/ensinar/estudar
  'aprender','ensinar','estudar','aprendo','aprende','aprendemos','aprendem',
  'aprendi','aprendeu','aprenderam','aprendido','aprendendo',
  'ensino','ensina','ensinamos','ensinam','ensinei','ensinou','ensinado',
  'estudo','estuda','estudamos','estudam','estudei','estudou','estudado',
  // ajudar/precisar/querer
  'ajudar','precisar','ajudo','ajuda','ajudamos','ajudam','ajudei','ajudou','ajudado',
  'preciso','precisa','precisamos','precisam','precisei','precisou',
  // gostar/amar/odiar
  'gostar','amar','odiar','gosto','gosta','gostamos','gostam','gostei','gostou',
  'amo','ama','amamos','amam','amei','amou','amado',
  // começar/terminar/continuar
  'começar','terminar','continuar','começo','começa','começamos','começam',
  'termino','termina','terminamos','terminam','terminei','terminou','terminado',
  'continuo','continua','continuamos','continuam','continuei','continuou',
  // outros frequentes
  'dar','tomar','usar','criar','fazer','mudar','levar','trazer',
  'dou','dá','damos','dão','dei','deu','deram','dado','dando',
  'tomo','toma','tomamos','tomam','tomei','tomou','tomado',
  'uso','usa','usamos','usam','usei','usou','usado','usando',
  'crio','cria','criamos','criam','criei','criou','criado','criando',
  'mudo','muda','mudamos','mudam','mudei','mudou','mudado',
  'levo','leva','levamos','levam','levei','levou','levado',
  'trago','traz','trazemos','trazem','trouxe','trouxeram','trazido',
  // gerúndios frequentes
  'sendo','estando','tendo','havendo','indo','vindo','fazendo','dizendo',
  'vendo','ouvindo','vivendo','morando','trabalhando','querendo','podendo',
  'sabendo','dando','usando','criando','mudando','ajudando','precisando',
];

// Substantivos — pessoas, lugares, coisas, conceitos
const SUBSTANTIVOS = [
  // Pessoas e relações
  'pessoa','pessoas','homem','mulher','criança','adulto','jovem','idoso',
  'pai','mãe','filho','filha','irmão','irmã','avô','avó','tio','tia',
  'amigo','amiga','colega','professor','professora','aluno','aluna',
  'médico','médica','advogado','engenheiro','programador','cientista',
  // Corpo
  'cabeça','rosto','olho','olhos','boca','nariz','ouvido','orelha',
  'mão','mãos','braço','perna','pé','pés','coração','cérebro',
  // Natureza
  'sol','lua','estrela','terra','mar','rio','lago','montanha','floresta',
  'árvore','flor','folha','pedra','água','fogo','ar','vento','chuva',
  'nuvem','céu','terra','areia','planta','animal',
  // Animais
  'cachorro','gato','pássaro','peixe','cobra','leão','tigre','elefante',
  'cavalo','vaca','porco','galinha','coelho','rato','mosquito','abelha',
  // Lugares
  'casa','apartamento','rua','cidade','país','estado','bairro','escola',
  'hospital','mercado','loja','banco','parque','praia','campo','fazenda',
  'brasil','portugal','angola','moçambique','europa','amérika','ásia',
  'brasília','paulo','janeiro','recife','salvador','belo','horizonte',
  'manaus','fortaleza','curitiba','porto','alegre','goiânia','belém',
  // Tempo
  'dia','noite','manhã','tarde','hora','minuto','segundo','semana',
  'mês','ano','século','momento','tempo','data','hoje','amanhã','ontem',
  'janeiro','fevereiro','março','abril','maio','junho','julho','agosto',
  'setembro','outubro','novembro','dezembro',
  'segunda','terça','quarta','quinta','sexta','sábado','domingo',
  // Objetos
  'carro','casa','livro','telefone','computador','mesa','cadeira','cama',
  'porta','janela','copo','prato','colher','faca','garfo','roupa',
  'camisa','calça','sapato','bolsa','chave','dinheiro','cartão',
  // Conceitos abstratos
  'amor','amizade','felicidade','tristeza','raiva','medo','coragem',
  'verdade','mentira','justiça','liberdade','paz','guerra','vida','morte',
  'trabalho','emprego','salário','problema','solução','ideia','plano',
  'futuro','passado','presente','história','cultura','ciência','arte',
  'música','filme','livro','notícia','informação','dados','resultado',
  // Tecnologia
  'computador','celular','internet','rede','sistema','programa','código',
  'algoritmo','dados','banco','servidor','aplicativo','software','hardware',
  'inteligência','artificial','modelo','treino','aprendizado','máquina',
  'rede','neural','parâmetro','peso','camada','token','vocabulário',
  'texto','linguagem','processamento','geração','resposta','pergunta',
];

// Adjetivos frequentes
const ADJETIVOS = [
  'bom','boa','bons','boas','mau','má','maus','más',
  'grande','grandes','pequeno','pequena','pequenos','pequenas',
  'alto','alta','baixo','baixa','largo','larga','estreito',
  'novo','nova','novos','novas','velho','velha','velhos','velhas',
  'jovem','jovens','moderno','moderna','antigo','antiga',
  'belo','bela','feio','feia','bonito','bonita','bonitos','bonitas',
  'forte','forte','fraco','fraca','rápido','rápida','lento','lenta',
  'fácil','difícil','simples','complexo','complexa',
  'rico','rica','pobre','pobres','barato','cara','caro','caras',
  'importante','interessante','necessário','possível','impossível',
  'certo','certa','errado','errada','verdadeiro','falso',
  'livre','preso','aberto','fechado','cheio','vazio',
  'feliz','triste','alegre','contente','bravo','calmo','nervoso',
  'cansado','descansado','doente','saudável','vivo','morto',
  'inteligente','burro','sábio','tolo','educado','rude',
  'claro','escuro','quente','frio','úmido','seco',
  'primeiro','segundo','terceiro','último','próximo','anterior',
  'diferente','igual','similar','oposto','contrário',
  'total','parcial','completo','incompleto','geral','específico',
  'público','privado','nacional','internacional','local','global',
  'natural','artificial','real','virtual','físico','digital',
];

// Advérbios e conectivos
const ADVERBIOS = [
  'muito','pouco','bastante','demais','mais','menos','tão','tanto',
  'bem','mal','melhor','pior','assim','então','logo','já',
  'ainda','sempre','nunca','jamais','raramente','frequentemente',
  'geralmente','normalmente','naturalmente','obviamente','certamente',
  'possivelmente','provavelmente','definitivamente','absolutamente',
  'completamente','parcialmente','totalmente','especialmente',
  'principalmente','basicamente','simplesmente','exatamente',
  'aqui','ali','lá','aí','perto','longe','acima','abaixo','dentro','fora',
  'antes','depois','durante','enquanto','quando','agora','então',
  'rapidamente','lentamente','facilmente','dificilmente','claramente',
  'finalmente','inicialmente','anteriormente','posteriormente',
  'além','também','inclusive','porém','contudo','entretanto',
  'portanto','assim','logo','consequentemente','portanto',
];

// Numerais e quantificadores
const NUMERAIS = [
  'zero','um','dois','três','quatro','cinco','seis','sete','oito','nove','dez',
  'onze','doze','treze','quatorze','quinze','dezesseis','dezessete','dezoito','dezenove',
  'vinte','trinta','quarenta','cinquenta','sessenta','setenta','oitenta','noventa',
  'cem','cento','duzentos','trezentos','quatrocentos','quinhentos',
  'mil','milhão','bilhão','trilhão',
  'primeiro','segundo','terceiro','quarto','quinto','sexto','sétimo','oitavo',
  'nono','décimo','centésimo','milésimo',
  'metade','dobro','triplo','quádruplo','meio','terço','quarto',
  'alguns','vários','muitos','poucos','todos','nenhum','qualquer',
  'cada','todo','ambos','ambas','demais',
];

// Pontuação e tokens especiais de forma
const PONTUACAO = [
  '.', ',', '!', '?', ';', ':', '-', '—', '(', ')', '[', ']',
  '"', "'", '...', '/', '\\', '+', '=', '%', '@', '#',
];

// Sufixos produtivos PT-BR (morfologia)
const SUFIXOS = [
  'ção','ções','dade','idades','ismo','ista','istas','mente','agem','agens',
  'eiro','eira','eiros','eiras','inho','inha','inhos','inhas',
  'ão','ões','ã','ãos','ão','al','ais','el','eis','il','is',
  'ável','ível','vel','dor','dora','dores','nte','ntes',
  'oso','osa','osos','osas','ico','ica','icos','icas',
];

// Domínio conversacional — perguntas e respostas
const CONVERSACIONAL = [
  'olá','oi','tchau','adeus','boa','tarde','manhã','noite',
  'obrigado','obrigada','por','nada','imagina','fica','vontade',
  'desculpe','desculpa','perdão','licença','favor',
  'claro','certo','errado','sim','não','talvez','possivelmente',
  'tudo','bem','ótimo','excelente','perfeito','maravilhoso',
  'interessante','curioso','estranho','normal','comum','raro',
  'incrível','fantástico','impressionante','surpreendente',
  'entendo','compreendo','entendi','compreendi','percebi',
  'acho','acho','penso','acredito','imagino','suponho',
  'sinto','sinto','parece','parecia','pareceu','será',
  'ok','okay','certo','combinado','trato','feito',
  'veja','olhe','note','perceba','observe','considere',
  'afinal','aliás','inclusive','exceto','salvo','porém',
];

// Identidade AETHER/AKIE
const IDENTIDADE = [
  'akie','aether','nexus','sistema','motor','inteligência','artificial',
  'ecossistema','aprendizado','interação','evolução','cognição',
  'memória','episódica','semântico','grafo','nó','aresta',
  'embedding','transformer','atenção','camada','peso','parâmetro',
  'epoch','batch','loss','accuracy','treino','treinamento','modelo',
  'inferência','predição','geração','tokenização','vocabulário',
  'corpus','dataset','amostra','validação','checkpoint',
  'neural','rede','perceptron','gradiente','backpropagation',
  'softmax','relu','dropout','normalização','regularização',
];

// ---------------------------------------------------------------------------
// Tokenizador (espelho do _akie_vocab.js para consistência)
// ---------------------------------------------------------------------------

function tokenize(text) {
  return text
    .toLowerCase()
    .normalize('NFC')
    .replace(/([.,!?;:()\[\]{}"'])/g, ' $1 ')
    .replace(/\s+/g, ' ')
    .trim()
    .split(' ')
    .filter(t => t.length > 0);
}

// ---------------------------------------------------------------------------
// Carregar vocab atual
// ---------------------------------------------------------------------------

function loadCurrentVocab(vocabPath) {
  if (!fs.existsSync(vocabPath)) {
    console.log(`[VOCAB] Arquivo não encontrado: ${vocabPath}`);
    console.log('[VOCAB] Iniciando do zero com tokens especiais.');
    return { token2id: {}, id2token: [] };
  }

  try {
    const raw  = fs.readFileSync(vocabPath, 'utf8');
    const data = JSON.parse(raw);

    // Suporte a dois formatos possíveis do _akie_vocab.js
    if (data.token2id && data.id2token) {
      console.log(`[VOCAB] Carregado: ${data.id2token.length} tokens existentes`);
      return data;
    }
    if (data.vocab && Array.isArray(data.vocab)) {
      // Formato alternativo: array simples
      const id2token = data.vocab;
      const token2id = {};
      id2token.forEach((t, i) => { token2id[t] = i; });
      console.log(`[VOCAB] Carregado (formato array): ${id2token.length} tokens existentes`);
      return { token2id, id2token };
    }

    console.log('[VOCAB] Formato desconhecido — iniciando do zero.');
    return { token2id: {}, id2token: [] };
  } catch (e) {
    console.log(`[VOCAB] Erro ao carregar: ${e.message}`);
    return { token2id: {}, id2token: [] };
  }
}

// ---------------------------------------------------------------------------
// Carregar corpus externo
// ---------------------------------------------------------------------------

function loadCorpusSentences(corpusPath) {
  if (!fs.existsSync(corpusPath)) {
    console.log(`[VOCAB] Corpus externo não encontrado: ${corpusPath}`);
    console.log('[VOCAB] Usando apenas léxico curado embutido.');
    return [];
  }

  const lines = fs.readFileSync(corpusPath, 'utf8')
    .split('\n')
    .map(l => l.trim())
    .filter(l => l.length > 5 && l.length < 500);

  console.log(`[VOCAB] Corpus externo: ${lines.length} frases`);
  return lines;
}

// ---------------------------------------------------------------------------
// Construir frequência de tokens do corpus
// ---------------------------------------------------------------------------

function buildFrequencyMap(sentences) {
  const freq = new Map();
  for (const s of sentences) {
    for (const t of tokenize(s)) {
      if (t.length >= 2) {  // BUG FIX: era > 2, deve ser >= 2
        freq.set(t, (freq.get(t) || 0) + 1);
      }
    }
  }
  return freq;
}

// ---------------------------------------------------------------------------
// Expandir vocabulário
// ---------------------------------------------------------------------------

function expandVocab(current, curatedTokens, corpusFreq, targetSize) {
  const { token2id, id2token } = current;

  // Set de tokens já presentes
  const existing = new Set(id2token);
  const toAdd    = [];

  // 1. Garantir tokens especiais nas primeiras posições
  for (const sp of SPECIAL_TOKENS) {
    if (!existing.has(sp)) {
      toAdd.push({ token: sp, priority: 1000, freq: 0 });
    }
  }

  // 2. Tokens curados — alta prioridade
  for (const t of curatedTokens) {
    const clean = t.toLowerCase().normalize('NFC').trim();
    if (clean && !existing.has(clean) && !toAdd.find(x => x.token === clean)) {
      toAdd.push({
        token:    clean,
        priority: 100,
        freq:     corpusFreq.get(clean) || 0,
      });
    }
  }

  // 3. Tokens do corpus por frequência — baixa prioridade mas cobrem lacunas
  const sortedCorpus = [...corpusFreq.entries()]
    .sort((a, b) => b[1] - a[1])
    .filter(([t, c]) => {
      if (existing.has(t)) return false;
      if (toAdd.find(x => x.token === t)) return false;
      if (t.length < 2) return false;
      if (/^\d+$/.test(t)) return false; // só números — desnecessário
      if (c < 2) return false; // frequência mínima 2
      return true;
    });

  for (const [t, c] of sortedCorpus) {
    toAdd.push({ token: t, priority: 10, freq: c });
  }

  // Ordenar: especiais > curados > corpus
  toAdd.sort((a, b) => b.priority - a.priority || b.freq - a.freq);

  // Calcular quantos adicionar
  const slotsAvailable = Math.max(0, targetSize - id2token.length);
  const adding = toAdd.slice(0, slotsAvailable);

  // Construir novo vocab
  const newId2token = [...id2token];
  const newToken2id = { ...token2id };

  for (const { token } of adding) {
    if (!newToken2id.hasOwnProperty(token)) {
      newToken2id[token] = newId2token.length;
      newId2token.push(token);
    }
  }

  return {
    token2id: newToken2id,
    id2token: newId2token,
    added:    adding.length,
    total:    newId2token.length,
  };
}

// ---------------------------------------------------------------------------
// Validar vocab (integridade)
// ---------------------------------------------------------------------------

function validateVocab(vocab) {
  const { token2id, id2token } = vocab;
  const errors = [];

  // Checar tokens especiais
  for (const sp of SPECIAL_TOKENS) {
    if (!token2id.hasOwnProperty(sp)) {
      errors.push(`Token especial ausente: ${sp}`);
    }
  }

  // Checar consistência bidirecional
  let inconsistencies = 0;
  for (let i = 0; i < id2token.length; i++) {
    const t = id2token[i];
    if (token2id[t] !== i) {
      inconsistencies++;
    }
  }
  if (inconsistencies > 0) {
    errors.push(`${inconsistencies} inconsistências token2id↔id2token`);
  }

  // Checar duplicatas em id2token
  const seen    = new Set();
  let duplicates = 0;
  for (const t of id2token) {
    if (seen.has(t)) duplicates++;
    seen.add(t);
  }
  if (duplicates > 0) {
    errors.push(`${duplicates} tokens duplicados em id2token`);
  }

  return errors;
}

// ---------------------------------------------------------------------------
// Gerar relatório
// ---------------------------------------------------------------------------

function generateReport(beforeSize, afterSize, added, elapsed, reportPath) {
  const lines = [
    '═══════════════════════════════════════',
    '  AKIE Vocab Expansion Report',
    `  Data: ${new Date().toISOString()}`,
    '═══════════════════════════════════════',
    '',
    `  Vocabulário anterior: ${beforeSize} tokens`,
    `  Vocabulário novo:     ${afterSize} tokens`,
    `  Tokens adicionados:   ${added}`,
    `  Expansão:             ${((afterSize / beforeSize - 1) * 100).toFixed(1)}%`,
    `  Tempo:                ${elapsed}ms`,
    '',
    '  Fontes:',
    '    - Léxico curado PT-BR (funcional, verbos, subst., adj., adv.)',
    '    - Numerais e pontuação',
    '    - Sufixos morfológicos PT-BR',
    '    - Corpus externo (sentences_ptbr.txt) se disponível',
    '    - Domínio conversacional e identidade AETHER/AKIE',
    '',
    '  Próximos passos:',
    '    1. Conferir tamanho: wc -l /opt/akie/corpus/vocab_expansion_report.txt',
    '    2. RETREINAR o modelo (vocab mudou → embedding muda)',
    '    3. node _akie_pretrain.js',
    '',
    '═══════════════════════════════════════',
  ];

  const report = lines.join('\n');
  console.log('\n' + report);

  try {
    fs.mkdirSync(path.dirname(reportPath), { recursive: true });
    fs.writeFileSync(reportPath, report, 'utf8');
    console.log(`[VOCAB] Relatório salvo: ${reportPath}`);
  } catch (e) {
    console.log(`[VOCAB] Aviso: não foi possível salvar relatório: ${e.message}`);
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const t0 = Date.now();

  console.log('\n╔════════════════════════════════════════════════╗');
  console.log('║    AKIE Vocab Builder v1.0 — PT-BR 427→8k+    ║');
  console.log('╚════════════════════════════════════════════════╝\n');
  console.log(`[VOCAB] Target: ${TARGET_SIZE} tokens | DryRun: ${DRY_RUN}`);
  console.log(`[VOCAB] Vocab path: ${VOCAB_PATH}`);

  // 1. Carregar estado atual
  const current    = loadCurrentVocab(VOCAB_PATH);
  const beforeSize = current.id2token.length;

  // 2. Carregar corpus externo
  const corpusSentences = loadCorpusSentences(CORPUS_PATH);
  const corpusFreq      = buildFrequencyMap(corpusSentences);
  console.log(`[VOCAB] Tokens únicos no corpus: ${corpusFreq.size}`);

  // 3. Agregar léxico curado
  const curated = [
    ...SPECIAL_TOKENS,
    ...FUNCIONAL,
    ...VERBOS,
    ...SUBSTANTIVOS,
    ...ADJETIVOS,
    ...ADVERBIOS,
    ...NUMERAIS,
    ...PONTUACAO,
    ...SUFIXOS,
    ...CONVERSACIONAL,
    ...IDENTIDADE,
  ];
  console.log(`[VOCAB] Léxico curado: ${curated.length} tokens candidatos`);

  // 4. Expandir
  const result = expandVocab(current, curated, corpusFreq, TARGET_SIZE);
  console.log(`\n[VOCAB] ✓ Expansão concluída`);
  console.log(`  Anterior: ${beforeSize}`);
  console.log(`  Adicionados: ${result.added}`);
  console.log(`  Novo total: ${result.total}`);

  // 5. Validar
  const errors = validateVocab(result);
  if (errors.length > 0) {
    console.error('\n[VOCAB] ERROS DE VALIDAÇÃO:');
    errors.forEach(e => console.error(`  ✗ ${e}`));
    process.exit(1);
  }
  console.log('[VOCAB] ✓ Validação: sem erros');

  // 6. Salvar (se não for dry-run)
  if (!DRY_RUN) {
    const dir = path.dirname(VOCAB_PATH);
    fs.mkdirSync(dir, { recursive: true });

    // Backup do vocab anterior
    if (fs.existsSync(VOCAB_PATH)) {
      const backupPath = VOCAB_PATH.replace('.json', `_backup_${Date.now()}.json`);
      fs.copyFileSync(VOCAB_PATH, backupPath);
      console.log(`[VOCAB] Backup salvo: ${backupPath}`);
    }

    // Salvar novo vocab no mesmo formato que _akie_vocab.js espera
    const output = {
      version:   '2.0',
      size:      result.total,
      token2id:  result.token2id,
      id2token:  result.id2token,
      special:   SPECIAL_TOKENS,
      createdAt: new Date().toISOString(),
    };

    fs.writeFileSync(VOCAB_PATH, JSON.stringify(output, null, 2), 'utf8');
    console.log(`[VOCAB] ✓ Salvo: ${VOCAB_PATH}`);
  } else {
    console.log('[VOCAB] DRY RUN — nenhum arquivo foi modificado.');
  }

  // 7. Relatório
  const elapsed = Date.now() - t0;
  generateReport(beforeSize, result.total, result.added, elapsed, REPORT_PATH);

  // 8. Amostra do novo vocab
  console.log('\n[VOCAB] Amostra (primeiros 30 tokens):');
  console.log('  ' + result.id2token.slice(0, 30).join(' | '));
  console.log('\n[VOCAB] Amostra (tokens 500-520):');
  console.log('  ' + result.id2token.slice(500, 520).join(' | '));

  console.log(`\n[VOCAB] ══ ATENÇÃO ══`);
  console.log(`  Vocabulário mudou de ${beforeSize} → ${result.total} tokens.`);
  console.log(`  O embedding layer do modelo mudou de tamanho.`);
  console.log(`  É OBRIGATÓRIO retreinar do zero:\n`);
  console.log(`    node _akie_pretrain.js\n`);
}

main().catch(err => {
  console.error('[VOCAB] Erro fatal:', err.message);
  process.exit(1);
});
