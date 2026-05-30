/**
 * _akie_corpus_downloader.js v4.0
 * Path: /opt/akie/app/
 * Meta: 8.000+ frases únicas PT-BR para vocab 8k-12k tokens
 * Delay: 3s sequencial — respeita rate limit Wikipedia
 *
 * USO:
 *   node _akie_corpus_downloader.js
 */

const https = require('https');
const fs    = require('fs');
const path  = require('path');

const CORPUS_DIR  = '/opt/akie/corpus';
const DELAY_MS    = 3000;
const RETRY_MS    = 15000;
const MAX_RETRIES = 2;

fs.mkdirSync(CORPUS_DIR, { recursive: true });

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function fetchJSON(url, retries = MAX_RETRIES) {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const req = https.request({
      hostname: u.hostname,
      path: u.pathname + u.search,
      method: 'GET',
      timeout: 25000,
      headers: {
        'User-Agent': 'AKIE-Corpus/4.0 (educational)',
        'Accept': 'application/json',
        'Accept-Encoding': 'identity',
      },
    }, async res => {
      if ([301,302,307,308].includes(res.statusCode)) {
        res.resume();
        try { resolve(await fetchJSON(res.headers.location, retries)); } catch(e) { reject(e); }
        return;
      }
      if (res.statusCode === 429) {
        res.resume();
        if (retries > 0) { await sleep(RETRY_MS); try { resolve(await fetchJSON(url, retries-1)); } catch(e) { reject(e); } }
        else reject(new Error('429 esgotado'));
        return;
      }
      if (res.statusCode !== 200) { res.resume(); reject(new Error(`HTTP ${res.statusCode}`)); return; }
      const chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => { try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8'))); } catch(e) { reject(e); } });
      res.on('error', reject);
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(); reject(new Error('timeout')); });
    req.end();
  });
}

// 200 tópicos PT-BR — cobertura ampla para vocab 8k+
const TOPICS = [
  // Geografia BR
  'Brasil','São_Paulo','Rio_de_Janeiro_(cidade)','Brasília','Salvador','Fortaleza',
  'Manaus','Curitiba','Recife','Porto_Alegre','Belém_(Pará)','Goiânia',
  'Florianópolis','Maceió','Natal_(Rio_Grande_do_Norte)','São_Luís','Teresina',
  'Campo_Grande','João_Pessoa','Aracaju','Porto_Velho','Macapá','Rio_Branco',
  'Palmas','Boa_Vista_(Roraima)',
  // Estados BR
  'Amazonas_(estado)','Pará','Mato_Grosso','Minas_Gerais','Bahia',
  'Rio_Grande_do_Sul','Paraná','Santa_Catarina','Goiás','Maranhão',
  'Piauí','Ceará','Rio_Grande_do_Norte','Paraíba','Pernambuco',
  'Alagoas','Sergipe','Espírito_Santo','Rio_de_Janeiro_(estado)',
  // Natureza
  'Amazônia','Rio_Amazonas','Pantanal','Cerrado','Mata_Atlântica',
  'Rio_São_Francisco','Rio_Paraná','Serra_do_Mar','Chapada_Diamantina',
  'Floresta_Amazônica','Biodiversidade','Ecossistema','Clima_tropical',
  // História BR
  'História_do_Brasil','Independência_do_Brasil','República_Velha',
  'Era_Vargas','Ditadura_militar_no_Brasil','Período_colonial_do_Brasil',
  'Pedro_I_do_Brasil','Pedro_II_do_Brasil','Proclamação_da_República_do_Brasil',
  'Abolição_da_escravatura_no_Brasil','Tiradentes','Zumbi',
  // Cultura BR
  'Carnaval_do_Brasil','Futebol_no_Brasil','Música_popular_brasileira',
  'Capoeira','Samba','Bossa_nova','Feijoada','Churrasco','Baião',
  'Frevo','Axé_(música)','Forró','Funk_carioca','Tropicalismo',
  // Literatura
  'Literatura_brasileira','Machado_de_Assis','Guimarães_Rosa',
  'Clarice_Lispector','Carlos_Drummond_de_Andrade','Fernando_Pessoa',
  'Luís_de_Camões','José_Saramago','Eça_de_Queirós','Jorge_Amado',
  'Graciliano_Ramos','Monteiro_Lobato','Cecília_Meireles',
  // Língua
  'Língua_portuguesa','Gramática','Sintaxe','Morfologia_(linguística)',
  'Fonologia','Semântica','Pragmática_(linguística)','Dialeto',
  'Língua_materna','Bilinguismo',
  // Países lusófonos
  'Portugal','Angola','Moçambique','Cabo_Verde','Guiné-Bissau',
  'São_Tomé_e_Príncipe','Timor-Leste',
  // Ciência e tecnologia
  'Inteligência_artificial','Aprendizado_de_máquina','Rede_neural_artificial',
  'Computador','Internet','Algoritmo','Robótica','Programação',
  'Banco_de_dados','Segurança_da_informação','Computação_em_nuvem',
  'Big_data','Processamento_de_linguagem_natural','Visão_computacional',
  'Blockchain','Criptomoeda','Realidade_virtual','Realidade_aumentada',
  // Ciências naturais
  'Física','Química','Biologia','Matemática','Astronomia',
  'Medicina','Genética','Ecologia','Geologia','Oceanografia',
  'Meteorologia','Botânica','Zoologia','Microbiologia','Bioquímica',
  'Neurociência','Imunologia','Epidemiologia','Farmacologia',
  // Física e astronomia
  'Mecânica_quântica','Relatividade_geral','Termodinâmica',
  'Eletromagnetismo','Sistema_Solar','Via_Láctea','Buraco_negro',
  'Estrela','Planeta','Lua','Marte','Júpiter',
  // Matemática
  'Álgebra','Geometria','Cálculo','Estatística','Probabilidade',
  'Teoria_dos_números','Lógica_matemática',
  // Sociedade
  'Educação','Economia','Democracia','Direitos_humanos',
  'Meio_ambiente','Sustentabilidade','Energia_solar','Energia_eólica',
  'Mudança_climática','Desenvolvimento_sustentável','Pobreza',
  'Desigualdade_social','Globalização','Urbanização',
  // Filosofia e ciências humanas
  'Filosofia','Ética','Epistemologia','Ontologia','Lógica',
  'Psicologia','Sociologia','Antropologia','História','Geografia',
  'Economia_política','Ciência_política','Direito',
  // Arte e cultura
  'Arte','Cinema','Teatro','Dança','Fotografia','Arquitetura',
  'Pintura','Escultura','Música','Ópera','Ballet',
  // Esportes
  'Futebol','Vôlei','Basquete','Tênis','Natação','Atletismo',
  'Fórmula_1','Boxe','Judô','Surfe',
  // Animais BR
  'Onça-pintada','Tucano','Arara','Boto-cor-de-rosa','Capivara',
  'Tamanduá','Preguiça','Mico-leão-dourado','Tartaruga-marinha',
  // Saúde
  'Saúde','Nutrição','Sistema_imunológico','Vacinação',
  'Saúde_mental','Diabetes','Hipertensão','Câncer',
  // Tecnologia cotidiana
  'Telefone_celular','Televisão','Rádio','Jornal','Fotografia_digital',
];

function extractSentences(text) {
  if (!text || text.length < 10) return [];
  return text
    .replace(/\n+/g, ' ').replace(/\s+/g, ' ')
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim().toLowerCase())
    .filter(s =>
      s.length >= 20 && s.length <= 500 &&
      /[a-záéíóúãõâêôçà]/i.test(s) &&
      !/[\[\]{}<>=|\\]/.test(s) &&
      (s.match(/\s/g) || []).length >= 3
    );
}

// Corpus embutido garantido
const EMBEDDED = [
  'sou a akie, uma inteligência artificial.','meu nome é akie.','me chamo akie.',
  'sou a akie, parte do ecossistema aether.','aprendo com cada conversa.',
  'cada interação me ensina algo novo.','estou sempre aprendendo e melhorando.',
  'olá, como vai?','bom dia, tudo bem?','boa tarde, como posso ajudar?',
  'boa noite, tudo certo?','oi, tudo bem?','como você está?',
  'sim, com certeza.','não, obrigada.','claro, pode perguntar.',
  'brasília é a capital do brasil.','o brasil fica na américa do sul.',
  'a língua oficial do brasil é o português.','são paulo é a maior cidade do brasil.',
  'o rio amazonas é o maior rio do mundo em volume de água.',
  'a amazônia é a maior floresta tropical do planeta.',
  'o real é a moeda oficial do brasil.',
  'o futebol é o esporte mais popular do brasil.',
  'inteligência artificial é a capacidade de máquinas realizarem tarefas cognitivas.',
  'redes neurais artificiais são inspiradas no funcionamento do cérebro humano.',
  'o aprendizado de máquina utiliza dados para melhorar o desempenho.',
  'processamento de linguagem natural permite que computadores entendam texto.',
  'eu falo português.','você fala muito bem.','ele estuda bastante.',
  'nós aprendemos juntos.','eu gosto de aprender.','você precisa de ajuda?',
  'quando chove, fico em casa.','se você precisar, pode me perguntar.',
  'embora seja difícil, vou tentar.','porque aprendo, consigo melhorar.',
  'apesar de não saber tudo, faço o meu melhor.',
  'aprender é um processo contínuo.','cada erro é uma oportunidade de melhora.',
  'a prática leva à perfeição.','o conhecimento é fundamental.',
  'um mais um é igual a dois.','há sete dias na semana.',
  'o sol nasce no leste e se põe no oeste.',
  'portugal fica na europa ocidental.',
  'o português é falado em vários países ao redor do mundo.',
  'angola e moçambique também têm o português como língua oficial.',
  'não tenho certeza sobre isso.','talvez seja possível.',
  'não sei ao certo.','é uma pergunta interessante.',
  'preciso pensar melhor sobre isso.','vou tentar explicar da melhor forma.',
];

async function main() {
  console.log('\n[CORPUS v4.0] Iniciando download — 200 tópicos PT-BR');
  console.log(`[CORPUS v4.0] Meta: 8.000+ frases | Delay: ${DELAY_MS}ms`);
  console.log(`[CORPUS v4.0] Tempo estimado: ~${Math.ceil(200 * DELAY_MS / 60000)} minutos\n`);

  let all = [...EMBEDDED];
  let ok = 0, fail = 0;

  for (let i = 0; i < TOPICS.length; i++) {
    const topic = TOPICS[i];
    try {
      const url  = `https://pt.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(topic)}`;
      const data = await fetchJSON(url);
      const frases = extractSentences(data.extract || '');
      all.push(...frases);
      ok++;
      process.stdout.write(`\r[CORPUS v4.0] ${i+1}/${TOPICS.length} | ok=${ok} fail=${fail} | total=${all.length}    `);
    } catch(e) {
      fail++;
      process.stdout.write(`\r[CORPUS v4.0] ${i+1}/${TOPICS.length} | ok=${ok} fail=${fail} | total=${all.length}    `);
    }
    if (i < TOPICS.length - 1) await sleep(DELAY_MS);
  }

  console.log(`\n\n[CORPUS v4.0] Wikipedia: ${ok} ok, ${fail} falhas`);

  // Deduplicar e filtrar
  const unique = [...new Set(all.map(s => s.trim().toLowerCase()).filter(s => s.length > 10))];
  console.log(`[CORPUS v4.0] Frases únicas: ${unique.length}`);

  // Salvar sentences
  fs.writeFileSync(path.join(CORPUS_DIR, 'sentences_ptbr.txt'), unique.join('\n'), 'utf8');

  // Vocab freq
  const freq = new Map();
  for (const s of unique) {
    for (const t of s.replace(/[.,!?;:()\[\]{}"]/g, ' ').split(/\s+/).filter(t => t.length >= 2)) {
      freq.set(t, (freq.get(t)||0) + 1);
    }
  }
  const sorted = [...freq.entries()].sort((a,b) => b[1]-a[1]);
  fs.writeFileSync(path.join(CORPUS_DIR, 'vocab_raw.txt'), sorted.map(([t,c]) => `${t}\t${c}`).join('\n'), 'utf8');

  console.log(`[CORPUS v4.0] Vocab único: ${sorted.length} tokens`);
  console.log(`[CORPUS v4.0] Tokens freq>2: ${sorted.filter(([,c])=>c>2).length}`);
  console.log(`\n[CORPUS v4.0] Salvo em: ${CORPUS_DIR}`);
  console.log(`[CORPUS v4.0] PRÓXIMO: node _akie_pretrain.js\n`);
}

main().catch(e => { console.error('\n[CORPUS v4.0] ERRO:', e.message); process.exit(1); });
