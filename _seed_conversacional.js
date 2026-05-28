/**
 * _seed_conversacional.js
 *
 * Injeta nós conversacionais básicos PT-BR no nexus_graph.
 * Chamado automaticamente pelo worker.js no init().
 * Idempotente: verifica flag no Firestore antes de executar.
 * Após rodar uma vez, nunca mais executa (mesmo com redeploys).
 */

const SEED_VERSION = 'conversacional_v1';
const now = () => new Date().toISOString();

const NOS_CONVERSACIONAIS = [
  {
    id: 'ola', label: 'olá',
    description: 'saudação usada ao iniciar uma conversa',
    relations: [
      { target: 'saudacao',  type: 'e_tipo',       weight: 0.95 },
      { target: 'bom_dia',   type: 'relacionado_a', weight: 0.7  },
      { target: 'boa_tarde', type: 'relacionado_a', weight: 0.7  },
      { target: 'boa_noite', type: 'relacionado_a', weight: 0.7  },
      { target: 'resposta',  type: 'requer',        weight: 0.8  },
    ],
    contexts: ['inicio_conversa', 'cumprimento', 'abertura'],
    verbs: ['cumprimentar', 'iniciar', 'saudar'],
  },
  {
    id: 'oi', label: 'oi',
    description: 'forma informal de saudação',
    relations: [
      { target: 'saudacao', type: 'e_tipo',       weight: 0.95 },
      { target: 'ola',      type: 'relacionado_a', weight: 0.9  },
    ],
    contexts: ['inicio_conversa', 'informal'],
    verbs: ['cumprimentar', 'saudar'],
  },
  {
    id: 'bom_dia', label: 'bom dia',
    description: 'saudação usada durante a manhã',
    relations: [
      { target: 'saudacao', type: 'e_tipo',       weight: 0.95 },
      { target: 'ola',      type: 'relacionado_a', weight: 0.7  },
    ],
    contexts: ['manha', 'cumprimento'],
    verbs: ['desejar', 'cumprimentar'],
  },
  {
    id: 'boa_tarde', label: 'boa tarde',
    description: 'saudação usada durante a tarde',
    relations: [
      { target: 'saudacao', type: 'e_tipo', weight: 0.95 },
    ],
    contexts: ['tarde', 'cumprimento'],
    verbs: ['desejar', 'cumprimentar'],
  },
  {
    id: 'boa_noite', label: 'boa noite',
    description: 'saudação usada durante a noite',
    relations: [
      { target: 'saudacao', type: 'e_tipo', weight: 0.95 },
    ],
    contexts: ['noite', 'cumprimento'],
    verbs: ['desejar', 'cumprimentar'],
  },
  {
    id: 'saudacao', label: 'saudação',
    description: 'ato de cumprimentar alguém ao iniciar contato',
    relations: [
      { target: 'ola',      type: 'gera',    weight: 0.9  },
      { target: 'oi',       type: 'gera',    weight: 0.9  },
      { target: 'bom_dia',  type: 'gera',    weight: 0.8  },
      { target: 'resposta', type: 'requer',  weight: 0.85 },
      { target: 'conversa', type: 'precede', weight: 0.8  },
    ],
    contexts: ['cumprimento', 'inicio_conversa', 'protocolo_social'],
    verbs: ['iniciar', 'responder', 'cumprimentar'],
  },
  {
    id: 'conversa', label: 'conversa',
    description: 'troca de mensagens entre duas partes',
    relations: [
      { target: 'resposta',    type: 'requer',   weight: 0.9 },
      { target: 'pergunta',    type: 'requer',   weight: 0.8 },
      { target: 'saudacao',    type: 'segue_de', weight: 0.7 },
      { target: 'entendimento', type: 'gera',    weight: 0.7 },
    ],
    contexts: ['dialogo', 'interacao', 'comunicacao'],
    verbs: ['iniciar', 'manter', 'conduzir'],
  },
  {
    id: 'resposta', label: 'resposta',
    description: 'o que é dito em reação a uma mensagem ou pergunta',
    relations: [
      { target: 'pergunta',    type: 'segue_de', weight: 0.9 },
      { target: 'conversa',    type: 'parte_de', weight: 0.8 },
      { target: 'entendimento', type: 'requer',  weight: 0.7 },
    ],
    contexts: ['dialogo', 'reacao', 'comunicacao'],
    verbs: ['gerar', 'formular', 'elaborar'],
  },
  {
    id: 'pergunta', label: 'pergunta',
    description: 'enunciado que solicita informação ou esclarecimento',
    relations: [
      { target: 'resposta',    type: 'gera',     weight: 0.95 },
      { target: 'conversa',    type: 'parte_de', weight: 0.8  },
      { target: 'curiosidade', type: 'expressa', weight: 0.7  },
    ],
    contexts: ['dialogo', 'curiosidade', 'aprendizado'],
    verbs: ['fazer', 'formular', 'responder'],
  },
  {
    id: 'entendimento', label: 'entendimento',
    description: 'capacidade de compreender o significado de uma mensagem',
    relations: [
      { target: 'resposta',    type: 'permite', weight: 0.9 },
      { target: 'conhecimento', type: 'requer', weight: 0.8 },
    ],
    contexts: ['compreensao', 'interpretacao'],
    verbs: ['demonstrar', 'desenvolver', 'melhorar'],
  },
  {
    id: 'curiosidade', label: 'curiosidade',
    description: 'desejo de aprender ou saber mais sobre algo',
    relations: [
      { target: 'pergunta',    type: 'gera',      weight: 0.9 },
      { target: 'aprendizado', type: 'facilita',  weight: 0.8 },
    ],
    contexts: ['motivacao', 'aprendizado'],
    verbs: ['despertar', 'expressar', 'satisfazer'],
  },
  {
    id: 'ajuda', label: 'ajuda',
    description: 'suporte ou assistência oferecida a alguém',
    relations: [
      { target: 'resposta', type: 'e_tipo',  weight: 0.8 },
      { target: 'conversa', type: 'parte_de', weight: 0.7 },
    ],
    contexts: ['suporte', 'assistencia'],
    verbs: ['oferecer', 'pedir', 'dar'],
  },
  {
    id: 'conhecimento', label: 'conhecimento',
    description: 'conjunto de informações e padrões aprendidos',
    relations: [
      { target: 'aprendizado', type: 'segue_de', weight: 0.9 },
      { target: 'resposta',    type: 'permite',  weight: 0.8 },
    ],
    contexts: ['informacao', 'memoria', 'capacidade'],
    verbs: ['acumular', 'aplicar', 'expandir'],
  },
];

/**
 * Ponto de entrada — chamado pelo worker.js no init().
 * Retorna true se inseriu nós, false se já tinha rodado antes.
 *
 * @param {FirebaseFirestore.Firestore} db
 */
async function runConversationalSeed(db) {
  // ── Verificar flag de execução anterior ──
  const flagRef  = db.collection('akie_worker_status').doc('seed_conversacional');
  const flagSnap = await flagRef.get();

  if (flagSnap.exists && flagSnap.data().done) {
    console.log(`[SEED] Seed conversacional já executado em ${flagSnap.data().executed_at}. Pulando.`);
    return false;
  }

  console.log(`[SEED] Iniciando seed conversacional (${NOS_CONVERSACIONAIS.length} nós)...`);

  // ── Verificar quais já existem no grafo ──
  const ids      = NOS_CONVERSACIONAIS.map(n => n.id);
  const existing = new Set();

  for (let i = 0; i < ids.length; i += 30) {
    const chunk = ids.slice(i, i + 30);
    const snap  = await db.collection('nexus_graph')
      .where('__name__', 'in', chunk)
      .get();
    snap.forEach(doc => existing.add(doc.id));
  }

  const toInsert = NOS_CONVERSACIONAIS.filter(n => !existing.has(n.id));

  if (toInsert.length > 0) {
    const batch = db.batch();
    for (const node of toInsert) {
      const ref = db.collection('nexus_graph').doc(node.id);
      batch.set(ref, {
        ...node,
        confidence:  'confirmed',
        usage_count: 5,
        created_at:  now(),
        updated_at:  now(),
        source:      SEED_VERSION,
      });
    }
    await batch.commit();
    console.log(`[SEED] ✓ ${toInsert.length} nós inseridos: ${toInsert.map(n => n.label).join(', ')}`);
  } else {
    console.log('[SEED] Todos os nós já existiam no grafo.');
  }

  // ── Marcar como executado — nunca rodará novamente ──
  await flagRef.set({
    done:        true,
    version:     SEED_VERSION,
    inserted:    toInsert.length,
    skipped:     existing.size,
    executed_at: now(),
  });

  return toInsert.length > 0;
}

module.exports = { runConversationalSeed };
