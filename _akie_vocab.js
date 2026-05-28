/**
 * _akie_vocab.js v2.0
 * Gerenciamento incremental de vocabulário para o AKIE.
 * 
 * UPGRADE:
 *   makeTrainingPairs: maxLen 32 → 64 (contexto duplicado)
 *   Tokens especiais reservados 0-3
 *
 * Tokens especiais:
 *   0 → <PAD>   padding
 *   1 → <UNK>   palavra desconhecida
 *   2 → <BOS>   início de sequência
 *   3 → <EOS>   fim de sequência
 */

const SPECIAL = {
  PAD: 0,
  UNK: 1,
  BOS: 2,
  EOS: 3,
};

const SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<BOS>', '<EOS>'];

class Vocabulary {
  constructor() {
    this.token2id = {};
    this.id2token = [];

    // Inicializar com tokens especiais
    for (const tk of SPECIAL_TOKENS) {
      this._add(tk);
    }

    this.frozen = false;
  }

  _add(token) {
    if (!(token in this.token2id)) {
      const id = this.id2token.length;
      this.token2id[token] = id;
      this.id2token.push(token);
    }
  }

  /**
   * Adiciona tokens ao vocabulário.
   * @param {string[]} tokens
   */
  addTokens(tokens) {
    if (this.frozen) return;
    for (const t of tokens) {
      if (t && t.length > 0) this._add(t);
    }
  }

  /**
   * Converte token → ID.
   */
  encode(token) {
    return this.token2id[token] ?? SPECIAL.UNK;
  }

  /**
   * Converte ID → token.
   */
  decode(id) {
    return this.id2token[id] ?? '<UNK>';
  }

  /**
   * Tokeniza texto em array de IDs, com BOS e EOS.
   */
  tokenize(text) {
    const tokens = tokenizeText(text);
    return [
      SPECIAL.BOS,
      ...tokens.map(t => this.encode(t)),
      SPECIAL.EOS,
    ];
  }

  /**
   * Converte array de IDs → texto legível.
   */
  detokenize(ids) {
    return ids
      .filter(id => id !== SPECIAL.PAD && id !== SPECIAL.BOS && id !== SPECIAL.EOS)
      .map(id => this.decode(id))
      .join(' ')
      .replace(/ ([.,!?;:])/, '$1')
      .trim();
  }

  get size() {
    return this.id2token.length;
  }

  /**
   * Serializa para JSON (salvar no Firestore ou disco).
   */
  toJSON() {
    return {
      token2id: this.token2id,
      id2token: this.id2token,
      frozen: this.frozen,
    };
  }

  /**
   * Restaura a partir de JSON salvo.
   */
  static fromJSON(data) {
    const v = new Vocabulary();
    v.token2id = data.token2id || {};
    v.id2token = data.id2token || [...SPECIAL_TOKENS];
    v.frozen = data.frozen || false;
    return v;
  }
}

// ---------------------------------------------------------------------------
// Tokenizador de texto para PT-BR
// ---------------------------------------------------------------------------

/**
 * Converte texto em array de tokens.
 * Estratégia: word-level + pontuação separada.
 */
function tokenizeText(text) {
  if (!text || typeof text !== 'string') return [];

  return text
    .toLowerCase()
    .normalize('NFC')
    // Separar pontuação
    .replace(/([.,!?;:()\[\]{}"])/g, ' $1 ')
    // Limpar espaços múltiplos
    .replace(/\s+/g, ' ')
    .trim()
    .split(' ')
    .filter(t => t.length > 0);
}

// ---------------------------------------------------------------------------
// Preparação de sequências para treino (next-word prediction)
// ---------------------------------------------------------------------------

/**
 * Gera pares (input_seq, target_token) para treino de language model.
 * 
 * UPGRADE v2.0: maxLen padrão agora é 64 (era 32)
 * Contexto duplicado para maior capacidade de modelagem de dependências longas.
 * 
 * Dado "BOS ola e saudacao EOS":
 *   [BOS] → ola
 *   [BOS, ola] → e
 *   [BOS, ola, e] → saudacao
 *   [BOS, ola, e, saudacao] → EOS
 * 
 * @param {number[]} ids      - Sequência tokenizada
 * @param {number}   maxLen   - Comprimento máximo de contexto (default 64)
 * @returns {Array<{x: number[], y: number}>}
 */
function makeTrainingPairs(ids, maxLen = 64) {  // ← ALTERADO: 32 → 64
  const pairs = [];

  for (let i = 1; i < ids.length; i++) {
    const context = ids.slice(Math.max(0, i - maxLen), i);
    // Padding à esquerda se necessário
    const padded = [
      ...Array(Math.max(0, maxLen - context.length)).fill(SPECIAL.PAD),
      ...context,
    ];
    pairs.push({ x: padded, y: ids[i] });
  }

  return pairs;
}

module.exports = { Vocabulary, tokenizeText, makeTrainingPairs, SPECIAL };
