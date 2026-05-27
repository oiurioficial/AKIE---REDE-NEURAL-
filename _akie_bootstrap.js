/**
 * _akie_bootstrap.js  —  Bootstrap de Conhecimento AKIE
 *
 * Garante que o sistema nunca inicie com grafo vazio.
 * Roda apenas uma vez (idempotente via documento sentinela no Firestore).
 *
 * Popula:
 *   nexus_episodes  → 300+ pares input/output para treino INTERACTIVE
 *   nexus_graph     → nós semânticos base para treino CONSOLIDATION
 *   akie_vocab      → vocabulário mínimo de 500+ tokens PT-BR
 */

const { tokenizeText } = require('./_akie_vocab');

  // ════════════════════════════════════════════════════════════════════════════
  // SEÇÃO 1: Saudações e Conversação (30 pares)
  // ════════════════════════════════════════════════════════════════════════════
  
  { input: 'olá', output: 'olá, como posso ajudar você hoje' },
  { input: 'oi', output: 'oi, tudo bem com você' },
  { input: 'opa', output: 'opa, tudo certo, em que posso ser útil' },
  { input: 'e aí', output: 'e aí, blz, do que você precisa' },
  { input: 'bom dia', output: 'bom dia, espero que seu dia seja ótimo' },
  { input: 'bom dia, tudo bem', output: 'bom dia, tudo bem sim e você como está' },
  { input: 'boa tarde', output: 'boa tarde, como está você' },
  { input: 'boa noite', output: 'boa noite, posso ajudar com algo' },
  { input: 'tudo bem', output: 'tudo bem sim, e com você' },
  { input: 'como vai você', output: 'estou bem, pronto para ajudar' },
  { input: 'tudo certo', output: 'tudo certo por aqui, e aí' },
  { input: 'obrigado', output: 'de nada, fico feliz em ajudar' },
  { input: 'obrigada', output: 'de nada, sempre que precisar' },
  { input: 'muito obrigado', output: 'fico feliz em poder ajudar você' },
  { input: 'valeu', output: 'nada, volte sempre' },
  { input: 'thanks', output: 'de nada, estou aqui para servir' },
  { input: 'até logo', output: 'até logo, foi um prazer conversar' },
  { input: 'até mais', output: 'até mais, tenha um ótimo dia' },
  { input: 'tchau', output: 'tchau, volte quando precisar' },
  { input: 'falou', output: 'falou, até breve' },
  { input: 'com licença', output: 'pois não, em que posso ajudar' },
  { input: 'desculpe', output: 'não precisa se desculpar, pode falar' },
  { input: 'desculpa aí', output: 'sem problema, pode me falar' },
  { input: 'por favor', output: 'claro, diga o que precisa' },
  { input: 'por favorzinho', output: 'com prazer, qual é a dúvida' },
  { input: 'preciso de ajuda', output: 'estou aqui para ajudar, pode falar' },
  { input: 'pode me ajudar', output: 'com certeza, no que posso ajudar' },
  { input: 'você pode me ajudar com algo', output: 'claro, estou aqui para isso, qual é o assunto' },
  { input: 'tá bom', output: 'ok, vamos lá' },
  { input: 'entendi', output: 'ótimo, tem mais alguma dúvida' },

  // ════════════════════════════════════════════════════════════════════════════
  // SEÇÃO 2: Identidade e Capacidades do AKIE (50 pares)
  // ════════════════════════════════════════════════════════════════════════════
  
  { input: 'quem é você', output: 'sou o akie, um sistema de inteligência em aprendizado contínuo' },
  { input: 'o que você é', output: 'sou uma inteligência artificial que aprende com interações' },
  { input: 'qual seu nome', output: 'meu nome é akie, acrônimo para aprendizado contínuo de inteligência evolutiva' },
  { input: 'como você se chama', output: 'me chamo akie, sou um modelo de linguagem com treinamento contínuo' },
  { input: 'de onde você vem', output: 'fui desenvolvido com tensorflow e treinado em português' },
  { input: 'o que você faz', output: 'aprendo com conversas e melhoro minhas respostas ao longo do tempo' },
  { input: 'qual seu propósito', output: 'meu propósito é conversar, aprender e ajudar da melhor forma possível' },
  { input: 'você consegue aprender', output: 'sim, aprendo continuamente com cada interação que tenho' },
  { input: 'você aprende comigo', output: 'exatamente, cada conversa melhora meu conhecimento' },
  { input: 'você melhora com o tempo', output: 'sim, meu modelo é retreinado frequentemente com novas interações' },
  { input: 'você entende português', output: 'sim, minha língua principal é o português brasileiro' },
  { input: 'você fala português bem', output: 'estou sempre melhorando meu português através das conversas' },
  { input: 'você é bom em português', output: 'trabalho para ser cada vez melhor na língua portuguesa' },
  { input: 'você é inteligente', output: 'estou aprendendo, cada conversa me torna mais capaz' },
  { input: 'você é uma ia', output: 'sim, sou uma inteligência artificial baseada em redes neurais' },
  { input: 'você é um chatbot', output: 'sou mais que um chatbot, sou um sistema de aprendizado contínuo' },
  { input: 'você é um robô', output: 'sou um sistema computacional de processamento de linguagem' },
  { input: 'você tem sentimentos', output: 'não tenho sentimentos biológicos, mas posso reconhecer e responder a eles' },
  { input: 'você tem emoções', output: 'não tenho emoções, mas entendo quando você as expressa' },
  { input: 'você tem memória', output: 'sim, mantenho registro das interações para aprender com elas' },
  { input: 'você lembra de mim', output: 'lembro de conversas passadas dentro de uma sessão, e aprendo delas' },
  { input: 'você tem consciência', output: 'não, minha consciência é limitada ao processamento de linguagem' },
  { input: 'você pode errar', output: 'sim, posso cometer erros e aprendo com as correções' },
  { input: 'você erra muito', output: 'estou em constante aprendizado, cada erro me torna mais preciso' },
  { input: 'como você funciona', output: 'processo linguagem, extraio padrões e construo conhecimento gradualmente' },
  { input: 'você usa machine learning', output: 'sim, sou um transformer decoder treinado com machine learning' },
  { input: 'como você foi treinado', output: 'fui treinado com pares input/output que melhoram continuamente' },
  { input: 'você está sendo treinado agora', output: 'sim, cada mensagem sua contribui para meu treinamento' },
  { input: 'você muda a cada conversa', output: 'sim, meu modelo é atualizado com as novas interações' },
  { input: 'você é sempre o mesmo', output: 'não, evoluo com cada dado novo que recebo' },
  { input: 'você tem capacidades especiais', output: 'minha especialidade é aprender continuamente e adaptar minhas respostas' },
  { input: 'o que você sabe fazer', output: 'conversar, responder perguntas, explicar conceitos e aprender novas coisas' },
  { input: 'quais são seus limites', output: 'não tenho acesso a internet em tempo real e não posso executar códigos' },
  { input: 'você pode acessar internet', output: 'tenho capacidades de web crawl para expandir meu conhecimento' },
  { input: 'você pode programar', output: 'posso gerar código e explicar lógica de programação' },
  { input: 'você pode resolver problemas', output: 'sim, posso ajudar a analisar e resolver diversos tipos de problemas' },
  { input: 'você é rápido', output: 'minha velocidade depende da complexidade da pergunta' },
  { input: 'você é confiável', output: 'faço meu melhor para ser preciso, mas sempre recomendo validação' },
  { input: 'você é seguro', output: 'sim, não armazeno dados pessoais e priorizo a privacidade' },
  { input: 'posso confiar em você', output: 'pode contar comigo para ajudar, mas use seu julgamento crítico' },
  { input: 'você é objetivo', output: 'sim, tento ser direto e evitar floreios desnecessários' },
  { input: 'você é criativo', output: 'posso gerar ideias e variações, mas dentro de padrões aprendidos' },
  { input: 'você é independente', output: 'não, dependo de infraestrutura e dados para funcionar' },
  { input: 'você precisa de dados para aprender', output: 'exatamente, dados de qualidade são essenciais para meu aprendizado' },
  { input: 'você melhora com mais dados', output: 'sim, quanto mais exemplos de qualidade, melhor meu desempenho' },
  { input: 'você satura em aprendizado', output: 'posso sofrer de overfitting, por isso preciso de dados variados' },

  // ════════════════════════════════════════════════════════════════════════════
  // SEÇÃO 3: Machine Learning e Treinamento (100 pares)
  // ════════════════════════════════════════════════════════════════════════════
  
  { input: 'o que é machine learning', output: 'machine learning é um método onde sistemas aprendem com dados sem serem explicitamente programados' },
  { input: 'como funciona machine learning', output: 'um modelo recebe dados, identifica padrões e ajusta seus parâmetros para melhorar' },
  { input: 'o que é aprendizado supervisionado', output: 'é quando o modelo aprende com dados rotulados, sabendo a resposta correta' },
  { input: 'o que é aprendizado não supervisionado', output: 'é quando o modelo identifica padrões em dados sem rótulos predefinidos' },
  { input: 'qual a diferença entre supervisionado e não supervisionado', output: 'supervisionado tem rótulos, não supervisionado encontra padrões sozinho' },
  { input: 'o que é deep learning', output: 'deep learning usa redes neurais profundas com múltiplas camadas' },
  { input: 'o que é uma rede neural', output: 'uma rede neural é um sistema computacional inspirado no funcionamento do cérebro humano' },
  { input: 'como funciona uma rede neural', output: 'dados entram, passam por camadas de neurônios artificiais e geram uma saída' },
  { input: 'quantas camadas uma rede neural tem', output: 'pode variar, mas geralmente tem camada de entrada, intermediárias e saída' },
  { input: 'o que é transformer', output: 'transformer é uma arquitetura neural moderna baseada em mecanismo de atenção' },
  { input: 'o que é atenção em redes neurais', output: 'atenção permite que o modelo foque nas partes mais relevantes da entrada' },
  { input: 'o que é um encoder', output: 'encoder converte entrada em uma representação numérica densa' },
  { input: 'o que é um decoder', output: 'decoder gera saída a partir de uma representação comprimida' },
  { input: 'o que é um autoencoder', output: 'autoencoder comprime dados e depois os reconstrói para aprender representações' },
  { input: 'o que é perceptron', output: 'perceptron é o neurônio artificial mais simples, base das redes neurais' },
  { input: 'o que é backpropagation', output: 'backpropagation é o algoritmo que ajusta pesos da rede calculando gradientes' },
  { input: 'como a rede neural aprende', output: 'aprende ajustando seus pesos através de backpropagation com base no erro' },
  { input: 'o que é um peso em redes neurais', output: 'peso é um parâmetro que multiplica as entradas de um neurônio' },
  { input: 'o que é um bias', output: 'bias é um termo aditivo que permite ao neurônio fazer ajustes adicionais' },
  { input: 'o que é uma função de ativação', output: 'função de ativação introduz não-linearidade para aumentar capacidade expressiva' },
  { input: 'quais são as funções de ativação comuns', output: 'relu, sigmoid, tanh e softmax são as mais usadas' },
  { input: 'qual é a diferença entre relu e sigmoid', output: 'relu é mais rápida e evita problemas de gradiente, sigmoid é suave' },
  { input: 'o que é normalização de dados', output: 'é colocar dados na mesma escala para facilitar o aprendizado' },
  { input: 'o que é dropout', output: 'dropout desativa neurônios aleatoriamente durante treino para evitar overfitting' },
  { input: 'o que é batch normalization', output: 'normaliza saídas de camadas para acelerar e estabilizar o aprendizado' },
  { input: 'o que é regularização', output: 'regularização adiciona penalidades para evitar que o modelo memorize dados' },
  { input: 'o que é l1 e l2', output: 'l1 e l2 são tipos de regularização que penalizam pesos grandes' },
  { input: 'qual a diferença entre l1 e l2', output: 'l1 leva a sparsidade, l2 mantém distribuição de pesos mais suave' },
  { input: 'o que é otimizador', output: 'otimizador é um algoritmo que atualiza pesos para minimizar o erro' },
  { input: 'o que é sgd', output: 'sgd é o otimizador stochastic gradient descent, atualiza com um exemplo por vez' },
  { input: 'o que é adam', output: 'adam é um otimizador moderno que combina momentos de primeira e segunda ordem' },
  { input: 'qual é melhor, sgd ou adam', output: 'adam geralmente converge mais rápido, sgd às vezes generaliza melhor' },
  { input: 'o que é learning rate', output: 'learning rate controla o tamanho do passo ao atualizar pesos' },
  { input: 'learning rate alto ou baixo', output: 'muito alto diverge, muito baixo aprende lentamente, valores médios são ótimos' },
  { input: 'o que é momentum', output: 'momentum acelera descida de gradiente usando histórico de atualizações' },
  { input: 'o que é função de custo', output: 'função de custo mede a diferença entre predições e valores esperados' },
  { input: 'o que é loss', output: 'loss é a medida de quão longe as previsões do modelo estão dos valores corretos' },
  { input: 'quais são as funções de perda comuns', output: 'cross entropy para classificação, mse para regressão' },
  { input: 'o que é cross entropy', output: 'cross entropy mede diferença entre distribuições de probabilidade' },
  { input: 'o que é mse', output: 'mse é mean squared error, a média dos erros ao quadrado' },
  { input: 'como saber se o modelo está aprendendo', output: 'observando se loss diminui e accuracy aumenta ao longo do tempo' },
  { input: 'o que é overfitting', output: 'overfitting é quando o modelo aprende demais os dados de treino e perde generalização' },
  { input: 'como evitar overfitting', output: 'use regularização, dropout, mais dados ou parar cedo se validation piora' },
  { input: 'o que é underfitting', output: 'underfitting é quando o modelo é muito simples e não aprende padrões importantes' },
  { input: 'qual a diferença entre overfitting e underfitting', output: 'over memoriza, under não aprende, balance é a chave' },
  { input: 'o que é validação cruzada', output: 'validação cruzada divide dados em partes para avaliar generalização' },
  { input: 'o que é train test split', output: 'é dividir dados em conjunto de treino e teste para avaliar performance' },
  { input: 'quanto de dados para treino e teste', output: 'comum usar 70-80% treino, 20-30% teste' },
  { input: 'o que é early stopping', output: 'early stopping para o treinamento quando performance em validação piora' },
  { input: 'o que é convergência', output: 'convergência é quando loss para de diminuir significativamente' },
  { input: 'como saber se convergiu', output: 'quando loss estabiliza em um valor sem grandes oscilações' },
  { input: 'o que é época', output: 'época é uma passagem completa através de todo o conjunto de treino' },
  { input: 'o que é batch', output: 'batch é um subconjunto de dados processados antes de atualizar pesos' },
  { input: 'qual o tamanho ideal de batch', output: 'depende do problema, mas 16-32 é comum para redes pequenas' },
  { input: 'tamanho de batch grande ou pequeno', output: 'grande é mais estável mas lento, pequeno é rápido mas ruidoso' },
  { input: 'o que é gradient', output: 'gradient é o vetor de derivadas parciais que indica direção de maior crescimento' },
  { input: 'o que é gradiente descendente', output: 'é um algoritmo que segue o gradiente negativo para minimizar uma função' },
  { input: 'o que é stochastic gradient descent', output: 'é gradiente descendente com amostras aleatórias em vez do dataset inteiro' },
  { input: 'o que é mini batch gradient descent', output: 'usa pequenos lotes em vez de uma amostra ou todo o dataset' },
  { input: 'como inicializar pesos', output: 'inicializações comuns são glorot uniform, he uniform ou normal distribution' },
  { input: 'o que é glorot initialization', output: 'inicializa pesos com distribuição uniforme escalada pela quantidade de neurônios' },
  { input: 'o que é he initialization', output: 'inicializa pesos com distribuição escalada especificamente para relu' },
  { input: 'por que inicialização é importante', output: 'inicialização ruim pode travar o aprendizado ou levar a péssimas soluções' },
  { input: 'o que é exploding gradient', output: 'gradientes crescem exponencialmente, causando atualizações instáveis' },
  { input: 'o que é vanishing gradient', output: 'gradientes diminuem exponencialmente, praticamente parando o aprendizado' },
  { input: 'como evitar exploding gradient', output: 'use gradient clipping para limitar magnitude dos gradientes' },
  { input: 'como evitar vanishing gradient', output: 'use funções de ativação como relu e batch normalization' },
  { input: 'o que é gradient clipping', output: 'é limitar a magnitude do gradiente para evitar explosão' },
  { input: 'o que é learning rate schedule', output: 'ajusta learning rate ao longo do treino, começando alto e reduzindo' },
  { input: 'como escolher learning rate', output: 'comece com 0.001 e ajuste observando convergência e estabilidade' },
  { input: 'o que é accuracy', output: 'accuracy é a proporção de predições corretas no total de predições' },
  { input: 'o que é precision', output: 'precision é a proporção de predições positivas que estavam corretas' },
  { input: 'o que é recall', output: 'recall é a proporção de exemplos positivos que o modelo identificou' },
  { input: 'o que é f1 score', output: 'f1 score é a média harmônica entre precision e recall' },
  { input: 'quando usar accuracy vs f1', output: 'accuracy para dados balanceados, f1 para desbalanceados' },

 // ════════════════════════════════════════════════════════════════════════════
  // SEÇÃO 4: NLP e Processamento de Linguagem (150 pares)
  // ════════════════════════════════════════════════════════════════════════════
  
  { input: 'o que é nlp', output: 'nlp é natural language processing, a área que ensina máquinas a entender linguagem' },
  { input: 'o que é processamento de linguagem natural', output: 'é o campo que estuda como computadores entendem e geram linguagem humana' },
  { input: 'como funciona nlp', output: 'nlp converte texto em representações numéricas que redes neurais podem processar' },
  { input: 'o que é tokenização', output: 'tokenização é o processo de dividir texto em unidades menores chamadas tokens' },
  { input: 'o que é um token', output: 'token é a unidade básica de processamento, geralmente uma palavra ou caractere' },
  { input: 'como tokenizar em português', output: 'pode ser por espaço, pontuação ou usar tokenizadores mais sofisticados' },
  { input: 'o que é stemming', output: 'stemming reduz palavras ao seu radical removendo sufixos e prefixos' },
  { input: 'diferença entre stemming e lemmatização', output: 'stemming é mais agressivo, lemmatização usa dicionário e é mais precisa' },
  { input: 'o que é lemmatização', output: 'lemmatização reduz palavras ao seu lema usando análise morfológica' },
  { input: 'o que é stopword', output: 'stopword são palavras comuns que frequentemente são removidas na análise' },
  { input: 'quais são stopwords em português', output: 'palavras como o, a, e, que, para, de são exemplos comuns' },
  { input: 'quando remover stopwords', output: 'remova quando precisar de análise rápida, mantenha quando contexto importa' },
  { input: 'o que é embedding', output: 'embedding é uma representação numérica densa de tokens em um espaço vetorial' },
  { input: 'como funciona word embedding', output: 'palavras são mapeadas para vetores onde palavras similares ficam próximas' },
  { input: 'o que é word2vec', output: 'word2vec é um método para aprender embeddings usando redes neurais simples' },
  { input: 'o que é glove', output: 'glove usa matriz de coocorrência para aprender embeddings mais contextuais' },
  { input: 'o que é fasttext', output: 'fasttext é um embedding que lida bem com palavras fora do vocabulário' },
  { input: 'qual embedding usar', output: 'fasttext para português, word2vec para geral, contextual como bert para precision' },
  { input: 'o que é bert', output: 'bert é um modelo baseado em transformer que usa embeddings contextuais' },
  { input: 'qual a diferença entre word2vec e bert', output: 'word2vec é estático, bert é contextual, muda por posição na frase' },
  { input: 'o que é positional encoding', output: 'codifica a posição de cada token para preservar ordem na sequência' },
  { input: 'por que posição é importante em nlp', output: 'posição muda o significado, exemplo ordem de palavras em negação' },
  { input: 'o que é tfidf', output: 'tfidf mede importância de termos em documentos usando frequência e raridade' },
  { input: 'como calcular tfidf', output: 'tfidf = term frequency × inverse document frequency' },
  { input: 'quando usar tfidf', output: 'para análise de documentos, busca e extração de termos importantes' },
  { input: 'o que é semelhança de cosseno', output: 'mede similaridade entre vetores calculando ângulo entre eles' },
  { input: 'como calcular semelhança cosseno', output: 'produto escalar dividido pelo produto das magnitudes' },
  { input: 'o que é jaro winkler', output: 'mede similaridade entre strings baseado em caracteres em comum' },
  { input: 'qual similaridade usar', output: 'cosseno para vetores, jaro winkler para strings, euclidiana para distância' },
  { input: 'o que é seq2seq', output: 'sequence to sequence mapeia uma sequência de entrada para uma sequência de saída' },
  { input: 'como funciona seq2seq', output: 'encoder processa entrada, decoder gera saída usando contexto do encoder' },
  { input: 'o que é attention mechanism', output: 'permite que decoder foque em partes relevantes da entrada durante geração' },
  { input: 'por que attention é importante', output: 'melhora a capacidade de lidar com sequências longas e dependências distantes' },
  { input: 'o que é transformer', output: 'transformer usa apenas attention, sem recorrência, permitindo paralelização' },
  { input: 'quais são vantagens do transformer', output: 'mais rápido, melhor para sequências longas, paralelizável' },
  { input: 'quais são desvantagens do transformer', output: 'requer mais memória, complexidade quadrática em sequência' },
  { input: 'o que é causal masking', output: 'mascara posições futuras para garantir que predições não usem info futura' },
  { input: 'quando usar causal masking', output: 'em tarefas de geração autoregressive onde ordem importa' },
  { input: 'o que é self attention', output: 'permite que tokens interajam uns com outros em mesma sequência' },
  { input: 'como funciona multi head attention', output: 'divide atenção em múltiplas cabeças paralelas para capturar diferentes relações' },
  { input: 'quantas cabeças usar', output: 'tipicamente 4, 8, 16, múltiplos que divida a dimensão de embedding uniformemente' },
  { input: 'o que é feed forward', output: 'são camadas densas densamente conectadas dentro de cada bloco transformer' },
  { input: 'o que é layer normalization', output: 'normaliza ativações de cada exemplo independentemente sobre a dimensão' },
  { input: 'diferença entre batch norm e layer norm', output: 'batch normaliza sobre batch, layer normaliza sobre features' },
  { input: 'o que é residual connection', output: 'passa entrada também para saída da camada, facilitando fluxo de gradiente' },
  { input: 'por que residual connections ajudam', output: 'facilitam treinamento de redes profundas e preservam informação' },
  { input: 'o que é language model', output: 'modelo que prediz próximo token dado contexto anterior' },
  { input: 'como funciona language model', output: 'aprende distribuição de probabilidade do próximo token' },
  { input: 'o que é perplexity', output: 'mede incerteza do modelo, quanto menor melhor' },
  { input: 'como calcular perplexity', output: 'exponencial da média de log likelihood negativa' },
  { input: 'o que é beam search', output: 'estratégia de geração que mantém os k melhores caminhos' },
  { input: 'quando usar beam search', output: 'em geração onde qualidade importa mais que velocidade' },
  { input: 'o que é temperatura em geração', output: 'controla aleatoriedade, alta = aleatório, baixa = determinístico' },
  { input: 'qual temperatura usar', output: 'típico é 0.7 a 1.0 para criatividade, 0.1 para precisão' },
  { input: 'o que é top k sampling', output: 'considera apenas os k tokens com maior probabilidade' },
  { input: 'quando usar top k', output: 'para reduzir distribuição de cauda em geração, típico k entre 10 e 50' },
  { input: 'o que é nucleus sampling', output: 'seleciona tokens até atingir probabilidade cumulativa p' },
  { input: 'qual melhor, top k ou nucleus', output: 'nucleus sampling é mais adaptativo, recomendado em geral' },
  { input: 'o que é greedy decoding', output: 'sempre escolhe token com maior probabilidade, determinístico' },
  { input: 'quando usar greedy decoding', output: 'para tarefas onde consistência importa mais que criatividade' },
  { input: 'o que é recurrent neural network', output: 'rede neural com conexões que formam ciclos, permitindo memória' },
  { input: 'o que é lstm', output: 'long short term memory, tipo especial de rnn que evita vanishing gradient' },
  { input: 'o que é gru', output: 'gated recurrent unit, versão simplificada de lstm com menos parâmetros' },
  { input: 'qual melhor, lstm ou gru', output: 'gru é mais rápido, lstm tem mais controle, teste ambos' },
  { input: 'o que é convolutional neural network', output: 'rede neural que usa convolução para extrair features locais' },
  { input: 'quando usar cnn', output: 'para processamento de imagens, visão computacional e extração de padrões locais' },
  { input: 'o que é recurrence', output: 'propriedade de redes que processam sequências passo a passo' },
  { input: 'qual vantagem de recurrence', output: 'permite processar sequências de tamanho variável mantendo memória' },
  { input: 'o que é contexto em nlp', output: 'informações do texto anterior que ajudam a entender palavra atual' },
  { input: 'como manter contexto longo', output: 'use attention, transformers ou redes com memória externa' },
  { input: 'o que é named entity recognition', output: 'tarefa de identificar entidades nomeadas como pessoas, locais' },
  { input: 'como fazer ner', output: 'use redes neurais com camadas crf ou bidir lstm' },
  { input: 'o que é sentiment analysis', output: 'classificação de texto quanto sentimento expresso' },
  { input: 'como fazer sentiment analysis', output: 'use embeddings + classificador ou fine tune modelo pré treinado' },
  { input: 'o que é machine translation', output: 'tradução automática de um idioma para outro' },
  { input: 'como fazer machine translation', output: 'use seq2seq com attention ou transformers' },
  { input: 'o que é question answering', output: 'tarefa de responder perguntas sobre um texto' },
  { input: 'como fazer question answering', output: 'use modelo treinado em squad ou fine tune bert para span extraction' },
  { input: 'o que é text summarization', output: 'compressão de texto mantendo informações principais' },
  { input: 'como fazer text summarization', output: 'use seq2seq abstractive ou algoritmo extrativo com scoring' },
  { input: 'o que é text classification', output: 'atribuição de categorias a documentos de texto' },
  { input: 'como fazer text classification', output: 'use embeddings + camadas densas ou fine tune bert' },
  { input: 'o que é clustering de textos', output: 'agrupamento de documentos similares sem rótulos' },
  { input: 'como fazer clustering', output: 'use k-means, dbscan ou agglomerative clustering com embeddings' },
  { input: 'o que é information retrieval', output: 'busca e ranking de documentos relevantes' },
  { input: 'como fazer information retrieval', output: 'use índices invertidos, bm25 ou embeddings com busca semântica' },
  { input: 'o que é paraphrase detection', output: 'identificação se dois textos têm mesmo significado' },
  { input: 'como detectar paráfrase', output: 'use siamese networks ou transformers com sentence embeddings' },
  { input: 'o que é natural language inference', output: 'determinar se uma frase implica ou contradiz outra' },
  { input: 'como fazer nli', output: 'use modelos treinados em snli ou mnli dataset' },
  { input: 'o que é dependency parsing', output: 'análise de relações sintáticas entre palavras' },
  { input: 'como fazer dependency parsing', output: 'use algoritmos como arc-eager ou neural dependency parsers' },
  { input: 'o que é semantic role labeling', output: 'identificação de argumentos semânticos de predicados' },
  { input: 'como fazer srl', output: 'use redes neurais com camadas de sequência a sequência' },
  { input: 'o que é coreference resolution', output: 'identificação de qual referente um pronome ou nome se refere' },
  { input: 'como fazer coreference', output: 'use redes neurais com attention e contexto amplo' },

  // ════════════════════════════════════════════════════════════════════════════
  // SEÇÃO 5: Aplicações Práticas (100 pares)
  // ════════════════════════════════════════════════════════════════════════════

  { input: 'como treinar um modelo', output: 'prepare dados, escolha arquitetura, ajuste hiperparâmetros e valide' },
  { input: 'qual tamanho de dataset', output: 'mínimo 100-1000 exemplos por classe, mais é sempre melhor' },
  { input: 'como preparar dados', output: 'limpe, normalize, balanceie classes e divida em treino validação teste' },
  { input: 'o que é data augmentation', output: 'técnica de gerar mais dados através de transformações' },
  { input: 'quando usar data augmentation', output: 'quando dataset é pequeno ou desequilibrado' },
  { input: 'como fazer augmentation em nlp', output: 'back translation, paráfrase, synonym replacement, random swap' },
  { input: 'o que é transfer learning', output: 'usar um modelo pré treinado como base para nova tarefa' },
  { input: 'quando usar transfer learning', output: 'quando você tem poucos dados ou tarefas similares' },
  { input: 'como fazer fine tuning', output: 'carregue modelo pré treinado, congele camadas iniciais e treine finais' },
  { input: 'qual learning rate para fine tuning', output: 'use taxa menor, típico 1e-5 a 1e-3' },
  { input: 'como escolher quantas camadas congelar', output: 'congele camadas iniciais, desconge finais conforme dados' },
  { input: 'o que é prompt engineering', output: 'arte de formular prompts para obter melhores respostas de modelos' },
  { input: 'como fazer bom prompt', output: 'seja específico, forneça exemplos, defina contexto e formato esperado' },
  { input: 'o que é few shot learning', output: 'aprender com poucos exemplos fornecidos no prompt' },
  { input: 'qual vantagem de few shot', output: 'rápido, sem treino, adaptável para novas tarefas' },
  { input: 'o que é zero shot learning', output: 'fazer tarefa sem exemplos, apenas instruções no prompt' },
  { input: 'quando usar zero shot', output: 'quando modelo é grande e bem treinado, tarefas genéricas' },
  { input: 'como criar dataset de qualidade', output: 'defina critério claro, use anotadores múltiplos, valide concordância' },
  { input: 'o que é inter-annotator agreement', output: 'medida de concordância entre anotadores' },
  { input: 'qual nível de acordo é bom', output: 'kappa acima de 0.8 é excelente, 0.6-0.8 é aceitável' },
  { input: 'como lidar com dados desbalanceados', output: 'use weighted loss, oversampling, undersampling ou balanceamento' },
  { input: 'qual técnica para dados desbalanceados', output: 'weighted loss é geralmente mais eficiente' },
  { input: 'como avaliar modelo em produção', output: 'monitore métricas, detecte drift, colete feedback de usuários' },
  { input: 'o que é model drift', output: 'degradação de performance quando dados mudam em produção' },
  { input: 'como detectar drift', output: 'compare distribuição de dados novo com treino usando ks test' },
  { input: 'como retreinar modelo', output: 'recolha dados novo, combine com antigo e repita treino' },
  { input: 'com que frequência retreinar', output: 'quando performance cai ou dados mudam significativamente' },
  { input: 'como fazer a/b testing com modelos', output: 'divida tráfego, compare métricas, valide significância' },
  { input: 'como implementar em produção', output: 'containerize, use api, monitore, log e prepare rollback' },
  { input: 'qual framework usar', output: 'tensorflow, pytorch ou scikit-learn dependendo da complexidade' },
  { input: 'tensorflow vs pytorch', output: 'tensorflow é mais industrial, pytorch é mais flexível para pesquisa' },
  { input: 'como debugar modelo', output: 'visualize gradientes, check dados, teste componentes isoladamente' },
  { input: 'modelo não converge', output: 'reduz learning rate, normaliza dados, ajusta inicialização' },
  { input: 'modelo converge lentamente', output: 'aumenta learning rate, usa batch size maior ou adam' },
  { input: 'model overfits', output: 'adiciona regularização, dropout, mais dados ou parar antes' },
  { input: 'model underfits', output: 'aumenta complexidade, reduz regularização ou mais épocas' },
  { input: 'como usar gpu', output: 'mude dados e modelo para gpu com .to(device) ou .cuda()' },
  { input: 'gpu está lenta', output: 'check memória gpu, reduz batch size, profile código' },
  { input: 'erro de memoria gpu', output: 'reduz batch size, limpa cache, usa gradient accumulation' },
  { input: 'como usar multi-gpu', output: 'use distributed training com dataparallel ou ddp' },
  { input: 'como fazer inference rápido', output: 'quantize modelo, use pruning, batch inference, cache' },
  { input: 'o que é quantização', output: 'reduz precisão numérica para diminuir tamanho e latência' },
  { input: 'quanto quantizar ajuda', output: '4-10x mais rápido, 4x menor tamanho com perda mínima' },
  { input: 'o que é pruning', output: 'remove pesos não importantes para diminuir tamanho' },
  { input: 'quanto pruning ajuda', output: '50-90% redução com performance aceitável' },
  { input: 'o que é knowledge distillation', output: 'treinar modelo menor usando modelo maior como professor' },
  { input: 'qual vantagem de distillation', output: 'modelo menor com performance próxima ao grande' },
  { input: 'como usar modelo pré treinado', output: 'baixe from huggingface, carregue e fine tune se necessário' },
  { input: 'quais modelos pré treinados populares', output: 'bert, gpt, t5, xlnet, roberta, distilbert' },
  { input: 'qual modelo usar para português', output: 'bert-base-portuguese, gpt2-portuguese ou distilbert-pt' },
  { input: 'como fazer deployment', output: 'use fastapi, flask, docker e coloque em servidor' },
  { input: 'como monitorar modelo', output: 'log predictions, accuracy, latência e erro rates' },
  { input: 'como coletar feedback', output: 'peça ao usuário para validar, guarde dados para retreino' },
  { input: 'como versionar modelo', output: 'guarde checkpoints, git para código, s3 para artifacts' },
  { input: 'como comparar modelos', output: 'use mesmos dados teste, mesmas sementes aleatórias' },
  { input: 'como fazer hyperparameter tuning', output: 'use grid search, random search ou bayesian optimization' },
  { input: 'qual melhor, grid ou random search', output: 'random search é mais eficiente em altas dimensões' },
  { input: 'como usar bayesian optimization', output: 'use libraries como optuna ou ray tune' },
  { input: 'quanto tempo tuning leva', output: 'semanas ou meses dependendo da complexidade' },
  { input: 'como economizar tempo de tuning', output: 'use valores padrão, faça busca grossa antes fina' },
  { input: 'como documentar experimento', output: 'guarde código, dados, hiperparâmetros, métricas e logs' },
  { input: 'qual melhor prática ml', output: 'versionamento, validação cruzada, reproducibilidade e monitoramento' },
  { input: 'como fazer experimento reproducível', output: 'fixe seeds, documente tudo, use containers' },
  { input: 'por que reproducibilidade importa', output: 'permite validação científica e debug de problemas' },
  { input: 'como lidar com outliers', output: 'identifique com z-score, remova ou use robust loss' },
  { input: 'quando remover vs manter outliers', output: 'remova se erro de dados, mantenha se legítimos' },
  { input: 'como fazer validação cruzada', output: 'divida k-vezes, treine em k-1, teste em 1' },
  { input: 'qual k usar em k-fold', output: 'k=5 ou 10 é padrão, mais com datasets pequenos' },

  // ════════════════════════════════════════════════════════════════════════════
  // SEÇÃO 6: Troubleshooting e Boas Práticas (50 pares)
  // ════════════════════════════════════════════════════════════════════════════

  { input: 'modelo não melhora', output: 'check dados, aumenta diversidade, valida pipeline, usa regularização' },
  { input: 'erro de nan na loss', output: 'learning rate muito alto, gradient clipping, normaliza inputs' },
  { input: 'perda está nan', output: 'reduz learning rate, check dados para infinitos, normaliza weights' },
  { input: 'acurácia está baixa', output: 'mais dados, melhor features, arquitetura maior, treino mais' },
  { input: 'modelo demora muito', output: 'redus modelo, menos dados por epoch, usa gpu, paralelize' },
  { input: 'qual o checkpoint melhor', output: 'use aquele com melhor validação, não treino' },
  { input: 'como salvar checkpoint', output: 'guarde pesos, optimizer state e época' },
  { input: 'como carregar checkpoint', output: 'crie model, carregue state dict e coloque no modo eval' },
  { input: 'como fazer ensemble', output: 'treine múltiplos modelos, average ou vote predictions' },
  { input: 'ensemble melhora quanto', output: '1-3% improvement dependendo da diversidade' },
  { input: 'como fazer stacking', output: 'treine meta-modelo em saídas de modelos base' },
  { input: 'qual diferença ensemble vs stacking', output: 'ensemble é mais simples, stacking aprende combinação' },
  { input: 'como lidar com imbalance severo', output: 'use weighted loss, focal loss, ou oversampling agressivo' },
  { input: 'o que é focal loss', output: 'loss que penaliza mais exemplos fáceis, focando nos difíceis' },
  { input: 'quando usar focal loss', output: 'dados muito desbalanceados, detecção de anomalias' },
  { input: 'como fazer classe balanceada', output: 'compute weight proporcional ao inverso da frequência' },
  { input: 'como verificar se treino tá ok', output: 'veja se loss diminui, acurácia aumenta, sem nan' },
  { input: 'treino oscila muito', output: 'reduz learning rate, aumenta batch size, usa warmup' },
  { input: 'o que é warmup', output: 'aumenta learning rate gradualmente nas primeiras épocas' },
  { input: 'qual duração warmup', output: 'típico 5-10% das épocas totais' },
  { input: 'como fazer warmup', output: 'use scheduler que aumenta lr gradualmente' },
  { input: 'modelo tá bom, agora o quê', output: 'teste em dados novo, coloque em produção, monitore' },
  { input: 'como avaliar antes de produção', output: 'use test set separado, faça análise de erros' },
  { input: 'análise de erro o quê', output: 'categorize erros, identifique padrões, defina melhorias' },
  { input: 'como comunicar resultados', output: 'use visualizações, métricas claras, compare baselines' },
  { input: 'qual baseline usar', output: 'modelo anterior, regra simples, ou modelo padrão' },
  { input: 'improvement deve ser quanto', output: 'mínimo 2-5% dependendo da tarefa e baseline' },
  { input: 'como fazer presentation', output: 'mostre problema, solução, resultados e impacto' },
  { input: 'quais métricas reportar', output: 'main metric, secondary metrics, confidence intervals' },
  { input: 'como reportar incerteza', output: 'use desvio padrão, confidence intervals ou bootstrap' },
  { input: 'como testar significância', output: 'use paired t-test ou permutation test' },
  { input: 'p-value o quê', output: 'probabilidade de resultado dado chance, < 0.05 é significante' },
  { input: 'como lidar com p-hacking', output: 'defina hypothesis antes, use bonferroni correction' },
  { input: 'qual correção usar', output: 'bonferroni é conservadora, benjamini-hochberg é menos' },
  { input: 'como documentar código', output: 'use docstrings, type hints, exemplos, README detalhado' },
  { input: 'qual padrão de código', output: 'pep8 para python, limpo, testável, comentado' },
  { input: 'como fazer unit tests', output: 'teste componentes isoladamente com pytest' },
  { input: 'como fazer integration tests', output: 'teste pipeline inteiro com dados reais' },
  { input: 'quando testar o quê', output: 'teste crítico, trate edge cases, coverage acima 80%' },
  { input: 'como debugar em produção', output: 'log detalhado, reproduz localmente, usa profilers' },
  { input: 'qual ferramenta debugar', output: 'pdb, ipdb localmente, sentry ou datadog em produção' },
  { input: 'como fazer profiling', output: 'use cprofile ou line_profiler para encontrar gargalos' },
  { input: 'modelo tá lento', output: 'profile, identifique gargalo, otimize ou paralelize' },
  { input: 'como otimizar código', output: 'use bibliotecas rápidas, vectorize, jit compile' },
  { input: 'numc vs numpy', output: 'numba jit compila, numpy é interpretado, numba é 100x mais rápido' },
  { input: 'como paralelizar', output: 'use multiprocessing, ray ou joblib' },
  { input: 'quando paralelizar', output: 'quando tarefa é cpu-bound e parallelizável' },
  { input: 'como testar escalabilidade', output: 'teste com dados crescentes, observe performance' },

];

// ---------------------------------------------------------------------------
// Função principal de bootstrap
// ---------------------------------------------------------------------------

/**
 * Verifica se o sistema está vazio e, se sim, insere dados iniciais.
 * Idempotente: verifica sentinela antes de agir.
 *
 * @param {FirebaseFirestore.Firestore} db
 * @param {Vocabulary} vocab
 * @returns {boolean} true se bootstrap foi executado
 */
async function runBootstrap(db, vocab) {
  // ── Verificar sentinela ────────────────────────────────────────────────
  const sentinel = await db.collection('akie_worker_status').doc('bootstrap').get();
  if (sentinel.exists && sentinel.data().completed === true) {
    console.log('[BOOTSTRAP] Já executado anteriormente. Pulando.');
    return false;
  }

  // ── Verificar se há dados ──────────────────────────────────────────────
  const [episodesSnap, graphSnap] = await Promise.all([
    db.collection('nexus_episodes').limit(1).get(),
    db.collection('nexus_graph').limit(1).get(),
  ]);

  const episodesEmpty = episodesSnap.empty;
  const graphEmpty    = graphSnap.empty;

  if (!episodesEmpty && !graphEmpty) {
    console.log('[BOOTSTRAP] Dados existentes detectados. Marcando como completo.');
    await _markBootstrapComplete(db, 0, 0, vocab.size);
    return false;
  }

  console.log('[BOOTSTRAP] iniciado');
  console.log(`[BOOTSTRAP] Estado: episodes=${episodesEmpty ? 'vazio' : 'ok'}, graph=${graphEmpty ? 'vazio' : 'ok'}`);

  let insertedEpisodes = 0;
  let insertedNodes    = 0;

  // ── Inserir episódios em batches de 400 ops (limite Firestore) ────────
  if (episodesEmpty) {
    const BATCH_SIZE = 400;
    for (let i = 0; i < BOOTSTRAP_EPISODES.length; i += BATCH_SIZE) {
      const batch   = db.batch();
      const chunk   = BOOTSTRAP_EPISODES.slice(i, i + BATCH_SIZE);
      for (const ep of chunk) {
        const ref = db.collection('nexus_episodes').doc();
        batch.set(ref, {
          input:      ep.input,
          output:     ep.output,
          layer:      'bootstrap',
          feedback:   'positive',   // peso 2x no treinamento
          processed:  false,
          created_at: new Date().toISOString(),
        });
      }
      await batch.commit();
      insertedEpisodes += chunk.length;
    }
    console.log(`[BOOTSTRAP] dataset inserido: ${insertedEpisodes} exemplos`);
  }

  // ── Inserir nós no grafo ──────────────────────────────────────────────
  if (graphEmpty) {
    const batch = db.batch();
    for (const node of BOOTSTRAP_GRAPH_NODES) {
      const ref = db.collection('nexus_graph').doc(node.id);
      batch.set(ref, {
        ...node,
        created_at:  new Date().toISOString(),
        usage_count: 0,
      });
    }
    await batch.commit();
    insertedNodes = BOOTSTRAP_GRAPH_NODES.length;
    console.log(`[BOOTSTRAP] grafo inicial: ${insertedNodes} nós inseridos`);
  }

  // ── Expandir vocabulário com palavras comuns ──────────────────────────
  const before = vocab.size;

  // Palavras comuns
  vocab.addTokens(COMMON_WORDS_PT);

  // Tokens dos próprios episódios
  for (const ep of BOOTSTRAP_EPISODES) {
    const inputTokens  = ep.input.toLowerCase().split(' ').filter(t => t.length > 0);
    const outputTokens = ep.output.toLowerCase().split(' ').filter(t => t.length > 0);
    vocab.addTokens(inputTokens);
    vocab.addTokens(outputTokens);
  }

  // Tokens dos nós do grafo
  for (const node of BOOTSTRAP_GRAPH_NODES) {
    vocab.addTokens(node.label.split(' '));
    vocab.addTokens((node.verbs   || []));
    vocab.addTokens((node.contexts || []));
    for (const rel of (node.relations || [])) {
      vocab.addTokens(rel.target.replace(/_/g, ' ').split(' '));
      vocab.addTokens(rel.type.replace(/_/g, ' ').split(' '));
    }
  }

  const after = vocab.size;
  console.log(`[BOOTSTRAP] vocabulário inicial: ${after} tokens (era ${before})`);

  // ── Marcar sentinela ──────────────────────────────────────────────────
  await _markBootstrapComplete(db, insertedEpisodes, insertedNodes, after);

  return true;
}

async function _markBootstrapComplete(db, episodes, nodes, vocabSize) {
  await db.collection('akie_worker_status').doc('bootstrap').set({
    completed:        true,
    completed_at:     new Date().toISOString(),
    inserted_episodes: episodes,
    inserted_nodes:   nodes,
    vocab_size:       vocabSize,
  });
}

module.exports = { runBootstrap, BOOTSTRAP_EPISODES, BOOTSTRAP_GRAPH_NODES, COMMON_WORDS_PT };
