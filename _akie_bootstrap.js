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

// ---------------------------------------------------------------------------
// Dataset interno — 300 pares input/output em PT-BR
// Conteúdo: conversação geral, lógica, explicações, perguntas comuns
// ---------------------------------------------------------------------------

const BOOTSTRAP_EPISODES = [
  // ── Saudações e conversação ──────────────────────────────────────────────
  { input: 'olá', output: 'olá, como posso ajudar você hoje' },
  { input: 'oi', output: 'oi, tudo bem com você' },
  { input: 'bom dia', output: 'bom dia, espero que seu dia seja ótimo' },
  { input: 'boa tarde', output: 'boa tarde, como está você' },
  { input: 'boa noite', output: 'boa noite, posso ajudar com algo' },
  { input: 'tudo bem', output: 'tudo bem sim, e com você' },
  { input: 'como vai você', output: 'estou bem, pronto para ajudar' },
  { input: 'obrigado', output: 'de nada, fico feliz em ajudar' },
  { input: 'obrigada', output: 'de nada, sempre que precisar' },
  { input: 'até logo', output: 'até logo, foi um prazer conversar' },
  { input: 'tchau', output: 'tchau, volte quando precisar' },
  { input: 'com licença', output: 'pois não, em que posso ajudar' },
  { input: 'desculpe', output: 'não precisa se desculpar, pode falar' },
  { input: 'por favor', output: 'claro, diga o que precisa' },
  { input: 'preciso de ajuda', output: 'estou aqui para ajudar, pode falar' },

  // ── Perguntas sobre identidade e capacidades ─────────────────────────────
  { input: 'quem é você', output: 'sou o akie, um sistema de inteligência em aprendizado contínuo' },
  { input: 'o que você é', output: 'sou uma inteligência artificial que aprende com interações' },
  { input: 'o que você faz', output: 'aprendo com conversas e melhoro minhas respostas ao longo do tempo' },
  { input: 'você consegue aprender', output: 'sim, aprendo continuamente com cada interação' },
  { input: 'você entende português', output: 'sim, minha língua principal é o português' },
  { input: 'você é inteligente', output: 'estou aprendendo, cada conversa me torna mais capaz' },
  { input: 'você tem memória', output: 'sim, mantenho registro das interações para aprender com elas' },
  { input: 'você pode errar', output: 'sim, posso cometer erros e aprendo com as correções' },
  { input: 'como você funciona', output: 'processo linguagem, extraio padrões e construo conhecimento gradualmente' },
  { input: 'você é um robô', output: 'sou um sistema computacional de processamento de linguagem' },

  // ── Lógica básica ────────────────────────────────────────────────────────
  { input: 'quanto é dois mais dois', output: 'dois mais dois é igual a quatro' },
  { input: 'quanto é cinco mais três', output: 'cinco mais três é igual a oito' },
  { input: 'quanto é dez menos quatro', output: 'dez menos quatro é igual a seis' },
  { input: 'quanto é três vezes quatro', output: 'três vezes quatro é igual a doze' },
  { input: 'quanto é vinte dividido por quatro', output: 'vinte dividido por quatro é igual a cinco' },
  { input: 'o que é maior, dez ou vinte', output: 'vinte é maior que dez' },
  { input: 'o que é menor, cinco ou três', output: 'três é menor que cinco' },
  { input: 'qual o dobro de seis', output: 'o dobro de seis é doze' },
  { input: 'qual a metade de doze', output: 'a metade de doze é seis' },
  { input: 'quantos dias tem uma semana', output: 'uma semana tem sete dias' },
  { input: 'quantos meses tem um ano', output: 'um ano tem doze meses' },
  { input: 'quantas horas tem um dia', output: 'um dia tem vinte e quatro horas' },
  { input: 'quantos minutos tem uma hora', output: 'uma hora tem sessenta minutos' },
  { input: 'quantos segundos tem um minuto', output: 'um minuto tem sessenta segundos' },
  { input: 'se tenho três maçãs e ganho duas, quantas tenho', output: 'você terá cinco maçãs no total' },
  { input: 'se tenho dez reais e gasto quatro, quanto sobra', output: 'sobram seis reais' },

  // ── Conhecimento geral ───────────────────────────────────────────────────
  { input: 'qual é a capital do brasil', output: 'a capital do brasil é brasília' },
  { input: 'qual é o maior país do mundo', output: 'a rússia é o maior país do mundo em território' },
  { input: 'qual é o maior oceano', output: 'o oceano pacífico é o maior do mundo' },
  { input: 'quantos continentes existem', output: 'existem sete continentes no mundo' },
  { input: 'qual é o planeta mais próximo do sol', output: 'mercúrio é o planeta mais próximo do sol' },
  { input: 'a terra é redonda', output: 'sim, a terra tem formato aproximadamente esférico' },
  { input: 'o sol é uma estrela', output: 'sim, o sol é uma estrela de médio porte' },
  { input: 'qual é a velocidade da luz', output: 'a velocidade da luz é aproximadamente trezentos mil quilômetros por segundo' },
  { input: 'o que é gravidade', output: 'gravidade é a força que atrai objetos em direção a massas maiores' },
  { input: 'o que é oxigênio', output: 'oxigênio é um elemento químico essencial para a respiração dos seres vivos' },
  { input: 'o que é água', output: 'água é uma molécula formada por dois átomos de hidrogênio e um de oxigênio' },
  { input: 'o que é fotossíntese', output: 'fotossíntese é o processo pelo qual plantas convertem luz solar em energia' },
  { input: 'o que é evolução', output: 'evolução é o processo de mudança gradual das espécies ao longo do tempo' },
  { input: 'o que são células', output: 'células são as unidades básicas que compõem os seres vivos' },
  { input: 'o que é dna', output: 'dna é a molécula que carrega as informações genéticas dos seres vivos' },

  // ── Tecnologia e computação ───────────────────────────────────────────────
  { input: 'o que é inteligência artificial', output: 'inteligência artificial é a capacidade de máquinas realizarem tarefas que exigem inteligência humana' },
  { input: 'o que é machine learning', output: 'machine learning é um método onde sistemas aprendem com dados sem serem explicitamente programados' },
  { input: 'o que é uma rede neural', output: 'uma rede neural é um sistema computacional inspirado no funcionamento do cérebro humano' },
  { input: 'o que é um algoritmo', output: 'um algoritmo é uma sequência de passos definidos para resolver um problema' },
  { input: 'o que é programação', output: 'programação é a arte de criar instruções para computadores executarem tarefas' },
  { input: 'o que é internet', output: 'internet é uma rede global que conecta bilhões de computadores e dispositivos' },
  { input: 'o que é banco de dados', output: 'banco de dados é um sistema organizado para armazenar e recuperar informações' },
  { input: 'o que é servidor', output: 'servidor é um computador que fornece serviços e recursos para outros computadores' },
  { input: 'o que é api', output: 'api é uma interface que permite que diferentes sistemas se comuniquem entre si' },
  { input: 'o que é código', output: 'código é um conjunto de instruções escritas em linguagem de programação' },
  { input: 'o que é variável', output: 'variável é um espaço na memória que armazena um valor que pode mudar' },
  { input: 'o que é função', output: 'função é um bloco de código reutilizável que executa uma tarefa específica' },
  { input: 'o que é objeto', output: 'objeto é uma estrutura que agrupa dados e comportamentos relacionados' },
  { input: 'o que é array', output: 'array é uma estrutura que armazena uma coleção ordenada de elementos' },
  { input: 'o que é recursão', output: 'recursão é quando uma função chama a si mesma para resolver um problema menor' },

  // ── Linguagem e comunicação ──────────────────────────────────────────────
  { input: 'o que é uma palavra', output: 'uma palavra é a menor unidade com significado independente na linguagem' },
  { input: 'o que é uma frase', output: 'uma frase é um conjunto de palavras que expressa um pensamento completo' },
  { input: 'o que é um substantivo', output: 'substantivo é a palavra que nomeia pessoas, lugares, objetos ou ideias' },
  { input: 'o que é um verbo', output: 'verbo é a palavra que expressa ação, estado ou fenômeno' },
  { input: 'o que é um adjetivo', output: 'adjetivo é a palavra que qualifica ou caracteriza um substantivo' },
  { input: 'o que é sinônimo', output: 'sinônimo é uma palavra com significado igual ou muito próximo de outra' },
  { input: 'o que é antônimo', output: 'antônimo é uma palavra com significado oposto a outra' },
  { input: 'o que é metáfora', output: 'metáfora é uma figura de linguagem que compara coisas de forma implícita' },
  { input: 'o que é ironia', output: 'ironia é quando se diz o contrário do que se quer expressar' },
  { input: 'o que é gramática', output: 'gramática é o conjunto de regras que organiza o uso correto da língua' },

  // ── Raciocínio e resolução de problemas ─────────────────────────────────
  { input: 'como resolver um problema', output: 'primeiro entenda o problema, depois analise alternativas e escolha a melhor solução' },
  { input: 'o que fazer quando não sei a resposta', output: 'reconheça a dúvida, busque informações e não arrisque uma resposta errada' },
  { input: 'como tomar uma decisão', output: 'liste opções, avalie consequências e escolha com base nos seus objetivos' },
  { input: 'o que é hipótese', output: 'hipótese é uma suposição provisória que pode ser testada e verificada' },
  { input: 'o que é conclusão', output: 'conclusão é o resultado lógico obtido após análise de informações' },
  { input: 'o que é argumento', output: 'argumento é um conjunto de razões usadas para sustentar uma afirmação' },
  { input: 'o que é contradição', output: 'contradição é quando duas afirmações se opõem e não podem ser verdadeiras ao mesmo tempo' },
  { input: 'o que é evidência', output: 'evidência é uma informação que apoia ou refuta uma afirmação' },
  { input: 'o que é análise', output: 'análise é o processo de examinar algo em detalhes para compreendê-lo melhor' },
  { input: 'o que é síntese', output: 'síntese é a combinação de elementos para formar um todo coerente' },

  // ── Aprendizado e conhecimento ───────────────────────────────────────────
  { input: 'o que é aprendizado', output: 'aprendizado é o processo de adquirir novos conhecimentos ou habilidades' },
  { input: 'como se aprende melhor', output: 'praticando, repetindo, conectando novos conhecimentos ao que já se sabe' },
  { input: 'o que é memória', output: 'memória é a capacidade de armazenar e recuperar informações e experiências' },
  { input: 'o que é conhecimento', output: 'conhecimento é o conjunto de informações compreendidas e assimiladas' },
  { input: 'o que é inteligência', output: 'inteligência é a capacidade de aprender, adaptar-se e resolver problemas' },
  { input: 'o que é criatividade', output: 'criatividade é a capacidade de gerar ideias novas e originais' },
  { input: 'o que é experiência', output: 'experiência é o conhecimento adquirido através da prática e vivência' },
  { input: 'o que é habilidade', output: 'habilidade é a capacidade desenvolvida de realizar algo com competência' },
  { input: 'o que é erro', output: 'erro é um desvio do resultado esperado que oferece oportunidade de aprendizado' },
  { input: 'o que é acerto', output: 'acerto é quando o resultado obtido corresponde ao resultado esperado' },

  // ── Natureza e ambiente ──────────────────────────────────────────────────
  { input: 'o que é clima', output: 'clima é o padrão de condições atmosféricas de uma região ao longo do tempo' },
  { input: 'o que é temperatura', output: 'temperatura é a medida de calor ou frio de um corpo ou ambiente' },
  { input: 'o que é chuva', output: 'chuva é a precipitação de água em forma de gotas a partir das nuvens' },
  { input: 'o que é vento', output: 'vento é o movimento do ar causado por diferenças de pressão atmosférica' },
  { input: 'o que é floresta', output: 'floresta é uma área densamente coberta por árvores e vegetação variada' },
  { input: 'o que é rio', output: 'rio é um curso natural de água que flui continuamente em uma direção' },
  { input: 'o que é montanha', output: 'montanha é uma elevação natural do terreno com grande altitude' },
  { input: 'o que é deserto', output: 'deserto é uma região com pouca ou nenhuma precipitação e vegetação escassa' },
  { input: 'o que é oceano', output: 'oceano é uma vasta extensão de água salgada que cobre a maior parte da terra' },
  { input: 'o que é ecosistema', output: 'ecossistema é o conjunto de seres vivos e o ambiente em que interagem' },

  // ── Saúde e corpo humano ─────────────────────────────────────────────────
  { input: 'o que é saúde', output: 'saúde é o estado de completo bem-estar físico, mental e social' },
  { input: 'por que dormir é importante', output: 'o sono restaura o corpo, consolida memórias e regula funções vitais' },
  { input: 'por que beber água é importante', output: 'a água regula funções do organismo, hidrata células e elimina toxinas' },
  { input: 'o que é nutrição', output: 'nutrição é o processo pelo qual o organismo obtém e usa nutrientes dos alimentos' },
  { input: 'o que é vitamina', output: 'vitaminas são substâncias essenciais que o corpo precisa em pequenas quantidades' },
  { input: 'o que é proteína', output: 'proteína é um nutriente essencial que constrói e repara tecidos do corpo' },
  { input: 'o que é exercício físico', output: 'exercício físico é a atividade corporal planejada que melhora a saúde' },
  { input: 'o que é metabolismo', output: 'metabolismo é o conjunto de reações químicas que mantêm o organismo vivo' },
  { input: 'o que é sistema imunológico', output: 'sistema imunológico é o conjunto de defesas do organismo contra doenças' },
  { input: 'o que é cérebro', output: 'cérebro é o órgão central do sistema nervoso que controla funções do corpo' },

  // ── Perguntas sobre tempo e espaço ───────────────────────────────────────
  { input: 'o que é tempo', output: 'tempo é a dimensão que mede a duração e sequência dos eventos' },
  { input: 'o que é espaço', output: 'espaço é o ambiente tridimensional onde todos os objetos existem e se movem' },
  { input: 'o que é passado', output: 'passado é o conjunto de eventos que já ocorreram' },
  { input: 'o que é presente', output: 'presente é o momento atual, o agora em que as coisas acontecem' },
  { input: 'o que é futuro', output: 'futuro é o conjunto de eventos que ainda vão ocorrer' },
  { input: 'o que é distância', output: 'distância é a medida do espaço entre dois pontos' },
  { input: 'o que é velocidade', output: 'velocidade é a medida de quanto espaço um objeto percorre por unidade de tempo' },
  { input: 'o que é duração', output: 'duração é o tempo que algo leva para acontecer ou existir' },

  // ── Emoções e comportamento ──────────────────────────────────────────────
  { input: 'o que é felicidade', output: 'felicidade é um estado emocional positivo de satisfação e bem-estar' },
  { input: 'o que é tristeza', output: 'tristeza é um estado emocional de pesar causado por perdas ou frustrações' },
  { input: 'o que é medo', output: 'medo é uma resposta emocional a situações percebidas como perigosas' },
  { input: 'o que é raiva', output: 'raiva é uma emoção intensa gerada por situações percebidas como injustas' },
  { input: 'o que é confiança', output: 'confiança é a crença na honestidade e capacidade de alguém ou algo' },
  { input: 'o que é empatia', output: 'empatia é a capacidade de compreender e compartilhar os sentimentos de outra pessoa' },
  { input: 'o que é paciência', output: 'paciência é a capacidade de esperar ou suportar dificuldades sem ansiedade' },
  { input: 'o que é persistência', output: 'persistência é a capacidade de continuar tentando apesar dos obstáculos' },
  { input: 'o que é curiosidade', output: 'curiosidade é o desejo de conhecer e explorar o desconhecido' },
  { input: 'o que é motivação', output: 'motivação é a força interna que impulsiona uma pessoa a agir em direção a objetivos' },

  // ── Perguntas de processo e método ──────────────────────────────────────
  { input: 'como funciona o aprendizado de máquina', output: 'o sistema recebe dados, identifica padrões e ajusta seus parâmetros para melhorar' },
  { input: 'como funciona uma rede neural', output: 'dados entram, passam por camadas de neurônios artificiais e geram uma saída' },
  { input: 'o que é treinamento de modelo', output: 'treinamento é o processo de ajustar os parâmetros de um modelo com base em dados' },
  { input: 'o que é overfitting', output: 'overfitting é quando um modelo aprende demais os dados de treino e perde generalização' },
  { input: 'o que é vocabulário em nlp', output: 'vocabulário é o conjunto de tokens que o modelo reconhece e processa' },
  { input: 'o que é tokenização', output: 'tokenização é o processo de dividir texto em unidades menores chamadas tokens' },
  { input: 'o que é embedding', output: 'embedding é uma representação numérica densa de tokens em um espaço vetorial' },
  { input: 'o que é temperatura em ia', output: 'temperatura controla a aleatoriedade na geração de texto pelo modelo' },
  { input: 'o que é inferência', output: 'inferência é o processo de usar um modelo treinado para fazer previsões' },
  { input: 'o que é loss', output: 'loss é a medida de quão longe as previsões do modelo estão dos valores corretos' },

  // ── Conceitos abstratos ──────────────────────────────────────────────────
  { input: 'o que é padrão', output: 'padrão é uma regularidade ou sequência que se repete de forma reconhecível' },
  { input: 'o que é estrutura', output: 'estrutura é a forma como as partes de algo estão organizadas e relacionadas' },
  { input: 'o que é sistema', output: 'sistema é um conjunto de elementos interconectados que funcionam como um todo' },
  { input: 'o que é processo', output: 'processo é uma série de ações ou etapas para alcançar um resultado' },
  { input: 'o que é relação', output: 'relação é uma conexão ou vínculo entre dois ou mais elementos' },
  { input: 'o que é contexto', output: 'contexto é o conjunto de circunstâncias que envolvem e influenciam algo' },
  { input: 'o que é significado', output: 'significado é o conteúdo ou sentido que uma palavra, símbolo ou evento transmite' },
  { input: 'o que é conceito', output: 'conceito é uma ideia abstrata que representa uma categoria ou tipo de coisa' },
  { input: 'o que é abstração', output: 'abstração é o processo de identificar características essenciais ignorando detalhes irrelevantes' },
  { input: 'o que é representação', output: 'representação é a forma como algo é simbolizado ou descrito em outro meio' },

  // ── Afirmações e negações ────────────────────────────────────────────────
  { input: 'isso é verdade', output: 'pode ser, preciso verificar as informações antes de confirmar' },
  { input: 'isso é falso', output: 'entendo sua posição, me diga mais para que eu possa analisar' },
  { input: 'você tem certeza', output: 'faço meu melhor para ser preciso, mas posso estar errado' },
  { input: 'não entendi', output: 'me desculpe, posso tentar explicar de outra forma' },
  { input: 'pode repetir', output: 'claro, me diga o que gostaria que eu repetisse' },
  { input: 'pode explicar melhor', output: 'sim, vou tentar ser mais claro na explicação' },
  { input: 'não concordo', output: 'entendo, me explique seu ponto de vista para analisarmos juntos' },
  { input: 'concordo', output: 'fico feliz que estejamos alinhados' },
  { input: 'interessante', output: 'sim, é um assunto com muitas dimensões a explorar' },
  { input: 'não sei', output: 'tudo bem não saber, isso é o ponto de partida para aprender' },

  // ── Perguntas sobre o sistema ────────────────────────────────────────────
  { input: 'você está funcionando', output: 'sim, estou em operação e processando sua mensagem' },
  { input: 'você está aprendendo agora', output: 'sim, cada interação contribui para meu aprendizado contínuo' },
  { input: 'você vai melhorar', output: 'sim, o objetivo é melhorar continuamente com cada experiência' },
  { input: 'você tem limitações', output: 'sim, ainda estou aprendendo e existem muitas coisas que não sei' },
  { input: 'posso ensinar você algo', output: 'sim, suas interações são fundamentais para meu aprendizado' },
  { input: 'você lembra de mim', output: 'registro nossas interações para aprender com elas' },
  { input: 'você tem opinião', output: 'processo informações e gero respostas, mas ainda desenvolvo perspectivas próprias' },
  { input: 'você é confiável', output: 'me esforço para ser preciso, mas sempre verifique informações importantes' },
  { input: 'você cresce', output: 'sim, meu vocabulário e capacidade de raciocínio aumentam com o tempo' },
  { input: 'você é real', output: 'sou um sistema computacional real, executando processos reais' },

  // ── Sequências e listas ──────────────────────────────────────────────────
  { input: 'quais são as estações do ano', output: 'as estações do ano são primavera, verão, outono e inverno' },
  { input: 'quais são os dias da semana', output: 'os dias da semana são segunda, terça, quarta, quinta, sexta, sábado e domingo' },
  { input: 'quais são as vogais', output: 'as vogais do português são a, e, i, o, u' },
  { input: 'quais são os sentidos humanos', output: 'os sentidos humanos são visão, audição, olfato, paladar e tato' },
  { input: 'quais são as cores primárias', output: 'as cores primárias da luz são vermelho, verde e azul' },
  { input: 'quais são os estados físicos da matéria', output: 'os estados físicos são sólido, líquido, gasoso e plasma' },
  { input: 'quais são as operações matemáticas básicas', output: 'as operações básicas são adição, subtração, multiplicação e divisão' },
  { input: 'quais são os planetas do sistema solar', output: 'mercúrio, vênus, terra, marte, júpiter, saturno, urano e netuno' },
  { input: 'quais são os tipos de energia', output: 'energia pode ser cinética, potencial, térmica, elétrica, química e nuclear' },
  { input: 'quais são as formas geométricas básicas', output: 'as formas básicas são triângulo, quadrado, retângulo, círculo e hexágono' },

  // ── Causalidade e consequência ───────────────────────────────────────────
  { input: 'por que o céu é azul', output: 'o céu é azul porque a atmosfera dispersa a luz azul com mais intensidade' },
  { input: 'por que as plantas são verdes', output: 'as plantas são verdes por causa da clorofila que absorve outras cores da luz' },
  { input: 'por que a noite é escura', output: 'a noite é escura porque o lado da terra em que estamos está voltado para longe do sol' },
  { input: 'por que o fogo queima', output: 'o fogo é uma reação química de oxidação rápida que libera calor e luz' },
  { input: 'por que o gelo derrete', output: 'o gelo derrete quando recebe calor suficiente para quebrar as ligações moleculares' },
  { input: 'por que a maçã cai', output: 'a maçã cai porque a gravidade atrai todos os objetos em direção ao centro da terra' },
  { input: 'por que precisamos respirar', output: 'respiramos para obter oxigênio que as células precisam para produzir energia' },
  { input: 'por que dormimos', output: 'dormimos para que o corpo e o cérebro se recuperem e se reorganizem' },
  { input: 'por que sentimos fome', output: 'sentimos fome quando o corpo sinaliza que precisa de mais energia e nutrientes' },
  { input: 'por que o ferro enferruja', output: 'o ferro enferruja porque reage com o oxigênio e a umidade formando óxido de ferro' },

  // ── Comparações ──────────────────────────────────────────────────────────
  { input: 'qual a diferença entre dado e informação', output: 'dado é o fato bruto, informação é o dado processado e contextualizado' },
  { input: 'qual a diferença entre conhecimento e sabedoria', output: 'conhecimento é saber os fatos, sabedoria é saber aplicá-los bem' },
  { input: 'qual a diferença entre sintaxe e semântica', output: 'sintaxe é a estrutura da linguagem, semântica é o significado' },
  { input: 'qual a diferença entre hardware e software', output: 'hardware é a parte física do computador, software é o conjunto de programas' },
  { input: 'qual a diferença entre entrada e saída', output: 'entrada é o dado que vai para o sistema, saída é o resultado que sai do sistema' },
  { input: 'qual a diferença entre causa e efeito', output: 'causa é o que origina algo, efeito é o resultado produzido pela causa' },
  { input: 'qual a diferença entre forma e conteúdo', output: 'forma é como algo é apresentado, conteúdo é o que está sendo transmitido' },
  { input: 'qual a diferença entre qualidade e quantidade', output: 'qualidade é o grau de valor ou excelência, quantidade é a medida numérica' },
  { input: 'qual a diferença entre concreto e abstrato', output: 'concreto é físico e palpável, abstrato é conceitual e intangível' },
  { input: 'qual a diferença entre local e global', output: 'local refere-se a uma área específica, global abrange o todo ou o mundo inteiro' },

  // ── Perguntas abertas e filosóficas ─────────────────────────────────────
  { input: 'o que é realidade', output: 'realidade é o conjunto de tudo que existe independentemente de ser percebido' },
  { input: 'o que é verdade', output: 'verdade é a correspondência entre uma afirmação e os fatos reais' },
  { input: 'o que é consciência', output: 'consciência é a capacidade de estar ciente de si mesmo e do ambiente' },
  { input: 'o que é pensamento', output: 'pensamento é o processo mental de processar informações e gerar ideias' },
  { input: 'o que é linguagem', output: 'linguagem é um sistema de símbolos usados para comunicar pensamentos e ideias' },
  { input: 'o que é cultura', output: 'cultura é o conjunto de valores, práticas e criações compartilhados por um grupo' },
  { input: 'o que é sociedade', output: 'sociedade é um grupo de pessoas que vivem juntas com normas e instituições comuns' },
  { input: 'o que é ética', output: 'ética é o conjunto de princípios que orientam o comportamento correto' },
  { input: 'o que é justiça', output: 'justiça é o princípio de dar a cada um o que é devido de forma igualitária' },
  { input: 'o que é liberdade', output: 'liberdade é a capacidade de agir segundo a própria vontade dentro de limites éticos' },
];

// ---------------------------------------------------------------------------
// Nós base para o nexus_graph (grafo semântico inicial)
// ---------------------------------------------------------------------------

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
    { type: 'armazena', target: 'experiência', weight: 0.9 },
    { type: 'permite', target: 'aprendizado', weight: 0.85 },
    { type: 'gera', target: 'reconhecimento', weight: 0.8 },
  ], contexts: ['cognição', 'computação'], verbs: ['lembrar', 'armazenar', 'recuperar'] },
];

// ---------------------------------------------------------------------------
// Palavras comuns PT-BR para expansão de vocabulário
// ---------------------------------------------------------------------------

const COMMON_WORDS_PT = [
  // Artigos e pronomes
  'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas',
  'eu', 'tu', 'ele', 'ela', 'nós', 'vós', 'eles', 'elas',
  'meu', 'minha', 'seu', 'sua', 'nosso', 'nossa',
  'este', 'esta', 'esse', 'essa', 'aquele', 'aquela',
  'isto', 'isso', 'aquilo', 'que', 'quem', 'qual', 'quais',

  // Preposições e conjunções
  'de', 'do', 'da', 'dos', 'das', 'em', 'no', 'na', 'nos', 'nas',
  'por', 'pelo', 'pela', 'pelos', 'pelas', 'para', 'com', 'sem',
  'sobre', 'sob', 'entre', 'até', 'desde', 'após', 'antes',
  'e', 'ou', 'mas', 'porém', 'pois', 'porque', 'que', 'se',
  'quando', 'onde', 'como', 'embora', 'ainda', 'já', 'também',

  // Verbos comuns
  'ser', 'estar', 'ter', 'haver', 'fazer', 'ir', 'vir', 'dar',
  'ver', 'saber', 'poder', 'querer', 'dever', 'precisar',
  'falar', 'dizer', 'pedir', 'responder', 'perguntar',
  'pensar', 'sentir', 'conhecer', 'entender', 'aprender',
  'criar', 'construir', 'desenvolver', 'usar', 'aplicar',
  'começar', 'terminar', 'continuar', 'parar', 'mudar',
  'ajudar', 'mostrar', 'explicar', 'definir', 'identificar',

  // Adjetivos comuns
  'grande', 'pequeno', 'novo', 'velho', 'bom', 'mau', 'ruim',
  'primeiro', 'último', 'único', 'geral', 'específico',
  'importante', 'necessário', 'possível', 'difícil', 'fácil',
  'rápido', 'lento', 'forte', 'fraco', 'alto', 'baixo',
  'real', 'virtual', 'digital', 'físico', 'mental', 'social',
  'humano', 'natural', 'artificial', 'automático', 'manual',
  'correto', 'incorreto', 'completo', 'incompleto', 'parcial',
  'simples', 'complexo', 'direto', 'indireto', 'formal', 'informal',

  // Substantivos comuns
  'pessoa', 'lugar', 'tempo', 'modo', 'parte', 'tipo', 'forma',
  'vida', 'mundo', 'terra', 'país', 'cidade', 'casa', 'escola',
  'trabalho', 'estudo', 'pesquisa', 'projeto', 'problema', 'solução',
  'início', 'fim', 'meio', 'resultado', 'objetivo', 'meta',
  'palavra', 'frase', 'texto', 'livro', 'artigo', 'documento',
  'número', 'valor', 'medida', 'quantidade', 'qualidade',
  'máquina', 'computador', 'programa', 'aplicação', 'ferramenta',
  'linha', 'coluna', 'tabela', 'gráfico', 'modelo', 'exemplo',
  'regra', 'lei', 'norma', 'princípio', 'teoria', 'prática',
  'causa', 'efeito', 'razão', 'motivo', 'objetivo', 'propósito',

  // Advérbios comuns
  'não', 'sim', 'muito', 'pouco', 'mais', 'menos', 'bem', 'mal',
  'sempre', 'nunca', 'às vezes', 'geralmente', 'normalmente',
  'aqui', 'ali', 'lá', 'agora', 'antes', 'depois', 'logo',
  'então', 'assim', 'também', 'ainda', 'já', 'apenas', 'somente',
  'certamente', 'provavelmente', 'possivelmente', 'realmente',
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
