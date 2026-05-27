# Dockerfile — AKIE Worker
# Imagem base com suporte a binários nativos do TF.js

FROM node:20-slim

# Dependências do sistema para TF.js native bindings
RUN apt-get update && apt-get install -y \
    python3 \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependências primeiro (cache de layer)
COPY package.json ./
RUN npm install --omit=dev

# Copiar código
COPY . .

# Diretório do modelo (Railway Volume monta aqui)
RUN mkdir -p /data/akie_model

# Healthcheck simples via arquivo de status
HEALTHCHECK --interval=120s --timeout=10s --start-period=30s \
  CMD node -e "require('fs').statSync('/data/akie_model')" || exit 1

CMD ["node", "worker.js"]
