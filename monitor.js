/**
 * monitor.js v2.0 — AKIE Training Monitor API
 * [FIX] Consome /status do worker diretamente em vez de ler pretrain.log
 * [FIX] Expõe histórico de loss, métricas reais e status do processo
 * Endpoints: GET /logs  GET /status  POST /control
 * Porta: 4242
 * PM2: pm2 start monitor.js --name akie-monitor
 */

const http     = require('http');
const fs       = require('fs');
const { exec } = require('child_process');

const PORT        = 4242;
// URL do worker — ajuste a porta se necessário
const WORKER_URL  = process.env.WORKER_URL || 'http://localhost:3000';

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

function json(res, data, code = 200) {
  cors(res);
  res.writeHead(code, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(data));
}

// Faz GET no worker e retorna o JSON parseado, ou null se falhar
function fetchWorker(path = '/status') {
  return new Promise((resolve) => {
    const url = new URL(WORKER_URL + path);
    const mod = url.protocol === 'https:' ? require('https') : require('http');

    const req = mod.get({
      hostname: url.hostname,
      port:     url.port || (url.protocol === 'https:' ? 443 : 80),
      path:     url.pathname,
      timeout:  5000,
    }, (res) => {
      let body = '';
      res.on('data', c => { body += c; });
      res.on('end', () => {
        try { resolve(JSON.parse(body)); }
        catch { resolve(null); }
      });
    });

    req.on('error', () => resolve(null));
    req.on('timeout', () => { req.destroy(); resolve(null); });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Routes
// ─────────────────────────────────────────────────────────────────────────────

async function handleLogs(res) {
  const workerData = await fetchWorker('/status');

  if (!workerData || !workerData.ok) {
    json(res, {
      ok:          false,
      error:       'Worker indisponível',
      workerUrl:   WORKER_URL,
      latest:      null,
      parsed:      [],
      raw:         ['Worker não respondeu. Verifique se está rodando em ' + WORKER_URL],
      completed:   false,
      totalEpochs: 0,
    });
    return;
  }

  // Monta "raw" a partir do histórico de loss para compatibilidade com o painel
  const lossHistory = workerData.loss_history || [];
  const raw = lossHistory.map(entry => {
    const acc = entry.accuracy != null ? entry.accuracy.toFixed(1) : '0.0';
    return `[WORKER] Cycle ${entry.cycle} | Mode: ${entry.mode} | loss=${entry.loss.toFixed(4)} acc=${acc}%`;
  });

  if (raw.length === 0) {
    raw.push('[WORKER] Aguardando primeiro ciclo de treino...');
    raw.push(`[WORKER] Ciclos concluídos: ${workerData.cycle || 0}`);
    raw.push(`[WORKER] Status: ${workerData.model?.ready ? 'Modelo pronto' : 'Inicializando...'}`);
  }

  // Adiciona linha de status atual
  const modeHistory = workerData.mode_history || [];
  if (modeHistory.length > 0) {
    raw.push(`[WORKER] Último modo: ${modeHistory[modeHistory.length - 1]}`);
  }

  // latest para o painel (adaptado ao formato esperado)
  const latest = workerData.latest ? {
    epoch:    workerData.cycle || 0,
    step:     workerData.model?.trainSteps || 0,
    loss:     workerData.latest.loss || 0,
    accuracy: workerData.latest.accuracy || 0,
    raw:      raw[raw.length - 1] || '',
  } : null;

  // parsed para o gráfico
  const parsed = lossHistory.map((entry, i) => ({
    epoch:    entry.cycle,
    step:     i,
    loss:     entry.loss,
    accuracy: entry.accuracy || 0,
    raw:      '',
  }));

  json(res, {
    ok:          true,
    latest,
    parsed:      parsed.slice(-50),
    raw:         raw.slice(-50),
    completed:   false,
    totalEpochs: workerData.cycle || 0,
    workerOnline: true,
  });
}

async function handleStatus(res) {
  // Checar processo worker pelo PM2
  exec('pm2 jlist 2>/dev/null', async (err, stdout) => {
    let workerStatus  = 'unknown';
    let monitorStatus = 'online';

    try {
      const list   = JSON.parse(stdout || '[]');
      const worker = list.find(p => p.name === 'akie-worker');
      if (worker) workerStatus = worker.pm2_env?.status || 'unknown';
      const monitor = list.find(p => p.name === 'akie-monitor');
      if (monitor) monitorStatus = monitor.pm2_env?.status || 'unknown';
    } catch { /* ignora */ }

    // Dados de memória
    exec('free -m', async (err2, stdout2) => {
      let memTotal = 0, memUsed = 0;
      try {
        const line  = stdout2.split('\n')[1];
        const parts = line.trim().split(/\s+/);
        memTotal = parseInt(parts[1]);
        memUsed  = parseInt(parts[2]);
      } catch { /* ignora */ }

      exec('cat /proc/loadavg', async (err3, stdout3) => {
        const load = parseFloat(stdout3?.trim().split(' ')[0] || '0');

        // Dados em tempo real do worker
        const workerData = await fetchWorker('/status');

        json(res, {
          ok:      true,
          pretrain: { running: false }, // pré-treino não existe mais — removido
          worker:  {
            status:      workerStatus,
            cycle:       workerData?.cycle || 0,
            train_steps: workerData?.model?.trainSteps || 0,
            ready:       workerData?.model?.ready || false,
            metrics:     workerData?.metrics || {},
            mode_history: workerData?.mode_history || [],
          },
          memory:    { total: memTotal, used: memUsed, free: memTotal - memUsed },
          load:      load,
          uptime:    process.uptime(),
          timestamp: new Date().toISOString(),
          workerUrl: WORKER_URL,
        });
      });
    });
  });
}

async function handleControl(req, res) {
  let body = '';
  req.on('data', c => { body += c; });
  req.on('end', () => {
    try {
      const { action } = JSON.parse(body || '{}');

      if (action === 'restart-worker') {
        exec('pm2 restart akie-worker', (err, stdout) => {
          json(res, { ok: true, action, done: !err, output: stdout?.trim() });
        });
      } else if (action === 'stop-worker') {
        exec('pm2 stop akie-worker', (err, stdout) => {
          json(res, { ok: true, action, done: !err, output: stdout?.trim() });
        });
      } else if (action === 'stop-pretrain') {
        // Mantido por compatibilidade, mas não há mais pretrain separado
        json(res, { ok: true, action, done: true, output: 'Nenhum processo de pré-treino ativo.' });
      } else {
        json(res, { ok: false, error: 'Ação desconhecida' }, 400);
      }
    } catch (e) {
      json(res, { ok: false, error: e.message }, 500);
    }
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Server
// ─────────────────────────────────────────────────────────────────────────────

const server = http.createServer((req, res) => {
  if (req.method === 'OPTIONS') { cors(res); res.writeHead(204); res.end(); return; }

  const url = req.url?.split('?')[0];

  if (req.method === 'GET'  && url === '/logs')    return handleLogs(res);
  if (req.method === 'GET'  && url === '/status')  return handleStatus(res);
  if (req.method === 'POST' && url === '/control') return handleControl(req, res);
  if (req.method === 'GET'  && url === '/health')  return json(res, { ok: true, service: 'akie-monitor', workerUrl: WORKER_URL });

  cors(res); res.writeHead(404); res.end('Not found');
});

server.listen(PORT, () => {
  console.log(`[AKIE-MONITOR] API rodando na porta ${PORT}`);
  console.log(`[AKIE-MONITOR] Consumindo worker em: ${WORKER_URL}`);
});
