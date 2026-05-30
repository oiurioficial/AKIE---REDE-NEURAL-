/**
 * monitor.js — AKIE Training Monitor API
 * Endpoints: GET /logs  GET /status  POST /control
 * Porta: 4242
 * PM2: pm2 start monitor.js --name akie-monitor
 */

const http     = require('http');
const fs       = require('fs');
const { exec } = require('child_process');

const PORT     = 4242;
const LOG_FILE = '/opt/akie/logs/pretrain.log';
const WORKER_LOG = '/opt/akie/logs/out.log';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

function parseLogLine(line) {
  // Formato: [PRETRAIN] Epoch 1 | Step 150 | loss=4.8644 acc=18.8%
  const m = line.match(/Epoch\s+(\d+)\s*\|\s*Step\s+(\d+)\s*\|\s*loss=([\d.]+)\s+acc=([\d.]+)%/);
  if (!m) return null;
  return {
    epoch:    parseInt(m[1]),
    step:     parseInt(m[2]),
    loss:     parseFloat(m[3]),
    accuracy: parseFloat(m[4]),
    raw:      line.trim(),
  };
}

function readLastLines(filePath, n = 100) {
  try {
    if (!fs.existsSync(filePath)) return [];
    const content = fs.readFileSync(filePath, 'utf8');
    const lines   = content.split('\n').filter(l => l.trim());
    return lines.slice(-n);
  } catch { return []; }
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

async function handleLogs(res) {
  const lines  = readLastLines(LOG_FILE, 200);
  const parsed = [];
  const raw    = [];

  for (const line of lines) {
    raw.push(line.trim());
    const p = parseLogLine(line);
    if (p) parsed.push(p);
  }

  // Latest metrics
  const latest = parsed.length ? parsed[parsed.length - 1] : null;

  // Detect completion
  const completed = raw.some(l => l.includes('PRÉ-TREINAMENTO CONCLUÍDO') || l.includes('CONCLUÍDO'));
  const totalEpochs = (() => {
    for (const l of raw) {
      const m = l.match(/Epochs:\s*(\d+)/);
      if (m) return parseInt(m[1]);
    }
    return 10;
  })();

  json(res, {
    ok:          true,
    latest,
    parsed:      parsed.slice(-50),
    raw:         raw.slice(-50),
    completed,
    totalEpochs,
    logExists:   fs.existsSync(LOG_FILE),
  });
}

async function handleStatus(res) {
  // Check pretrain process
  exec('ps aux | grep _akie_pretrain | grep -v grep', (err, stdout) => {
    const pretrainRunning = stdout.trim().length > 0;

    // Check PM2 worker
    exec('pm2 jlist 2>/dev/null', (err2, stdout2) => {
      let workerStatus = 'unknown';
      try {
        const list = JSON.parse(stdout2 || '[]');
        const worker = list.find(p => p.name === 'akie-worker');
        if (worker) workerStatus = worker.pm2_env?.status || 'unknown';
      } catch { /* ignore */ }

      // Memory
      exec('free -m', (err3, stdout3) => {
        let memTotal = 0, memUsed = 0, memFree = 0;
        try {
          const line = stdout3.split('\n')[1];
          const parts = line.trim().split(/\s+/);
          memTotal = parseInt(parts[1]);
          memUsed  = parseInt(parts[2]);
          memFree  = parseInt(parts[3]);
        } catch { /* ignore */ }

        // CPU load
        exec('cat /proc/loadavg', (err4, stdout4) => {
          const load = stdout4?.trim().split(' ')[0] || '0';

          json(res, {
            ok:              true,
            pretrain:        { running: pretrainRunning },
            worker:          { status: workerStatus },
            memory:          { total: memTotal, used: memUsed, free: memFree },
            load:            parseFloat(load),
            uptime:          process.uptime(),
            timestamp:       new Date().toISOString(),
          });
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

      if (action === 'stop-pretrain') {
        exec('pkill -f _akie_pretrain', (err) => {
          json(res, { ok: true, action, done: !err });
        });
      } else if (action === 'restart-worker') {
        exec('pm2 restart akie-worker', (err, stdout) => {
          json(res, { ok: true, action, done: !err, output: stdout?.trim() });
        });
      } else if (action === 'stop-worker') {
        exec('pm2 stop akie-worker', (err, stdout) => {
          json(res, { ok: true, action, done: !err, output: stdout?.trim() });
        });
      } else {
        json(res, { ok: false, error: 'Ação desconhecida' }, 400);
      }
    } catch (e) {
      json(res, { ok: false, error: e.message }, 500);
    }
  });
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

const server = http.createServer((req, res) => {
  if (req.method === 'OPTIONS') { cors(res); res.writeHead(204); res.end(); return; }

  const url = req.url?.split('?')[0];

  if (req.method === 'GET'  && url === '/logs')    return handleLogs(res);
  if (req.method === 'GET'  && url === '/status')  return handleStatus(res);
  if (req.method === 'POST' && url === '/control') return handleControl(req, res);
  if (req.method === 'GET'  && url === '/health')  return json(res, { ok: true, service: 'akie-monitor' });

  cors(res); res.writeHead(404); res.end('Not found');
});

server.listen(PORT, () => {
  console.log(`[AKIE-MONITOR] API rodando na porta ${PORT}`);
});
