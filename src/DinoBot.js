/**
 * DinoBot — main controller
 *
 * Reads game state from Runner.instance_, feeds it into the current neural
 * network, and sends actions directly via the game's own API.
 *
 * Depends on: NeuralNetwork, GeneticAlgorithm (load them first).
 */
window.DinoBot = (() => {
  // ── Constants ──────────────────────────────────────────────────────────────

  const DINO_X         = 75;
  const CANVAS_WIDTH   = 600;
  const CANVAS_HEIGHT  = 150;
  const MAX_SPEED      = 13;
  const TOPOLOGY       = [7, 8, 2];
  const POLL_MS        = 50;
  const CRASH_PAUSE_MS = 600;

  const INPUT_LABELS = ['dist  ', 'width ', 'height', 'type  ', 'obs-y ', 'speed ', 'dino-y'];

  // ── GA setup ───────────────────────────────────────────────────────────────

  const ga = new GeneticAlgorithm({
    populationSize: 15,
    mutationRate:   0.15,
    mutationScale:  0.4,
    eliteCount:     2,
    topology:       TOPOLOGY,
    autoSaveEvery:  5,
    maxCheckpoints: 10,
  });

  ga.loadFromStorage();

  // ── State ──────────────────────────────────────────────────────────────────

  let intervalId        = null;
  let overlay           = null;   // top-right: main HUD
  let cpPanel           = null;   // bottom-right: checkpoints
  let genPanel          = null;   // top-left: generation performance
  let cpExpanded        = true;
  let genExpanded       = true;
  let competitionActive = false;
  let competitionDone   = false;
  let competitionResults = null;  // sorted scores after competition completes
  let currentScore = 0;
  let lastAction   = 'waiting...';
  let jumpOut      = 0;
  let duckOut      = 0;
  let lastInputs   = [0, 0, 0, 0, 0, 0, 0];
  let crashedAt    = null;
  let forcedRunner = null;
  let runnerLogged = false;

  // ── Runner detection ───────────────────────────────────────────────────────

  function getRunner() {
    if (forcedRunner) return forcedRunner;
    if (window.Runner && typeof Runner.getInstance === 'function') {
      const r = Runner.getInstance();
      if (r) return r;
    }
    if (window.Runner && Runner.instance_) return Runner.instance_;
    if (window.Runner && typeof Runner.currentSpeed === 'number') return Runner;
    const canvas = document.querySelector('canvas');
    if (canvas && canvas.runner) return canvas.runner;
    return null;
  }

  // ── Score helper ───────────────────────────────────────────────────────────

  function getDisplayScore(runner) {
    if (runner.distanceMeter && runner.distanceMeter.digits && runner.distanceMeter.digits.length) {
      return parseInt(runner.distanceMeter.digits.join(''), 10) || 0;
    }
    return Math.ceil(runner.distanceRan * 0.025);
  }

  // ── Input vector ───────────────────────────────────────────────────────────

  function buildInputs(runner) {
    const obstacles = runner.horizon.obstacles;
    let dist = 1, width = 0, height = 0, type = 0, obsY = 0;
    if (obstacles && obstacles.length > 0) {
      const obs = obstacles.reduce((a, b) => (b.xPos < a.xPos ? b : a));
      dist   = Math.max(0, obs.xPos - DINO_X) / CANVAS_WIDTH;
      width  = (obs.width || 0) / CANVAS_WIDTH;
      height = (obs.typeConfig.height || 0) / CANVAS_HEIGHT;
      type   = obs.typeConfig.type === 'PTERODACTYL' ? 1 : 0;
      obsY   = obs.yPos / CANVAS_HEIGHT;
    }
    return [dist, width, height, type, obsY, runner.currentSpeed / MAX_SPEED, runner.tRex.yPos / CANVAS_HEIGHT];
  }

  // ── Game actions ───────────────────────────────────────────────────────────

  function gameJump(runner) {
    if (!runner.tRex.jumping) { runner.tRex.setDuck(false); runner.tRex.startJump(runner.currentSpeed); }
  }
  function gameDuck(runner, enable) { runner.tRex.setDuck(enable); }
  function gameRestart(runner) {
    if (typeof runner.restart === 'function') runner.restart();
    else document.dispatchEvent(new KeyboardEvent('keydown', { keyCode: 32, which: 32, bubbles: true }));
  }

  let lastStartAttempt = 0;
  function gameStart() {
    const now = Date.now();
    if (now - lastStartAttempt < 1000) return;
    lastStartAttempt = now;
    document.dispatchEvent(new KeyboardEvent('keydown', { keyCode: 32, which: 32, bubbles: true }));
  }

  // ── Main loop ──────────────────────────────────────────────────────────────

  function tick() {
    const runner = getRunner();

    if (!runner) {
      if (!overlay) return;
      const hasClass = typeof Runner !== 'undefined';
      const keys     = hasClass ? Object.keys(Runner).slice(0, 6).join(', ') : 'N/A';
      overlay.innerHTML = [
        '<b style="color:#fff">DINO BOT</b>', '',
        '<span style="color:#ff4">&#9654; Press SPACE once to start</span>',
        '  then bot takes over', '',
        '<span style="color:#555">Runner class : ' + (hasClass ? '<span style="color:#0f0">ok</span>' : '<span style="color:#f44">missing</span>') + '</span>',
        '<span style="color:#555">static keys  : ' + keys + '</span>', '',
        'If stuck, run in console:',
        '<span style="color:#39ff14">DinoBot.forceRunner(Runner.getInstance())</span>',
      ].join('<br>');
      return;
    }

    if (!runnerLogged) {
      runnerLogged = true;
      console.log('[DinoBot] Runner found:', runner);
      console.log('[DinoBot] playing:', runner.playing, '| crashed:', runner.crashed);
      console.log('[DinoBot] tRex keys:', runner.tRex ? Object.keys(runner.tRex) : 'no tRex');
    }

    const isPlaying = runner.playing || runner.activated;
    if (!isPlaying && !runner.crashed) {
      lastAction = 'starting...';
      updateOverlay(); updateGenPanel();
      gameStart();
      return;
    }

    if (runner.crashed || runner.crashed_) {
      if (!crashedAt) {
        ga.recordFitness(getDisplayScore(runner));
        currentScore = 0; jumpOut = 0; duckOut = 0;
        lastAction   = 'DEAD';
        crashedAt    = Date.now();
      }
      updateOverlay(); updateGenPanel();
      if (Date.now() - crashedAt >= CRASH_PAUSE_MS) { crashedAt = null; gameRestart(runner); }
      return;
    }

    currentScore = getDisplayScore(runner);
    lastInputs   = buildInputs(runner);
    const [jOut, dOut] = ga.getCurrentNetwork().forward(lastInputs);
    jumpOut = jOut; duckOut = dOut;

    if (jumpOut > duckOut && jumpOut > 0.5) {
      gameJump(runner); gameDuck(runner, false); lastAction = 'JUMP';
    } else if (duckOut > jumpOut && duckOut > 0.5) {
      gameDuck(runner, true); lastAction = 'DUCK';
    } else {
      gameDuck(runner, false); lastAction = 'run';
    }

    updateOverlay(); updateGenPanel();
  }

  // ── Main HUD (top-right) ──────────────────────────────────────────────────

  function bar(v, len = 10) {
    const f = Math.round(Math.max(0, Math.min(1, v)) * len);
    return '[' + '|'.repeat(f) + '.'.repeat(len - f) + '] ' + Math.round(v * 100) + '%';
  }

  function actionColor(a) {
    return a === 'JUMP' ? '#ffff00' : a === 'DUCK' ? '#00aaff' : a === 'DEAD' ? '#ff4444' : '#39ff14';
  }

  function createOverlay() {
    overlay = document.createElement('div');
    Object.assign(overlay.style, {
      position: 'fixed', top: '10px', right: '10px',
      background: '#111', color: '#39ff14',
      fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.9',
      padding: '10px 14px', borderRadius: '6px', border: '1px solid #333',
      zIndex: '99999', pointerEvents: 'none', userSelect: 'none', minWidth: '240px',
    });
    document.body.appendChild(overlay);
    updateOverlay();
  }

  function updateOverlay() {
    if (!overlay) return;
    const s       = ga.getStats();
    const obsType = lastInputs[3] > 0.5 ? 'pterodactyl' : 'cactus';
    overlay.innerHTML = [
      '<b style="color:#fff;font-size:13px">DINO BOT</b>',
      '<span style="color:#666">Gen ' + s.generation + '  #' + s.individual + ' / ' + s.populationSize + '</span>',
      '',
      'Score    : <b>' + Math.round(currentScore) + '</b>',
      'Gen best : ' + s.generationBest,
      'All-time : ' + s.allTimeBest,
      '',
      '<span style="color:#aaa">── INPUTS ──────────────────</span>',
      ...INPUT_LABELS.map((l, i) => l + ' ' + bar(lastInputs[i]) + (i === 3 ? ' ' + obsType : '')),
      '',
      '<span style="color:#aaa">── OUTPUTS ─────────────────</span>',
      'jump  ' + bar(jumpOut),
      'duck  ' + bar(duckOut),
      '',
      'Action: <b style="color:' + actionColor(lastAction) + '">' + lastAction + '</b>',
    ].join('<br>');
  }

  // ── Competition mode ──────────────────────────────────────────────────────

  function toggleCompetition() {
    if (competitionActive) { exitCompetition(); return; }
    competitionActive  = true;
    competitionDone    = false;
    competitionResults = null;
    ga.competitionMode = true;
    // Reset current gen so all individuals play from the start
    ga.currentIndex = 0;
    ga.fitnesses    = new Array(ga.populationSize).fill(0);
    updateGenPanel();
    // Make sure the bot is running
    if (!intervalId) start();
  }

  function exitCompetition() {
    competitionActive  = false;
    competitionDone    = false;
    competitionResults = null;
    ga.competitionMode = false;
    ga.currentIndex    = 0;
    ga.fitnesses       = new Array(ga.populationSize).fill(0);
    updateGenPanel();
    if (!intervalId) start();
  }

  // ── Generation performance panel (top-left) ───────────────────────────────

  function createGenPanel() {
    genPanel = document.createElement('div');
    Object.assign(genPanel.style, {
      position: 'fixed', top: '10px', left: '10px',
      background: '#111', color: '#39ff14',
      fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.7',
      borderRadius: '6px', border: '1px solid #333',
      zIndex: '99999', userSelect: 'none', minWidth: '260px',
    });

    genPanel.addEventListener('click', (e) => {
      const btn = e.target.closest('button');
      if (btn) {
        if (btn.dataset.action === 'savegen')    ga.exportPopulationToFile();
        if (btn.dataset.action === 'newgen') {
          if (window.confirm('Reset all training and start a completely new generation?')) {
            competitionActive = false; competitionDone = false; competitionResults = null;
            ga.competitionMode = false;
            ga.reset();
          }
        }
        if (btn.dataset.action === 'competition') toggleCompetition();
        if (btn.dataset.action === 'resumetrain') exitCompetition();
        return;
      }
      if (e.target.closest('[data-toggle]')) { genExpanded = !genExpanded; updateGenPanel(); }
    });

    ga.onEvolve              = () => updateGenPanel();
    ga.onGenerationComplete  = () => {
      // Competition finished — collect sorted results and stop the bot
      competitionDone    = true;
      competitionResults = ga.fitnesses
        .map((score, i) => ({ individual: i + 1, score }))
        .sort((a, b) => b.score - a.score);
      updateGenPanel();
      stop();
    };

    document.body.appendChild(genPanel);
    updateGenPanel();
  }

  function updateGenPanel() {
    if (!genPanel) return;
    const s     = ga.getStats();
    const arrow = genExpanded ? '▲' : '▼';

    // Max score in this generation so far (for normalising bar widths)
    const scores   = ga.fitnesses.slice(0, ga.currentIndex);
    const maxScore = Math.max(...scores, currentScore, 1);

    const modeLabel  = competitionActive ? '<span style="color:#ff8c00"> ⚔ COMPETITION</span>' : '';
    const compBtnTxt = competitionActive ? '✕ Exit' : '⚔ Compete';
    const compBtnClr = competitionActive ? '#6b2a1a' : '#4a3a00';

    let html =
      '<div style="padding:8px 14px;border-bottom:1px solid #222;display:flex;align-items:center;gap:6px">' +
      '<span data-toggle style="flex:1;cursor:pointer"><b style="color:#fff">GEN ' + s.generation + modeLabel + '</b></span>' +
      '<button data-action="newgen"      style="' + BTN('#3a1a1a') + '">+ New</button>' +
      '<button data-action="savegen"     style="' + BTN('#1a3a6b') + '">&#8595; Save</button>' +
      '<button data-action="competition" style="' + BTN(compBtnClr) + '">' + compBtnTxt + '</button>' +
      '<span data-toggle style="color:#666;cursor:pointer">' + arrow + '</span>' +
      '</div>';

    if (genExpanded && competitionDone && competitionResults) {
      // ── Competition results view ─────────────────────────────────────────
      const scores  = competitionResults.map(r => r.score);
      const best    = scores[0];
      const avg     = Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);
      const worst   = scores[scores.length - 1];

      html += '<div style="padding:4px 0">';
      html += '<div style="padding:4px 14px;color:#ff8c00;font-size:11px;border-bottom:1px solid #222">' +
        'Best: <b style="color:#fff">' + best + '</b> &nbsp; ' +
        'Avg: <b style="color:#aaa">' + avg + '</b> &nbsp; ' +
        'Worst: <b style="color:#555">' + worst + '</b>' +
        '</div>';
      competitionResults.forEach((r, rank) => {
        const pct       = Math.min(100, Math.round((r.score / (best || 1)) * 100));
        const rankColor = rank === 0 ? '#ffd700' : rank === 1 ? '#aaa' : rank === 2 ? '#cd7f32' : '#555';
        html +=
          '<div style="padding:3px 14px;display:flex;align-items:center;gap:6px">' +
          '<span style="color:' + rankColor + ';width:20px;text-align:right">#' + (rank + 1) + '</span>' +
          '<span style="color:#555;width:28px;text-align:right;font-size:10px">i' + r.individual + '</span>' +
          '<span style="color:#39ff14;width:34px;text-align:right">' + r.score + '</span>' +
          '<div style="flex:1;background:#1a1a1a;height:8px;border-radius:3px;overflow:hidden">' +
            '<div style="width:' + pct + '%;height:100%;background:#39ff14;border-radius:3px"></div>' +
          '</div>' +
          '</div>';
      });
      html += '<div style="padding:6px 14px;border-top:1px solid #222">' +
        '<button data-action="resumetrain" style="' + BTN('#1a6b2a') + ';width:100%;margin:0">▶ Resume Training</button>' +
        '</div>';
      html += '</div>';
    } else if (genExpanded) {
      html += '<div style="padding:4px 0">';
      for (let i = 0; i < ga.populationSize; i++) {
        const isDone    = i < ga.currentIndex;
        const isCurrent = i === ga.currentIndex;
        const score     = isDone ? ga.fitnesses[i] : (isCurrent ? currentScore : 0);
        const pct       = Math.min(100, Math.round((score / maxScore) * 100));

        const numColor  = isDone ? '#39ff14' : isCurrent ? '#ffff00' : '#444';
        const barColor  = isDone ? '#1a6b2a' : isCurrent ? '#6b6b00' : '#1a1a1a';
        const fillColor = isDone ? '#39ff14' : isCurrent ? '#ffff00' : '#222';
        const label     = isCurrent ? '&#9654;' : isDone ? '&#10003;' : '&nbsp;';
        const scoreText = (isDone || isCurrent) ? String(score).padStart(4, '\u00a0') : '\u00a0---';

        // Origin badge from evolutionLog (only available from gen 2 onward)
        const origin    = ga.evolutionLog && ga.evolutionLog.origins[i];
        const isElite   = origin && origin.type === 'elite';
        const badge     = origin
          ? (isElite
              ? '<span style="color:#39ff14;font-size:10px">[E]</span>'
              : '<span style="color:#ff8c00;font-size:10px">[M]</span>')
          : '<span style="width:22px;display:inline-block"></span>';
        const from      = origin
          ? '<span style="color:#444;font-size:10px">\u2190' + Math.round(origin.prevScore) + '</span>'
          : '';

        html +=
          '<div style="padding:3px 14px;display:flex;align-items:center;gap:6px' + (isCurrent ? ';background:#1a1a00' : '') + '">' +
          '<span style="color:#555;width:20px;text-align:right">#' + (i + 1) + '</span>' +
          badge +
          '<span style="color:' + numColor + ';width:34px;text-align:right">' + scoreText + '</span>' +
          '<div style="flex:1;background:#1a1a1a;height:8px;border-radius:3px;overflow:hidden;border:1px solid ' + barColor + '">' +
            '<div style="width:' + pct + '%;height:100%;background:' + fillColor + ';border-radius:3px;transition:width 0.1s"></div>' +
          '</div>' +
          '<span style="color:' + numColor + ';width:14px;text-align:center">' + label + '</span>' +
          from +
          '</div>';
      }
      html += '<div style="padding:4px 14px;color:#444;font-size:10px;border-top:1px solid #1a1a1a">' +
        '<span style="color:#39ff14">[E]</span> elite carried over &nbsp; ' +
        '<span style="color:#ff8c00">[M]</span> mutated offspring' +
        '</div>';
      html += '</div>';
    } // end else-if genExpanded

    genPanel.innerHTML = html;
  }

  // ── Checkpoint panel (bottom-right) ───────────────────────────────────────

  const BTN = (color) =>
    'background:' + color + ';color:#fff;border:none;border-radius:3px;' +
    'padding:2px 7px;cursor:pointer;font-family:monospace;font-size:11px;margin-left:4px';

  function createCheckpointPanel() {
    cpPanel = document.createElement('div');
    Object.assign(cpPanel.style, {
      position: 'fixed', bottom: '10px', right: '10px',
      background: '#111', color: '#39ff14',
      fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.7',
      borderRadius: '6px', border: '1px solid #333',
      zIndex: '99999', userSelect: 'none', minWidth: '300px',
    });

    cpPanel.addEventListener('click', (e) => {
      const btn = e.target.closest('button');
      if (btn) {
        const idx = parseInt(btn.dataset.idx, 10);
        if (btn.dataset.action === 'dl')     ga.downloadCheckpoint(idx);
        if (btn.dataset.action === 'load')   { ga.loadCheckpoint(idx); updateCheckpointPanel(); }
        if (btn.dataset.action === 'import') ga.importFromFile();
        return;
      }
      if (e.target.closest('[data-toggle]')) { cpExpanded = !cpExpanded; updateCheckpointPanel(); }
    });

    ga.onCheckpoint = () => updateCheckpointPanel();

    document.body.appendChild(cpPanel);
    updateCheckpointPanel();
  }

  function updateCheckpointPanel() {
    if (!cpPanel) return;
    const count = ga.checkpoints.length;
    const arrow = cpExpanded ? '▲' : '▼';

    let html =
      '<div style="padding:8px 14px;border-bottom:1px solid #222;display:flex;align-items:center;gap:8px">' +
      '<span data-toggle style="flex:1;cursor:pointer"><b style="color:#fff">CHECKPOINTS</b> <span style="color:#555">(auto every 5 gens)</span></span>' +
      '<button data-action="import" style="' + BTN('#555') + '">&#8593; Import</button>' +
      '<span data-toggle style="color:#666;cursor:pointer">' + count + ' ' + arrow + '</span>' +
      '</div>';

    if (cpExpanded) {
      if (count === 0) {
        html += '<div style="padding:8px 14px;color:#555">No checkpoints yet — keeps training...</div>';
      } else {
        ga.checkpoints.forEach((cp, i) => {
          html +=
            '<div style="padding:5px 14px;border-bottom:1px solid #1a1a1a;display:flex;align-items:center">' +
            '<span style="flex:1;color:#aaa">' +
              'Gen <b style="color:#fff">' + String(cp.generation).padStart(3, '\u00a0') + '</b>' +
              ' &nbsp;|&nbsp; <b style="color:#39ff14">' + Math.round(cp.fitness) + '</b> pts' +
              ' &nbsp;|&nbsp; <span style="color:#555">' + cp.timestamp + '</span>' +
            '</span>' +
            '<button data-action="dl"   data-idx="' + i + '" style="' + BTN('#1a6b2a') + '">&#8595; Save</button>' +
            '<button data-action="load" data-idx="' + i + '" style="' + BTN('#1a3a6b') + '">&#9654; Load</button>' +
            '</div>';
        });
      }
    }

    cpPanel.innerHTML = html;
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  function start() {
    if (intervalId) { console.log('[DinoBot] Already running.'); return; }
    createOverlay();
    createGenPanel();
    createCheckpointPanel();
    intervalId = setInterval(tick, POLL_MS);
    console.log(
      '[DinoBot] Started.\n' +
      '  DinoBot.stop()                           — pause\n' +
      '  DinoBot.save()                           — download best genome as dino_genome.json\n' +
      '  DinoBot.load()                           — import a saved file (best or population)\n' +
      '  DinoBot.ga.reset()                       — wipe training and start over\n' +
      '  DinoBot.forceRunner(Runner.getInstance()) — set runner manually if needed'
    );
  }

  function stop() {
    clearInterval(intervalId); intervalId = null;
    const runner = getRunner();
    if (runner) gameDuck(runner, false);
    if (overlay)  { overlay.remove();  overlay  = null; }
    if (genPanel) { genPanel.remove(); genPanel = null; }
    if (cpPanel)  { cpPanel.remove();  cpPanel  = null; }
    console.log('[DinoBot] Stopped.');
  }

  start();

  return {
    start, stop, ga,
    save() { ga.exportToFile(); },
    load() { ga.importFromFile(); },
    forceRunner(r) { forcedRunner = r; console.log('[DinoBot] Runner set to:', r); },
  };
})();
