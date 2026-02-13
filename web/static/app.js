const boardEl = document.getElementById("board");
const scoreEl = document.getElementById("score");
const maxTileEl = document.getElementById("max-tile");
const doneEl = document.getElementById("done");
const lastActionEl = document.getElementById("last-action");
const lastRewardEl = document.getElementById("last-reward");
const qValuesEl = document.getElementById("q-values-list");
const errorEl = document.getElementById("error");

const resetBtn = document.getElementById("reset-btn");
const modelStepBtn = document.getElementById("model-step-btn");
const autoplayBtn = document.getElementById("autoplay-btn");
const speedInput = document.getElementById("speed");
const speedLabel = document.getElementById("speed-label");

const manualButtons = Array.from(document.querySelectorAll("[data-action]"));

const actionNames = ["Up", "Down", "Left", "Right"];
let autoMode = false;
let autoDelayMs = Number(speedInput.value);
let lastState = null;
let inFlight = false;

function showError(message) {
  errorEl.textContent = message || "";
}

async function requestJson(url, method = "GET", body = null) {
  const options = { method, headers: {} };
  if (body !== null) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }

  const response = await fetch(url, options);
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status})`);
  }
  return payload;
}

function renderBoard(board) {
  boardEl.innerHTML = "";
  for (const row of board) {
    for (const value of row) {
      const tile = document.createElement("div");
      tile.className = "tile";
      if (value > 0) {
        const level = Math.min(12, Math.max(1, Math.floor(Math.log2(value))));
        tile.classList.add("active", `level-${level}`);
        tile.textContent = String(value);
      }
      boardEl.appendChild(tile);
    }
  }
}

function renderQValues(qValues) {
  qValuesEl.innerHTML = "";
  const values = Array.isArray(qValues) ? qValues : [];
  values.forEach((value, action) => {
    const item = document.createElement("li");
    item.textContent = `${actionNames[action]} (${action}): ${Number(value).toFixed(3)}`;
    qValuesEl.appendChild(item);
  });
}

function renderState(state) {
  lastState = state;
  renderBoard(state.board || []);
  scoreEl.textContent = `Score: ${state.score ?? 0}`;
  maxTileEl.textContent = `Max Tile: ${state.max_tile ?? 0}`;
  doneEl.textContent = state.done ? "Game Over" : "Running";
  lastActionEl.textContent = state.last_action_name || "None";
  lastRewardEl.textContent = Number(state.last_reward || 0).toFixed(3);
  renderQValues(state.q_values);
}

function updateAutoplayButton() {
  autoplayBtn.textContent = autoMode ? "Stop Autoplay" : "Start Autoplay";
}

async function loadState() {
  const state = await requestJson("/api/state");
  renderState(state);
}

async function resetGame() {
  const state = await requestJson("/api/reset", "POST", {});
  renderState(state);
}

async function modelStep() {
  if (inFlight) return;
  inFlight = true;
  try {
    const state = await requestJson("/api/step/model", "POST", {});
    renderState(state);
    if (state.done) {
      autoMode = false;
      updateAutoplayButton();
    }
  } finally {
    inFlight = false;
  }
}

async function manualStep(action) {
  if (inFlight) return;
  inFlight = true;
  try {
    const state = await requestJson("/api/step/manual", "POST", { action });
    renderState(state);
    if (state.done) {
      autoMode = false;
      updateAutoplayButton();
    }
  } finally {
    inFlight = false;
  }
}

async function autoLoop() {
  if (!autoMode) return;
  if (lastState && lastState.done) {
    autoMode = false;
    updateAutoplayButton();
    return;
  }
  await modelStep();
  if (autoMode) {
    setTimeout(autoLoop, autoDelayMs);
  }
}

resetBtn.addEventListener("click", async () => {
  showError("");
  try {
    autoMode = false;
    updateAutoplayButton();
    await resetGame();
  } catch (error) {
    showError(error.message);
  }
});

modelStepBtn.addEventListener("click", async () => {
  showError("");
  try {
    await modelStep();
  } catch (error) {
    showError(error.message);
  }
});

autoplayBtn.addEventListener("click", async () => {
  showError("");
  try {
    autoMode = !autoMode;
    updateAutoplayButton();
    if (autoMode) {
      autoLoop();
    }
  } catch (error) {
    showError(error.message);
  }
});

speedInput.addEventListener("input", () => {
  autoDelayMs = Number(speedInput.value);
  speedLabel.textContent = `${autoDelayMs} ms`;
});

manualButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    showError("");
    try {
      const action = Number(button.dataset.action);
      await manualStep(action);
    } catch (error) {
      showError(error.message);
    }
  });
});

(async () => {
  try {
    await loadState();
  } catch (error) {
    showError(error.message);
  }
})();
