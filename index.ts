import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { spawn, type ChildProcess } from "node:child_process";
import { createInterface } from "node:readline";
import { createWriteStream, existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

// --- Config ---
const SETTINGS_PATH = join(homedir(), ".pi", "agent", "settings.json");
const PLAUDER_DIR = join(homedir(), ".pi", "agent", "plauder");

// Built-in openwakeword model names (lowercase).
// These get passed through as-is; anything else is resolved as a file path.
const BUILTIN_MODELS = new Set(["alexa", "hey_mycroft", "hey_jarvis", "timer", "weather"]);

interface WakeConfig {
  model: string;
  threshold: number;
  cooldown: number;
  stt_model: string;
  stt_language: string;
  silence_timeout: number;
  max_record_sec: number;
  rms_threshold: number;
  autosubmit: boolean;
}

function readWakeConfig(): WakeConfig {
  const defaults: WakeConfig = {
    model: "hey_jarvis",
    threshold: 0.5,
    cooldown: 2.0,
    stt_model: "tiny",
    stt_language: "",
    silence_timeout: 0.8,
    max_record_sec: 10.0,
    rms_threshold: 500,
    autosubmit: false,
  };
  try {
    const raw = readFileSync(SETTINGS_PATH, "utf-8");
    const parsed = JSON.parse(raw);
    const cfg = parsed.plauder;
    if (!cfg || typeof cfg !== "object") return defaults;
    return {
      model: typeof cfg.model === "string" && cfg.model ? cfg.model : defaults.model,
      threshold: typeof cfg.threshold === "number" ? cfg.threshold : defaults.threshold,
      cooldown: typeof cfg.cooldown === "number" ? cfg.cooldown : defaults.cooldown,
      stt_model: typeof cfg.stt_model === "string" && cfg.stt_model ? cfg.stt_model : defaults.stt_model,
      stt_language: typeof cfg.stt_language === "string" ? cfg.stt_language : defaults.stt_language,
      silence_timeout: typeof cfg.silence_timeout === "number" ? cfg.silence_timeout : defaults.silence_timeout,
      max_record_sec: typeof cfg.max_record_sec === "number" ? cfg.max_record_sec : defaults.max_record_sec,
      rms_threshold: typeof cfg.rms_threshold === "number" ? cfg.rms_threshold : defaults.rms_threshold,
      autosubmit: typeof cfg.autosubmit === "boolean" ? cfg.autosubmit : defaults.autosubmit,
    };
  } catch {
    return defaults;
  }
}

/** Resolve model name to a path or pass-through name. */
function resolveModel(name: string): string {
  const lower = name.toLowerCase();
  if (BUILTIN_MODELS.has(lower)) return lower;
  // Check custom plauder dir
  const customPath = join(PLAUDER_DIR, `${lower}.onnx`);
  if (existsSync(customPath)) return customPath;
  // Fall through — maybe user passed a full path or openwakeword knows it
  return name;
}

// --- Module-scoped state ---
let proc: ChildProcess | null = null;
let retryCount = 0;
let retryTimer: ReturnType<typeof setTimeout> | null = null;
let _startListener: (() => void) | null = null;
let _setStatus: ((status: string | undefined) => void) | null = null;
let autoSubmitEnabled = false;
let _intentionalStop = false; // prevents crash-detection restart when we meant to stop

/**
 * Kill the subprocess: SIGTERM, then SIGKILL after 2s.
 * Also nulls out `proc` so stale refs don't prevent new spawns.
 */
function killProc(): void {
  _intentionalStop = true; // mark intentional so exit handler won't restart
  // Clear pending retry timer (proc may already be null if waiting to retry)
  if (retryTimer !== null) {
    clearTimeout(retryTimer);
    retryTimer = null;
  }
  retryCount = 0;

  const p = proc;
  proc = null;
  if (p === null) return;

  try { p.kill("SIGTERM"); } catch { /* ignore */ }
  const force = setTimeout(() => {
    try { p.kill("SIGKILL"); } catch { /* already dead */ }
  }, 2000);

  p.on("exit", () => clearTimeout(force));

  _setStatus?.("Stopped");
}

/**
 * Spawn the Python wake-word listener.
 */
function spawnListener(args: string[]): ChildProcess {
  const child = spawn("python3", args, {
    stdio: ["ignore", "pipe", "pipe"],
  });
  return child;
}

// --- Safety net for unclean pi exit ---
// Only register once (flag prevents duplicates on reload)
if (!process.listeners("exit").some((fn) => (fn as any)._openwakeword_exit)) {
  const exitHandler = (): void => {
    if (retryTimer !== null) {
      clearTimeout(retryTimer);
      retryTimer = null;
    }
    if (proc !== null) {
      try { proc.kill("SIGKILL"); } catch { /* ignore */ }
      proc = null;
    }
  };
  (exitHandler as any)._openwakeword_exit = true;
  process.on("exit", exitHandler);
}

// =============================================================================
// Extension factory
// =============================================================================
export default function (pi: ExtensionAPI): void {
  // Kill existing process on reload
  killProc();

  const scriptPath = new URL("plauder.py", import.meta.url).pathname;
  const config = readWakeConfig();
  const modelArg = resolveModel(config.model);
  const args = [
    "-u", scriptPath,
    "--model", modelArg,
    "--threshold", String(config.threshold),
    "--cooldown", String(config.cooldown),
    "--stt-model", config.stt_model,
    "--stt-language", config.stt_language,
    "--silence-timeout", String(config.silence_timeout),
    "--max-record-sec", String(config.max_record_sec),
    "--rms-threshold", String(config.rms_threshold),
  ];

  let ctxUi: ExtensionContext["ui"] | null = null;

  pi.on("session_start", (_event, ctx) => {
    ctxUi = ctx.ui;
    autoSubmitEnabled = config.autosubmit;
    _setStatus = (s) => ctx.ui.setStatus("plauder", s ? `plauder: ${s}` : undefined);
    _setStatus("Running");
  });

  // --- Spawn ---
  const startListener = (): void => {
    // Clear any pending retry before spawning
    if (retryTimer !== null) {
      clearTimeout(retryTimer);
      retryTimer = null;
    }
    // Kill any leftover process before spawning a new one
    killProc();

    proc = spawnListener(args);
    _intentionalStop = false; // new process crash = real crash, not intentional stop
    const pid = proc.pid;

    // --- stdout: JSON lines ---
    const rl = createInterface({ input: proc.stdout! });
    rl.on("line", (line: string) => {
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(line);
      } catch {
        return; // skip malformed lines
      }

      switch (msg.event) {
        case "ready":
          retryCount = 0; // reset retry counter on successful start
          _setStatus?.("Running");
          ctxUi?.notify("Wake word listener ready", "info");
          break;

        case "listening":
          ctxUi?.notify("🎤 Listening...", "info");
          break;

        case "transcription": {
          const text = String(msg.text ?? "");
          if (text) {
            if (autoSubmitEnabled) {
              pi.sendUserMessage(text, { deliverAs: "followUp" });
            } else {
              ctxUi?.setEditorText(text);
            }
            ctxUi?.notify("✓ Transcribed", "info");
          }
          break;
        }

        case "transcription_error":
          ctxUi?.notify(`✗ ${String(msg.message ?? "STT error")}`, "warning");
          break;

        case "idle":
          // Optional: clear any listening/status indicator
          break;

        case "error": {
          const message = String(msg.message ?? "unknown error");
          ctxUi?.notify(`Wake word error: ${message}`, "error");
          break;
        }
      }
    });

    // --- stderr → log file (not TUI) ---
    const logFile = "/tmp/openwakeword.log";
    const logStream = createWriteStream(logFile, { flags: "a" });
    logStream.write(`\n--- openwakeword listener pid=${pid} started at ${new Date().toISOString()} ---\n`);
    proc.stderr!.pipe(logStream);
    proc.on("exit", () => logStream.end());

    // --- Handle exit / crash ---
    proc.on("exit", (code: number | null, signal: string | null) => {
      rl.close();
      proc = null;

      if (_intentionalStop) {
        _intentionalStop = false;
        return;
      }

      // Restart with backoff (max 3 retries)
      if (retryCount < 3) {
        const delay = 1000 * Math.pow(2, retryCount); // 1s, 2s, 4s
        retryCount++;
        _setStatus?.(`Crashed, retry ${retryCount}/3 in ${delay / 1000}s`);
        ctxUi?.notify(
          `Wake word listener crashed (exit ${code}), retrying in ${delay / 1000}s...`,
          "warning",
        );
        retryTimer = setTimeout(startListener, delay);
      } else {
        _setStatus?.("Failed");
        ctxUi?.notify(
          "Wake word listener failed after 3 retries. Restart pi or reload extension.",
          "error",
        );
      }
    });
  };

  _startListener = startListener;
  startListener();

  // --- Commands ---
  pi.registerCommand("plauder", {
    description: "Plauder commands: help, toggle",
    handler: async (args, ctx) => {
      const sub = args.trim().split(/\s+/)[0];

      if (sub === "help") {
        ctx.ui.notify(
          "Plauder commands:\n" +
          "  /plauder help       - Show this help\n" +
          "  /plauder toggle     - Start/stop the wake word listener\n" +
          "  /plauder autosubmit - Toggle instant submit on transcription",
          "info",
        );
      } else if (sub === "toggle") {
        if (proc !== null) {
          killProc();
          ctx.ui.setStatus("plauder", "plauder: Stopped");
          ctx.ui.notify("Plauder listener stopped", "info");
        } else {
          _startListener?.();
          ctx.ui.setStatus("plauder", "plauder: Running");
          ctx.ui.notify("Plauder listener started", "info");
        }
      } else if (sub === "autosubmit") {
        autoSubmitEnabled = !autoSubmitEnabled;
        const status = autoSubmitEnabled ? "on" : "off";
        ctx.ui.setStatus("plauder", `plauder: Running (autosubmit ${status})`);
        ctx.ui.notify(`Auto-submit ${status}`, "info");
      } else {
        ctx.ui.notify(
          "Unknown plauder command. Try: /plauder help",
          "warning",
        );
      }
    },
  });

  // --- Lifecycle: shutdown ---
  pi.on("session_shutdown", (event) => {
    if (event.reason === "quit" || event.reason === "reload") {
      if (retryTimer !== null) {
        clearTimeout(retryTimer);
        retryTimer = null;
      }
      killProc();
    }
  });
}
