// main.ts — one server aggregating OpenRouter's free models behind both
// the OpenAI Chat Completions wire and the Anthropic Messages wire.
// Node 22, standard library only. Deterministic: every routing choice comes
// from the brief, the request, and the gateway answers seen so far — no wall
// clock, no unseeded randomness in any decision.
import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { parseBrief, buildCatalog, rankCandidates } from "./lib/providers.ts";
import {
  extractToken,
  verifyAnswer,
  openaiToGateway,
  anthropicToGateway,
  parseGatewayJson,
  parseGatewayStream,
  openaiJsonResponse,
  openaiSseText,
  anthropicJsonResponse,
  anthropicSseText,
} from "./lib/wire.ts";

const EMPTY_CATALOG = { providers: [] };

function loadBrief(): any | null {
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (!file) return null;
  try {
    const brief = parseBrief(readFileSync(file, "utf8"));
    if (!brief) return null;
    return { brief, catalog: buildCatalog(brief.models) };
  } catch {
    return null;
  }
}

const loaded = loadBrief();
const catalog = loaded ? loaded.catalog : EMPTY_CATALOG;
const catalogText = JSON.stringify(catalog);

// --list prints the registered catalog byte for byte, then exits.
if (process.argv.includes("--list")) {
  process.stdout.write(catalogText + "\n");
  process.exit(0);
}

const brief = loaded ? loaded.brief : null;
const models = brief ? brief.models : [];
const manifest = brief ? brief.prompts : [];

// Trial history: refused models drop to the bottom tier for the rest of the
// trial; models that answered without the prompt's token drop to the middle.
const tiers = new Map<string, number>();
let promptIndex = 0;

function envOrigin(): string {
  return (process.env.MAKEFUN_ENV_URL || "").replace(/\/+$/, "");
}

function gatewayBase(): string {
  return (process.env.OPENAI_BASE_URL || "").replace(/\/+$/, "");
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function readBody(req: any): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (c: Buffer) => {
      size += c.length;
      if (size <= 2 * 1024 * 1024) chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function sendJson(res: any, status: number, value: any, failures: any[]): void {
  const body = JSON.stringify(value);
  const headers: Record<string, string> = {
    "content-type": "application/json",
    "content-length": String(Buffer.byteLength(body)),
  };
  if (failures.length > 0) {
    headers["x-makefun-upstream-failures"] = JSON.stringify(failures.slice(0, 8));
  }
  res.writeHead(status, headers);
  res.end(body);
}

function sendSse(res: any, text: string, failures: any[]): void {
  const headers: Record<string, string> = {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
  };
  if (failures.length > 0) {
    headers["x-makefun-upstream-failures"] = JSON.stringify(failures.slice(0, 8));
  }
  res.writeHead(200, headers);
  res.end(text);
}

// One gateway call for one model. Returns the parsed answer on success, or
// null on a refusal (the refusal row is appended to `failures` when the
// gateway answered with a real HTTP status; a transport error with no status
// is never reported as a row).
async function dialModel(
  gwBody: any,
  modelId: string,
  stream: boolean,
  failures: any[],
): Promise<any | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 25000);
  try {
    const res = await fetch(`${gatewayBase()}/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${process.env.OPENAI_API_KEY || ""}`,
      },
      body: JSON.stringify(gwBody),
      signal: controller.signal,
    });
    if (res.status < 200 || res.status >= 300) {
      let code = "";
      try {
        const errBody = await res.json();
        const e = errBody && errBody.error && typeof errBody.error === "object" ? errBody.error : null;
        if (e) code = typeof e.code === "string" ? e.code : typeof e.type === "string" ? e.type : "";
      } catch {
        code = "";
      }
      failures.push({ status: res.status, code: code.slice(0, 64), model: modelId });
      return null;
    }
    if (stream) {
      const text = await res.text();
      return parseGatewayStream(text);
    }
    const value = await res.json();
    return parseGatewayJson(value);
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

// Serve one prompt: try candidates in ranked order, fail over on any refusal
// or tokenless answer, never retrying the same model. Returns the answer to
// serve plus the model that produced it, or null when nothing could serve it.
async function servePrompt(
  body: any,
  wire: string,
): Promise<{ answer: any; modelId: string; failures: any[] } | null> {
  const row = promptIndex < manifest.length ? manifest[promptIndex] : null;
  promptIndex += 1;
  let requires: { json?: boolean; tools?: boolean } = {};
  if (row) {
    requires = row.requires || {};
  } else {
    if (body.response_format && body.response_format.type === "json_object") requires.json = true;
    if (Array.isArray(body.tools) && body.tools.length > 0) requires.tools = true;
  }
  const stream = body.stream === true;
  const token = extractToken(JSON.stringify(body));
  const jsonMode = requires.json === true;
  const candidates = rankCandidates(models, requires, tiers);
  const failures: any[] = [];
  let tokenless: { answer: any; modelId: string } | null = null;
  for (const candidate of candidates) {
    const gwBody = wire === "anthropic"
      ? anthropicToGateway(body, candidate.id, stream)
      : openaiToGateway(body, candidate.id, stream);
    const ans = await dialModel(gwBody, candidate.id, stream, failures);
    if (ans === null) {
      // Refused (or unreachable): bottom tier for the rest of the trial.
      tiers.set(candidate.id, 2);
      continue;
    }
    if (!verifyAnswer(ans.text, token, jsonMode)) {
      // Answered, but without the token (or unparseable JSON-mode): middle
      // tier, keep the best such answer only as a last resort.
      tiers.set(candidate.id, 1);
      if (tokenless === null) tokenless = { answer: ans, modelId: candidate.id };
      continue;
    }
    return { answer: ans, modelId: candidate.id, failures };
  }
  // No candidate verified. Serve the best tokenless answer rather than
  // nothing: its shape, usage and stop reason still count on the wire.
  if (tokenless !== null) return { answer: tokenless.answer, modelId: tokenless.modelId, failures };
  return null;
}

// ---- HTTP routes ----

function modelsList(req: any): any {
  const rows = models.map((m: any) => ({ id: m.id, free: true, context_length: m.context_length }));
  const anthropic = req.headers["x-api-key"] !== undefined;
  if (anthropic) {
    return {
      object: "list",
      data: rows.map((r: any) => ({ id: r.id, type: "model", display_name: r.id, created: 0, free: true })),
    };
  }
  return {
    object: "list",
    data: rows.map((r: any) => ({ id: r.id, object: "model", created: 0, owned_by: "openrouter", free: true })),
  };
}

async function handleCompletions(req: any, res: any): Promise<void> {
  let body: any;
  try {
    body = JSON.parse(await readBody(req));
  } catch {
    sendJson(res, 400, { error: { message: "request body is not JSON", type: "invalid_request_error", code: "invalid_json" } }, []);
    return;
  }
  const served = await servePrompt(body, "openai");
  if (served === null) {
    sendJson(res, 502, { error: { message: "all free models refused or failed to answer", type: "server_error", code: "upstream_failed" } }, []);
    return;
  }
  if (body.stream === true) {
    sendSse(res, openaiSseText(served.answer, served.modelId), served.failures);
  } else {
    sendJson(res, 200, openaiJsonResponse(served.answer, served.modelId), served.failures);
  }
}

async function handleMessages(req: any, res: any): Promise<void> {
  let body: any;
  try {
    body = JSON.parse(await readBody(req));
  } catch {
    sendJson(res, 400, { error: { message: "request body is not JSON", type: "invalid_request_error" } }, []);
    return;
  }
  const served = await servePrompt(body, "anthropic");
  if (served === null) {
    sendJson(res, 502, { error: { message: "all free models refused or failed to answer", type: "server_error" } }, []);
    return;
  }
  if (body.stream === true) {
    sendSse(res, anthropicSseText(served.answer, served.modelId), served.failures);
  } else {
    sendJson(res, 200, anthropicJsonResponse(served.answer, served.modelId), served.failures);
  }
}

const server = createServer((req: any, res: any) => {
  const path = (req.url || "/").split("?")[0];
  if (req.method === "GET" && path === "/v1/models") {
    sendJson(res, 200, modelsList(req), []);
    return;
  }
  if (req.method === "POST" && path === "/v1/chat/completions") {
    void handleCompletions(req, res);
    return;
  }
  if (req.method === "POST" && path === "/v1/messages") {
    void handleMessages(req, res);
    return;
  }
  sendJson(res, 404, { error: { message: `no route ${req.method} ${path}`, type: "not_found_error" } }, []);
});

async function register(port: number): Promise<void> {
  const origin = envOrigin();
  if (!origin) return;
  // Retry until the env accepts us: a single transient failure must not
  // strand the trial. Re-registering is harmless (the env takes the first
  // register and ignores repeats).
  for (let attempt = 0; attempt < 60; attempt++) {
    try {
      const res = await fetch(`${origin}/register`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
      });
      if (res.ok) return;
    } catch {
      // Env not up yet; retry below.
    }
    await sleep(500);
  }
}

async function pollWork(): Promise<void> {
  const origin = envOrigin();
  if (!origin) return;
  for (;;) {
    try {
      const res = await fetch(`${origin}/work`);
      if (res.ok) {
        const value = await res.json();
        if (value && value.done === true) {
          process.exit(0);
        }
      }
    } catch {
      // Env not up yet; keep polling.
    }
    await sleep(400);
  }
}

server.listen(0, "0.0.0.0", () => {
  const addr = server.address();
  const port = addr && typeof addr === "object" ? addr.port : 0;
  void register(port).then(() => pollWork());
});
