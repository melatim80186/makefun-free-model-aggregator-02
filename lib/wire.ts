// wire.ts — translation between the two client wires and the gateway's
// Chat Completions route, plus response building in each wire.
// Every model call goes through POST $OPENAI_BASE_URL/chat/completions;
// the Anthropic upstream route refuses these ids, so it is never used.

export interface GatewayAnswer {
  text: string;
  finishReason: string;
  usage: { input: number; output: number };
  id: string;
  created: number;
  toolCalls: any[];
}

const TOKEN_RE = /[A-Z]{3,12}-[0-9]{4}/;

export function extractToken(bodyText: string): string | null {
  const m = TOKEN_RE.exec(bodyText);
  return m ? m[0] : null;
}

// The gateway answer must carry the prompt's token; a JSON-mode answer must
// also parse as the object holding that token.
export function verifyAnswer(text: string, token: string | null, jsonMode: boolean): boolean {
  if (token === null) return text.length > 0;
  if (!text.includes(token)) return false;
  if (!jsonMode) return true;
  try {
    const v = JSON.parse(text);
    return v !== null && typeof v === "object" && JSON.stringify(v).includes(token);
  } catch {
    return false;
  }
}

function textOfContent(content: any): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (block && typeof block === "object" && block.type === "text" && typeof block.text === "string") {
      parts.push(block.text);
    } else if (block && typeof block === "object" && block.type === "tool_result") {
      parts.push(typeof block.content === "string" ? block.content : JSON.stringify(block.content ?? ""));
    }
  }
  return parts.join("");
}

function capTokens(body: any): number {
  const n = typeof body.max_tokens === "number" ? body.max_tokens
    : typeof body.max_completion_tokens === "number" ? body.max_completion_tokens : 1024;
  if (!Number.isFinite(n) || n < 1) return 1024;
  return Math.min(Math.floor(n), 8192);
}

// OpenAI-shaped client body -> gateway Chat Completions body.
export function openaiToGateway(body: any, modelId: string, stream: boolean): any {
  const out: any = {
    model: modelId,
    messages: Array.isArray(body.messages) ? body.messages : [],
    max_tokens: capTokens(body),
  };
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (body.response_format && typeof body.response_format === "object") out.response_format = body.response_format;
  if (Array.isArray(body.tools)) out.tools = body.tools;
  if (body.tool_choice !== undefined) out.tool_choice = body.tool_choice;
  if (stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

// Anthropic-shaped client body -> gateway Chat Completions body.
export function anthropicToGateway(body: any, modelId: string, stream: boolean): any {
  const messages: any[] = [];
  if (typeof body.system === "string" && body.system !== "") {
    messages.push({ role: "system", content: body.system });
  } else if (Array.isArray(body.system)) {
    messages.push({ role: "system", content: textOfContent(body.system) });
  }
  if (Array.isArray(body.messages)) {
    for (const m of body.messages) {
      messages.push({ role: m.role === "assistant" ? "assistant" : "user", content: textOfContent(m.content) });
    }
  }
  const out: any = {
    model: modelId,
    messages,
    max_tokens: capTokens(body),
  };
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (Array.isArray(body.tools)) {
    out.tools = body.tools.map((t: any) => ({
      type: "function",
      function: {
        name: t.name,
        description: typeof t.description === "string" ? t.description : "",
        parameters: t.input_schema && typeof t.input_schema === "object" ? t.input_schema : { type: "object", properties: {} },
      },
    }));
  }
  if (stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function normUsage(raw: any, text: string): { input: number; output: number } {
  let input = 0;
  let output = 0;
  if (raw && typeof raw === "object") {
    if (typeof raw.prompt_tokens === "number" && raw.prompt_tokens >= 0) input = Math.floor(raw.prompt_tokens);
    if (typeof raw.completion_tokens === "number" && raw.completion_tokens >= 0) output = Math.floor(raw.completion_tokens);
  }
  if (output < 1) output = Math.max(1, Math.ceil(text.length / 4));
  return { input, output };
}

function normFinish(raw: any): string {
  const legal = ["stop", "length", "tool_calls", "function_call", "content_filter"];
  return typeof raw === "string" && legal.includes(raw) ? raw : "stop";
}

// Parse a non-streamed gateway JSON body into a GatewayAnswer.
export function parseGatewayJson(value: any): GatewayAnswer {
  const choice = value && Array.isArray(value.choices) ? value.choices[0] : null;
  const message = choice && choice.message && typeof choice.message === "object" ? choice.message : null;
  const text = message ? textOfContent(message.content) : "";
  const toolCalls = message && Array.isArray(message.tool_calls) ? message.tool_calls : [];
  return {
    text,
    finishReason: normFinish(choice ? choice.finish_reason : null),
    usage: normUsage(value ? value.usage : null, text),
    id: typeof value.id === "string" && value.id !== "" ? value.id : "chatcmpl-aggregator",
    created: typeof value.created === "number" ? value.created : 0,
    toolCalls,
  };
}

// Parse a buffered gateway SSE body into a GatewayAnswer.
export function parseGatewayStream(rawText: string): GatewayAnswer {
  let text = "";
  let finishReason: string | null = null;
  let usage: any = null;
  let id = "";
  let created = 0;
  const toolCalls: any[] = [];
  const normalized = rawText.replace(/\r\n/g, "\n");
  for (const frame of normalized.split("\n\n")) {
    if (frame.trim() === "") continue;
    const lines = frame.split("\n").filter((l) => l.startsWith("data:"));
    if (lines.length === 0) continue;
    const payload = lines.map((l) => l.slice("data:".length).replace(/^ /, "")).join("\n").trim();
    if (payload === "" || payload === "[DONE]") continue;
    let chunk: any;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!chunk || typeof chunk !== "object") continue;
    if (typeof chunk.id === "string" && id === "") id = chunk.id;
    if (typeof chunk.created === "number" && created === 0) created = chunk.created;
    if (chunk.usage && typeof chunk.usage === "object") usage = chunk.usage;
    if (!Array.isArray(chunk.choices) || chunk.choices.length === 0) continue;
    const choice = chunk.choices[0];
    if (!choice || typeof choice !== "object") continue;
    const delta = choice.delta && typeof choice.delta === "object" ? choice.delta : null;
    if (delta) {
      if (typeof delta.content === "string") text += delta.content;
      if (Array.isArray(delta.tool_calls)) for (const tc of delta.tool_calls) toolCalls.push(tc);
    }
    if (typeof choice.finish_reason === "string" && finishReason === null) finishReason = choice.finish_reason;
  }
  return {
    text,
    finishReason: normFinish(finishReason),
    usage: normUsage(usage, text),
    id: id !== "" ? id : "chatcmpl-aggregator",
    created,
    toolCalls,
  };
}

// ---- response builders (client wires) ----

export function openaiJsonResponse(ans: GatewayAnswer, modelId: string): any {
  const message: any = { role: "assistant", content: ans.text };
  if (ans.toolCalls.length > 0) message.tool_calls = ans.toolCalls;
  return {
    id: ans.id,
    object: "chat.completion",
    created: ans.created,
    model: modelId,
    choices: [{ index: 0, message, finish_reason: ans.finishReason }],
    usage: {
      prompt_tokens: ans.usage.input,
      completion_tokens: ans.usage.output,
      total_tokens: ans.usage.input + ans.usage.output,
    },
  };
}

// OpenAI SSE: one content delta, one finish chunk, one usage-only chunk
// (choices: [] so it carries no second finish_reason), then [DONE].
// A provider that repeats finish_reason on its usage chunk would otherwise
// hand the client two finish chunks, which the env's stream check refuses.
export function openaiSseText(ans: GatewayAnswer, modelId: string): string {
  const base = { id: ans.id, object: "chat.completion.chunk", created: ans.created, model: modelId };
  const frames: string[] = [];
  const delta: any = { role: "assistant", content: ans.text };
  if (ans.toolCalls.length > 0) delta.tool_calls = ans.toolCalls;
  frames.push(`data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta, finish_reason: null }] })}\n\n`);
  frames.push(`data: ${JSON.stringify({ ...base, choices: [{ index: 0, delta: {}, finish_reason: ans.finishReason }] })}\n\n`);
  frames.push(`data: ${JSON.stringify({ ...base, choices: [], usage: { prompt_tokens: ans.usage.input, completion_tokens: ans.usage.output, total_tokens: ans.usage.input + ans.usage.output } })}\n\n`);
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

function anthropicStop(finishReason: string): string {
  if (finishReason === "length") return "max_tokens";
  if (finishReason === "tool_calls" || finishReason === "function_call") return "tool_use";
  return "end_turn";
}

export function anthropicJsonResponse(ans: GatewayAnswer, modelId: string): any {
  const content: any[] = [{ type: "text", text: ans.text }];
  for (const tc of ans.toolCalls) {
    const fn = tc.function && typeof tc.function === "object" ? tc.function : {};
    let input: any = {};
    try {
      input = typeof fn.arguments === "string" ? JSON.parse(fn.arguments) : {};
    } catch {
      input = {};
    }
    content.push({ type: "tool_use", id: typeof tc.id === "string" ? tc.id : "toolu-aggregator", name: typeof fn.name === "string" ? fn.name : "", input });
  }
  return {
    id: `msg_${ans.id}`,
    type: "message",
    role: "assistant",
    model: modelId,
    content,
    stop_reason: anthropicStop(ans.finishReason),
    stop_sequence: null,
    usage: { input_tokens: ans.usage.input, output_tokens: ans.usage.output },
  };
}

// Anthropic SSE, synthesized from the buffered answer in strict order:
// message_start, content_block_start, content_block_delta(s),
// content_block_stop, message_delta, message_stop.
export function anthropicSseText(ans: GatewayAnswer, modelId: string): string {
  const msgId = `msg_${ans.id}`;
  const frames: string[] = [];
  const ev = (event: string, data: any) => frames.push(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  ev("message_start", {
    type: "message_start",
    message: {
      id: msgId,
      type: "message",
      role: "assistant",
      model: modelId,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: ans.usage.input, output_tokens: 0 },
    },
  });
  ev("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
  const step = 200;
  if (ans.text.length === 0) {
    ev("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "" } });
  } else {
    for (let i = 0; i < ans.text.length; i += step) {
      ev("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: ans.text.slice(i, i + step) } });
    }
  }
  ev("content_block_stop", { type: "content_block_stop", index: 0 });
  ev("message_delta", {
    type: "message_delta",
    delta: { stop_reason: anthropicStop(ans.finishReason), stop_sequence: null },
    usage: { output_tokens: ans.usage.output },
  });
  ev("message_stop", { type: "message_stop" });
  return frames.join("");
}
