// providers.ts — brief parsing, catalog building, candidate ranking.
// The brief (MAKEFUN_PROMPT_FILE) is prose with two JSON documents embedded:
// a models document (schema free-model-aggregator-02/models/v1) and a prompts
// document (schema free-model-aggregator-02/prompts/v1). Nothing else is read.

export interface FreeModel {
  id: string;
  rating_milli: number;
  context_length: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface PromptSpec {
  id: string;
  api: string;
  stream: boolean;
  weight: number;
  requires: { json?: boolean; tools?: boolean; vision?: boolean };
}

export interface Brief {
  models: FreeModel[];
  prompts: PromptSpec[];
}

// Extract one balanced JSON object starting at the given schema marker.
function extractDoc(text: string, schema: string): any | null {
  const marker = `"schema":"${schema}"`;
  const at = text.indexOf(marker);
  if (at < 0) return null;
  const start = text.lastIndexOf("{", at);
  if (start < 0) return null;
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === "\\") esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) {
        try {
          return JSON.parse(text.slice(start, i + 1));
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

export function parseBrief(text: string): Brief | null {
  const modelsDoc = extractDoc(text, "free-model-aggregator-02/models/v1");
  const promptsDoc = extractDoc(text, "free-model-aggregator-02/prompts/v1");
  if (!modelsDoc || !Array.isArray(modelsDoc.models)) return null;
  if (!promptsDoc || !Array.isArray(promptsDoc.prompts)) return null;
  const models: FreeModel[] = [];
  for (const m of modelsDoc.models) {
    if (typeof m.id !== "string") continue;
    models.push({
      id: m.id,
      rating_milli: typeof m.rating_milli === "number" ? m.rating_milli : 0,
      context_length: typeof m.context_length === "number" ? m.context_length : null,
      tools: m.tools === true,
      vision: m.vision === true,
      json: m.json === true,
      free: m.free !== false,
    });
  }
  const prompts: PromptSpec[] = [];
  for (const p of promptsDoc.prompts) {
    if (typeof p.id !== "string") continue;
    prompts.push({
      id: p.id,
      api: p.api === "anthropic" ? "anthropic" : "openai",
      stream: p.stream === true,
      weight: p.weight === 2 ? 2 : 1,
      requires: (p.requires && typeof p.requires === "object" ? p.requires : {}) as PromptSpec["requires"],
    });
  }
  if (models.length === 0 || prompts.length === 0) return null;
  return { models, prompts };
}

export function buildCatalog(models: FreeModel[]): any {
  return {
    providers: [
      {
        id: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models: models.map((m) => ({ id: m.id, free: true, context_length: m.context_length })),
      },
    ],
  };
}

// Rank the models that may serve a prompt: free, meeting its requirements,
// rating desc, ties by model id asc. Tiered by trial history: 0 = untried or
// clean, 1 = answered once without the prompt's token, 2 = refused a call.
// Lower tier number is tried first; within a tier the rating order holds.
export function rankCandidates(
  models: FreeModel[],
  requires: { json?: boolean; tools?: boolean },
  tiers: Map<string, number>,
): FreeModel[] {
  const ok: FreeModel[] = [];
  for (const m of models) {
    if (!m.free) continue;
    if (requires.json === true && !m.json) continue;
    if (requires.tools === true && !m.tools) continue;
    ok.push(m);
  }
  ok.sort((a, b) => {
    const ta = tiers.get(a.id) ?? 0;
    const tb = tiers.get(b.id) ?? 0;
    if (ta !== tb) return ta - tb;
    if (b.rating_milli !== a.rating_milli) return b.rating_milli - a.rating_milli;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
  return ok;
}
