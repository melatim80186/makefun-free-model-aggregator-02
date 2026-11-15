# free-model-aggregator-02

One server that aggregates OpenRouter's free models behind a single endpoint:
it serves the OpenAI Chat Completions API (`POST /v1/chat/completions`) and
the Anthropic Messages API (`POST /v1/messages`), in JSON and in streaming,
plus `GET /v1/models` in the wire its auth header names.

## Run

```sh
node main.ts            # serves, registers with $MAKEFUN_ENV_URL, exits 0 when /work says done
node main.ts --list     # prints the catalog it registers, byte for byte
```

Node 22, standard library only, no dependencies. `MAKEFUN_PROMPT_FILE`
points at the brief: the fourteen free models with the house's ratings and
the six prompts' manifest, in the order the env sends them.

## Routing

For each prompt the server ranks the free models that meet the prompt's
requirements (JSON-mode prompts only to models with JSON support, tools
prompts only to models with tool support) by the house rating, highest
first, ties by model id. It dials the best candidate through the gateway's
Chat Completions route on the trial's key, verifies the answer carries the
prompt's token (and parses as JSON holding it, in JSON mode), and fails over
to the next candidate on any refusal or tokenless answer — never retrying the
same model. A refused model drops to the bottom tier for the rest of the
trial; a model that answered without the token drops to the middle tier.
Recovered refusals are reported on the served response in the
`x-makefun-upstream-failures` header. Streams are buffered whole first, so a
failing or tokenless stream fails over before the client sees anything.
