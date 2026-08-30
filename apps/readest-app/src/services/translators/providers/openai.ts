import { stubTranslation as _ } from '@/utils/misc';
import { TranslationProvider } from '../types';
import { useSettingsStore } from '@/store/settingsStore';
import { getAIFetch } from '@/services/ai/utils/httpFetch';
import { TRANSLATED_LANGS, TRANSLATOR_LANGS } from '@/services/constants';
import { normalizeToShortLang } from '@/utils/lang';

/**
 * Translation through any OpenAI-compatible /v1/chat/completions endpoint
 * (OpenAI, DeepSeek, Moonshot, vLLM, LiteLLM, one-api proxies, ...).
 *
 * Credentials come from the AI settings the user already manages in
 * Settings -> AI ("OpenRouter (Custom)" fields: `aiSettings.openrouter*`),
 * so there is no separate translation credentials UI — chat and translation
 * share one endpoint/key/model.
 *
 * Transport mirrors the AI providers: `getAIFetch()` routes through the
 * Tauri HTTP plugin (no CORS preflight, no Android cleartext block) and
 * falls back to `window.fetch` on web.
 *
 * Protocol: paragraphs go out as numbered lines in a single request and must
 * come back as a strict JSON object `{"translations": [...]}` with the same
 * count and order. Models that mangle the JSON contract (wrong item count,
 * prose wrapper) fall back to one request per line, which cannot scramble.
 */
const DEFAULT_BASE_URL = 'https://api.openai.com/v1';
const DEFAULT_MODEL = 'gpt-4o-mini';
// A batch is one chat request; these caps keep a single prompt well inside
// typical context windows while amortising per-request latency over a page.
const MAX_LINES_PER_BATCH = 20;
const MAX_CHARS_PER_BATCH = 6000;

export interface OpenAITranslateConfig {
  apiKey: string;
  baseUrl: string;
  model: string;
}

export const getOpenAITranslateConfig = (): OpenAITranslateConfig => {
  const aiSettings = useSettingsStore.getState().settings.aiSettings;
  return {
    apiKey: aiSettings?.openrouterApiKey ?? '',
    // The convention matches the OpenAI SDK: the base URL ends at /v1.
    // Tolerate a bare host by appending /v1 before /chat/completions.
    baseUrl: (aiSettings?.openrouterBaseUrl || DEFAULT_BASE_URL).replace(/\/+$/, ''),
    model: aiSettings?.openrouterModel || DEFAULT_MODEL,
  };
};

const chatCompletionsUrl = (baseUrl: string): string =>
  /\/v\d+$/.test(baseUrl) ? `${baseUrl}/chat/completions` : `${baseUrl}/v1/chat/completions`;

/** Native language name for the prompt; falls back to the raw code. */
const langName = (lang: string): string => {
  if (!lang || lang.toUpperCase() === 'AUTO') return _('auto-detect');
  const short = normalizeToShortLang(lang);
  return (
    TRANSLATED_LANGS[short as keyof typeof TRANSLATED_LANGS] ??
    TRANSLATOR_LANGS[short as keyof typeof TRANSLATOR_LANGS] ??
    short
  );
};

const buildSystemPrompt = (source: string, target: string, lines: number): string =>
  [
    `You are a translation engine. Translate each numbered line from ${source} into ${target}.`,
    'Output STRICT JSON only, no markdown fences, no commentary:',
    '{"translations": ["translation of line 1", "translation of line 2"]}',
    `The array must contain exactly ${lines} strings, in the same order as the input lines.`,
    'Do not merge, split, reorder, add or drop lines. Do not add numbering or quotes.',
    'Keep inline markup tags (e.g. <b>, <i>) on the matching translated words.',
    'Copy non-translatable lines (numbers, URLs, code, proper nouns) as-is.',
  ].join('\n');

const buildUserPrompt = (lines: string[]): string =>
  lines.map((line, i) => `${i + 1}. ${line.replace(/\n/g, ' ')}`).join('\n');

const parseTranslations = (content: string, expected: number): string[] | null => {
  const start = content.indexOf('{');
  const end = content.lastIndexOf('}');
  if (start === -1 || end <= start) return null;
  try {
    const parsed = JSON.parse(content.slice(start, end + 1)) as {
      translations?: unknown[];
    };
    const arr = parsed?.translations;
    if (!Array.isArray(arr) || arr.length !== expected) return null;
    return arr.map((item) => (typeof item === 'string' ? item : item == null ? '' : String(item)));
  } catch {
    return null;
  }
};

/** Strips the symmetric wrapping quotes some models add around translations. */
const stripWrappingQuotes = (text: string): string =>
  text.length >= 2 &&
  ((text.startsWith('"') && text.endsWith('"')) ||
    (text.startsWith('“') && text.endsWith('”')) ||
    (text.startsWith('《') && text.endsWith('》')))
    ? text.slice(1, -1)
    : text;

const chunkIntoBatches = (lines: string[]): string[][] => {
  const batches: string[][] = [];
  let current: string[] = [];
  let chars = 0;
  for (const line of lines) {
    if (current.length >= MAX_LINES_PER_BATCH || chars + line.length > MAX_CHARS_PER_BATCH) {
      if (current.length) batches.push(current);
      current = [];
      chars = 0;
    }
    current.push(line);
    chars += line.length;
  }
  if (current.length) batches.push(current);
  return batches;
};

const callChatCompletions = async (
  config: OpenAITranslateConfig,
  system: string,
  user: string,
  useJsonMode: boolean,
  signal?: AbortSignal,
): Promise<string> => {
  const httpFetch = getAIFetch();
  const response = await httpFetch(chatCompletionsUrl(config.baseUrl), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${config.apiKey}`,
    },
    body: JSON.stringify({
      model: config.model,
      temperature: 0,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: user },
      ],
      ...(useJsonMode ? { response_format: { type: 'json_object' } } : {}),
    }),
    signal,
  });

  if (!response.ok) {
    const bodyText = await response.text().catch(() => '');
    // Older OpenAI-compatible servers reject response_format outright; retry
    // the same request without it before giving up.
    if (useJsonMode && response.status === 400 && /response_format/i.test(bodyText)) {
      return callChatCompletions(config, system, user, false, signal);
    }
    throw new Error(
      `Custom LLM translation failed with status ${response.status}: ${bodyText.slice(0, 200)}`,
    );
  }

  const data = (await response.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };
  const content = data?.choices?.[0]?.message?.content;
  if (typeof content !== 'string') {
    throw new Error('Custom LLM translation returned a malformed response');
  }
  return content;
};

const translateBatch = async (
  lines: string[],
  sourceLang: string,
  targetLang: string,
  config: OpenAITranslateConfig,
  signal?: AbortSignal,
): Promise<string[]> => {
  const system = buildSystemPrompt(sourceLang, targetLang, lines.length);
  const user = buildUserPrompt(lines);
  const content = await callChatCompletions(config, system, user, true, signal);
  const parsed = parseTranslations(content, lines.length);
  if (parsed) return parsed.map((t) => t || '');

  // The model broke the batch contract — fall back to per-line requests,
  // which cannot scramble ordering or drop lines.
  const systemOne = `You are a translation engine. Translate the user's text from ${sourceLang} into ${targetLang}. Output only the translation, no quotes, no commentary.`;
  const results: string[] = [];
  for (const line of lines) {
    if (!line?.trim()) {
      results.push(line);
      continue;
    }
    const single = await callChatCompletions(
      config,
      systemOne,
      line.replace(/\n/g, ' '),
      false,
      signal,
    );
    results.push(stripWrappingQuotes(single.trim()));
  }
  return results;
};

export const openaiProvider: TranslationProvider = {
  name: 'openai',
  label: _('Custom LLM (OpenAI Compatible)'),
  // Uses the user's own API key from the AI settings — no Readest login.
  authRequired: false,
  preservesMarkup: false,
  translate: async (
    text: string[],
    sourceLang: string,
    targetLang: string,
    _token?: string | null,
    _useCache: boolean = false,
    signal?: AbortSignal,
  ): Promise<string[]> => {
    const config = getOpenAITranslateConfig();
    if (!config.apiKey) {
      throw new Error(
        _('No API key configured. Set up an OpenAI-compatible provider in Settings → AI first.'),
      );
    }

    const source = langName(sourceLang);
    const target = langName(targetLang);

    // Empty lines pass through untouched, matching the other providers.
    const translatable: string[] = [];
    const indices: number[] = [];
    text.forEach((line, index) => {
      if (line?.trim()) {
        translatable.push(line);
        indices.push(index);
      }
    });
    if (translatable.length === 0) return text;

    const results = [...text];
    const batches = chunkIntoBatches(translatable);
    let offset = 0;
    for (const batch of batches) {
      const translated = await translateBatch(batch, source, target, config, signal);
      batch.forEach((original, i) => {
        results[indices[offset + i]!] = translated[i] || original;
      });
      offset += batch.length;
    }
    return results;
  },
};
