import { describe, test, expect, vi, beforeEach } from 'vitest';
import { useSettingsStore } from '@/store/settingsStore';

vi.mock('@/services/environment', () => ({
  isTauriAppPlatform: vi.fn(() => false),
  getAPIBaseUrl: vi.fn(() => 'https://api.example.com'),
}));

vi.mock('@/utils/misc', () => ({
  stubTranslation: (s: string) => s,
}));

vi.mock('@/utils/supabase', () => ({
  supabase: {
    auth: { getSession: vi.fn().mockResolvedValue({ data: { session: null } }) },
    from: vi.fn(),
  },
}));

const mockFetch = vi.fn();
vi.mock('@/services/ai/utils/httpFetch', () => ({
  getAIFetch: () => mockFetch,
}));

import { openaiProvider, getOpenAITranslateConfig } from '@/services/translators/providers/openai';

const setAIConfig = (config: {
  openrouterApiKey?: string;
  openrouterBaseUrl?: string;
  openrouterModel?: string;
}) => {
  useSettingsStore.setState({
    settings: {
      version: 1,
      aiSettings: { enabled: false, provider: 'openrouter', ...config },
    } as never,
  });
};

const chatResponse = (content: string) => ({
  ok: true,
  status: 200,
  json: async () => ({ choices: [{ message: { content } }] }),
  text: async () => content,
});

describe('getOpenAITranslateConfig', () => {
  beforeEach(() => {
    mockFetch.mockReset();
    setAIConfig({});
  });

  test('falls back to defaults when nothing is configured', () => {
    const config = getOpenAITranslateConfig();
    expect(config.baseUrl).toBe('https://api.openai.com/v1');
    expect(config.model).toBe('gpt-4o-mini');
    expect(config.apiKey).toBe('');
  });

  test('trims trailing slashes and keeps a custom /v1 base URL', () => {
    setAIConfig({ openrouterBaseUrl: 'https://api.deepseek.com/v1/' });
    expect(getOpenAITranslateConfig().baseUrl).toBe('https://api.deepseek.com/v1');
  });
});

describe('openaiProvider.translate', () => {
  beforeEach(() => {
    mockFetch.mockReset();
    setAIConfig({
      openrouterApiKey: 'sk-test',
      openrouterBaseUrl: 'https://my.llm.example/v1',
      openrouterModel: 'test-model',
    });
  });

  test('passes empty lines through and sends one JSON-mode batch request', async () => {
    mockFetch.mockResolvedValue(chatResponse('{"translations": ["你好", "再见"]}'));

    const results = await openaiProvider.translate(['hello', '', 'goodbye'], 'en', 'zh-CN');

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, init] = mockFetch.mock.calls[0]!;
    expect(url).toBe('https://my.llm.example/v1/chat/completions');
    const body = JSON.parse(init.body);
    expect(body.model).toBe('test-model');
    expect(body.response_format).toEqual({ type: 'json_object' });
    expect(init.headers.Authorization).toBe('Bearer sk-test');
    expect(body.messages[0].role).toBe('system');
    // The empty line never reaches the model.
    expect(body.messages[1].content).toContain('1. hello');
    expect(body.messages[1].content).toContain('2. goodbye');
    expect(results).toEqual(['你好', '', '再见']);
  });

  test('splits large inputs into batches and maps results back in order', async () => {
    const lines = Array.from({ length: 45 }, (_, i) => `line ${i + 1}`);
    mockFetch.mockImplementation(async (_url, init) => {
      const body = JSON.parse(init.body);
      const user = body.messages[1].content as string;
      return chatResponse(
        JSON.stringify({
          translations: user.split('\n').map((l: string) => `译:${l.replace(/^\d+\.\s*/, '')}`),
        }),
      );
    });

    const results = await openaiProvider.translate(lines, 'en', 'zh-CN');

    expect(mockFetch).toHaveBeenCalledTimes(3); // 20 + 20 + 5
    expect(results).toHaveLength(45);
    expect(results[0]).toBe('译:line 1');
    expect(results[44]).toBe('译:line 45');
  });

  test('falls back to per-line requests when the model breaks the JSON contract', async () => {
    mockFetch
      // batch request: prose garbage without valid JSON
      .mockResolvedValueOnce(chatResponse('I cannot do that.'))
      // per-line fallbacks
      .mockResolvedValueOnce(chatResponse('你好'))
      .mockResolvedValueOnce(chatResponse('再见'));

    const results = await openaiProvider.translate(['hello', 'goodbye'], 'en', 'zh-CN');

    expect(mockFetch).toHaveBeenCalledTimes(3);
    expect(results).toEqual(['你好', '再见']);
  });

  test('retries without response_format when the server rejects it', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: false,
        status: 400,
        text: async () => 'Unknown field: response_format',
      })
      .mockResolvedValueOnce(chatResponse('{"translations": ["你好"]}'));

    const results = await openaiProvider.translate(['hello'], 'en', 'zh-CN');

    expect(mockFetch).toHaveBeenCalledTimes(2);
    expect(JSON.parse(mockFetch.mock.calls[0]![1].body).response_format).toBeDefined();
    expect(JSON.parse(mockFetch.mock.calls[1]![1].body).response_format).toBeUndefined();
    expect(results).toEqual(['你好']);
  });

  test('throws a descriptive error when no API key is configured', async () => {
    setAIConfig({});
    await expect(openaiProvider.translate(['hello'], 'en', 'zh-CN')).rejects.toThrow(
      /No API key configured/,
    );
    expect(mockFetch).not.toHaveBeenCalled();
  });
});
