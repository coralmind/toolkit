/**
 * Crawler policy, the same as the Knowledge Library site's: no indexing, no archiving, no AI
 * training.
 *
 * `noai` and `noimageai` are not part of any standard; crawlers that honor them do so
 * voluntarily. The user-agent block in `src/proxy.ts` is what stops a self-identified AI
 * crawler, and only one that identifies itself.
 */

export const ROBOTS_DIRECTIVES =
  'noindex, nofollow, noarchive, nosnippet, noimageindex, notranslate, noai, noimageai';

/** Substrings of the user agents that AI crawlers and AI fetchers send. Matched case-insensitively. */
export const AI_USER_AGENTS = [
  'AI2Bot',
  'Ai2Bot-Dolma',
  'Amazonbot',
  'anthropic-ai',
  'Bytespider',
  'CCBot',
  'ChatGPT-User',
  'Claude-SearchBot',
  'Claude-User',
  'Claude-Web',
  'ClaudeBot',
  'cohere-ai',
  'cohere-training-data-crawler',
  'Diffbot',
  'DuckAssistBot',
  'FacebookBot',
  'FirecrawlAgent',
  'Google-CloudVertexBot',
  'GoogleOther',
  'GPTBot',
  'ImagesiftBot',
  'img2dataset',
  'Kangaroo Bot',
  'meta-externalagent',
  'meta-externalfetcher',
  'MistralAI-User',
  'OAI-SearchBot',
  'omgili',
  'PanguBot',
  'PerplexityBot',
  'Perplexity-User',
  'PetalBot',
  'Scrapy',
  'Timpibot',
  'VelenPublicWebCrawler',
  'webzio-extended',
  'YouBot',
] as const;

const AI_USER_AGENT_PATTERN = new RegExp(
  AI_USER_AGENTS.map((ua) => ua.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'),
  'i',
);

export function isAiUserAgent(userAgent: string | null): boolean {
  return userAgent !== null && AI_USER_AGENT_PATTERN.test(userAgent);
}
