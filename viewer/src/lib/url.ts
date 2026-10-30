/** Links between pages. Every view state is a URL, so every control is a plain link. */

export type QueryValue = string | number | null | undefined;

/** A link. Empty values are left out, so a cleared filter drops from the URL. */
export function url(path: string, params: Record<string, QueryValue> = {}, fragment = ''): string {
  const pairs = Object.entries(params).filter(
    (entry): entry is [string, string | number] => entry[1] !== null && entry[1] !== undefined && entry[1] !== '',
  );
  const query = new URLSearchParams(pairs.map(([k, v]) => [k, String(v)])).toString();
  return `${path}${query ? `?${query}` : ''}${fragment ? `#${fragment}` : ''}`;
}

/** One value per query key: the first of a repeated key. Empty values are dropped. */
export function firstValues(
  searchParams: Record<string, string | string[] | undefined>,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(searchParams)) {
    const first = Array.isArray(value) ? value[0] : value;
    if (first) out[key] = first;
  }
  return out;
}
