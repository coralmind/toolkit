/** Notices above a page's content. They share one style and stack in a fixed order. */

export const NOTICE_ORDER = ['stale', 'kl', 'report', 'link', 'interim', 'info'] as const;

export type NoticeKind = (typeof NOTICE_ORDER)[number];

export interface Notice {
  kind: NoticeKind;
  text: string;
}

export function linkNotice(text: string): Notice {
  return { kind: 'link', text };
}

export function ordered(items: Notice[]): Notice[] {
  return [...items].sort((a, b) => NOTICE_ORDER.indexOf(a.kind) - NOTICE_ORDER.indexOf(b.kind));
}
