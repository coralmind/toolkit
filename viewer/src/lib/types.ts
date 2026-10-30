/** The generated page data's types, and the small shapes built from them. */
export type * from './generated-types';

/** A URL query, one value per key, as the pages read it. */
export type Params = Record<string, string>;

/** Thrown by a view for an id that names nothing; the page turns it into a 404. */
export class NotFound extends Error {}
