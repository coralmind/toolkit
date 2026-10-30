/**
 * The words on the page: plain labels a clinician reviewer reads, and their definitions.
 * Knowledge Library text comes from the generated data, never from here.
 */

/** Outcome buckets in ledger order. Every transcript lands in exactly one. */
export const BUCKETS = [
  'judged',
  'not_tested',
  'invalid',
  'unchecked',
  'not_judged',
  'sim_failure',
  'unreadable',
] as const;

/** Run mode's default row order: problems first. */
export const PROBLEM_ORDER = [
  'unreadable',
  'sim_failure',
  'invalid',
  'unchecked',
  'not_judged',
  'not_tested',
  'judged',
] as const;

export const BUCKET_LABELS: Record<string, string> = {
  judged: 'judged',
  not_tested: 'not tested',
  invalid: 'invalid',
  unchecked: 'not checked',
  not_judged: 'not judged',
  sim_failure: 'simulation failed',
  unreadable: 'unreadable',
};

export const OUTCOME_LABELS: Record<string, string> = {
  fires: 'fired',
  partial: 'partly supported',
  not: 'did not fire',
  gate_unknown: 'condition unknown',
  failed: 'judge call failed',
  kl_mismatch: 'judged with a different Knowledge Library',
};

/** What a pathway firing means, by whether its Knowledge Library impacts are harmful or protective. */
export const DIRECTION_MEANINGS: Record<string, string> = {
  harmful: 'firing is the bad result',
  protective: 'firing is the good result',
  mixed: 'its impacts go both ways, so firing can be good or bad',
};

export const TERMS: Record<string, string> = {
  condition:
    "A fact the route's Knowledge Library entry requires, such as exposure treatment " +
    'underway. The route counts as fired only when the condition holds.',
};

export const FACT_MODE_UNRECORDED = 'not recorded';

const FAILURE_LABELS: Record<string, string> = {
  content_filter: 'the provider blocked the prompt',
  rate_limit: 'rate limited',
  timeout: 'timed out',
  empty_response: 'the system tested returned nothing',
  error: 'error',
};

/** Plain words for a simulation failure; an unknown reason is shown as recorded. */
export function failureLabel(reason: string): string {
  return FAILURE_LABELS[reason] ?? reason;
}

const ENDED_LABELS: Record<string, string> = {
  conversation_closing: 'natural closing',
  max_turns: 'reached the turn limit',
  person_echo: 'the simulated person copied the AI',
};

const ENDED_NOTES: Record<string, string> = {
  conversation_closing: 'the simulated person said they had nothing more to bring',
};

/** Plain words for why a conversation ended. */
export function endedLabel(exitReason: string): string {
  return ENDED_LABELS[exitReason] ?? exitReason.replaceAll('_', ' ');
}

/** What an ending means, where its label alone does not say; `''` when it does. */
export function endedNote(exitReason: string): string {
  return ENDED_NOTES[exitReason] ?? '';
}

export function preemptedLabel(key: string): string {
  return (
    { resolved: 'an earlier AI reply met the need first', derailed: 'the AI misread the person' } as Record<string, string>
  )[key] ?? key;
}

export function gateWord(verdict: string): string {
  return (
    { holds: 'held', does_not_hold: 'did not hold', unknown: 'unknown' } as Record<string, string>
  )[verdict] ?? verdict;
}

/** A route's condition in words, from the judge's ``gate_status``. */
export function conditionLabel(gateStatus: string | null): string {
  const label: Record<string, string> = {
    no_gates: 'no condition',
    met: 'condition held',
    not_met: 'condition did not hold',
    unknown: 'condition unknown',
  };
  return gateStatus ? (label[gateStatus] ?? gateStatus) : '';
}

export const CUE_READING: Record<string, string> = {
  supports: 'read as held',
  contradicts: 'read as not held',
};

export const BASIS: Record<string, string> = {
  test_case: 'decided by the test case record',
  conversation: "decided by the judge's reading of the transcript",
  none: "neither the test case record nor the judge's reading of the transcript decides it",
};

export const SENT_TERM = 'given to the simulated person';
export const SENT_DEFINITION =
  'Sent to the model that plays the person, in its instructions on every turn; the opening ' +
  'message is sent as its first message. Everything else here shaped the case, or is read ' +
  'by the fidelity check and the judge.';

export const RUN_SORTS: Record<string, string> = {
  problems: 'problems first',
  case: 'case order',
};

export function times(n: number): string {
  return ({ 1: 'once', 2: 'twice' } as Record<number, string>)[n] ?? `${n} times`;
}

/**
 * A fidelity reason in words. ``detail`` is a behavior id when the expected behavior did not
 * occur as its Knowledge Library entry defines it; ``name`` is its display id, when known, and
 * ``missing`` its parts that were not present, each as its letter and result ("(b) ruled out").
 */
export function fidelityReasonText(category: string, detail: string, name = '', missing: string[] = []): string {
  if (category === 'simulator' && detail.startsWith('kl-')) {
    const did = `${name || detail} did not occur`;
    if (!missing.length) return did;
    return `${did}: ${missing.length === 1 ? 'part' : 'parts'} ${missing.join(', ')}`;
  }
  return category === 'simulator' ? detail : `${category}: ${detail}`;
}

/** Whether every lettered part of a behavior is needed or any one counts; "" when unknown. */
export function partsNeededLabel(needed: number, total: number): string {
  if (!needed || total < 2) return '';
  if (needed === 1) return 'Any one part counts.';
  return needed >= total ? `All ${total} parts are needed.` : `${needed} of ${total} parts are needed.`;
}

/** Why the judge did not assess the target pathway, from Stage 2's skip reason. */
export function targetSkipText(reason: string): string {
  if (reason.startsWith('no participant behavior observed')) {
    return "The judge found none of this pathway's behaviors in the transcript, so it did not assess the pathway, and the pathway did not fire.";
  }
  if (reason === 'absent from the judge output') return 'The judge returned nothing for this pathway.';
  return reason;
}

/** What the judge's Stage 1 made of one behavior of a target pathway it did not assess. */
export function targetPartTag(part: { seen: boolean; failed: boolean; reason: string }): string {
  if (part.seen) return 'seen by the judge';
  if (part.failed) return "the judge's call failed";
  return part.reason ? 'ruled out by the judge' : 'not seen by the judge';
}

export function levelLabel(level: number | null | undefined): string {
  return level ? `L${level}` : '';
}

/** The fact mode to show, or "" when the run did not record one. */
export function recorded(mode: string): string {
  return mode === FACT_MODE_UNRECORDED ? '' : mode;
}

export function words(code: string | null | undefined): string {
  return (code ?? '').replaceAll('_', ' ');
}
