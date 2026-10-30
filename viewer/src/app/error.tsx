'use client';

/** A page that failed to render. Plain words; the server log has the error. */
export default function PageError({ reset }: { error: Error; reset: () => void }) {
  return (
    <main id="main" className="vw-main">
      <h1>Could not show this page</h1>
      <p className="vw-lede">Something went wrong while reading these results.</p>
      <p>
        <button type="button" className="vw-button" onClick={reset}>
          Try again
        </button>{' '}
        or go to the <a href="/suites">Suites</a>.
      </p>
    </main>
  );
}
