'use client';

/** The last resort when the layout itself fails: no brand styles, plain words. */
export default function GlobalError({ reset }: { error: Error; reset: () => void }) {
  return (
    <html lang="en">
      <body>
        <h1>Could not show this page</h1>
        <p>Something went wrong while reading these results.</p>
        <p>
          <button type="button" onClick={reset}>
            Try again
          </button>
        </p>
      </body>
    </html>
  );
}
