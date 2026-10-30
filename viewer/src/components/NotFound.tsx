/** The page for a link that names nothing: an unknown id, an old link, or a mistyped one. */
export function NotFoundBody() {
  return (
    <>
      <h1>Not found</h1>
      <p className="vw-lede">Nothing with that id in these results.</p>
      <p>
        Start from the <a href="/">Introduction</a>, the <a href="/suites">Suites</a> or the{' '}
        <a href="/runs">Runs</a>.
      </p>
    </>
  );
}
