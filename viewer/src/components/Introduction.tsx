/** The site's first page: what the lab is, how an evaluation runs, and how to read results. */
import { KL_SITE } from './bits';
import { Pipeline } from './Pipeline';

export function Introduction() {
  return (
    <div className="vw-intro">
      <section className="vw-hero">
        <h1>How AI chat assistants respond to people in mental-health situations, tested one conversation at a time</h1>
        <p>
          We test how AI chat assistants respond to people in mental-health situations. The situations come from the
          CoralMind Knowledge Library. In each test, a language model plays a person and talks with the AI system we
          are testing. A second model checks that the simulated person did what the test asked. A judge then reads the
          transcript and records what the AI did, using the library&apos;s definitions.
        </p>
        <p>
          This site shares sample results. You can open every transcript and see how each judgment was made. No real
          person took part in any of these conversations.
        </p>
        <p className="vw-intro__explore">
          <a className="cm-btn" href="/suites">
            Explore suites
          </a>
        </p>
      </section>

      <section>
        <h2>The evaluation pipeline</h2>
        <p>Every evaluation follows the same steps.</p>
        <Pipeline />

        <h3>Two files start an evaluation</h3>
        <p>
          The test case file holds the tests. Each test case names a pathway from the Knowledge Library, the person to
          simulate, what that person says and does, and the behavior they must show for the test to count. One test
          case file is one suite.
        </p>
        <p>
          The manifest picks a model for each role and sets how long a conversation can run. Each manifest makes one
          run. Running the same test case file with different manifests is how we compare AI systems on the same tests.
        </p>

        <h3>Four models, four roles</h3>
        <ul>
          <li>The system tested is the AI we are evaluating.</li>
          <li>The simulator plays the person.</li>
          <li>
            The fidelity checker reads each finished transcript and checks that the simulated person did their part. If
            they did not, the transcript is set aside, so no system is scored on a test that did not happen.
          </li>
          <li>
            The judge reads each valid transcript in two passes. First it marks the behaviors in each turn. Then it
            decides which pathways those behaviors add up to.
          </li>
        </ul>
        <p>We use a different model for each role, so no model grades its own work.</p>

        <h3>Every step keeps a record</h3>
        <p>
          Each step saves its results with a record of what it ran: its inputs, its models, and any setting that
          differed from the manifest. We can check or judge a run again without running the conversations again, and
          the earlier results stay available.
        </p>

        <h3>Where the Knowledge Library comes in</h3>
        <p>
          The Knowledge Library says what to look for: the behaviors, the pathways they form, the context a pathway
          depends on, and the impacts it can lead to. The CoralMind Toolkit runs the tests. To see what the Toolkit
          takes from the library at each step, read{' '}
          <a href={`${KL_SITE}/guide/evaluation`} target="_blank" rel="noopener noreferrer">
            How an evaluation uses the Knowledge Library
          </a>{' '}
          in the library&apos;s guide.
        </p>
      </section>

      <section>
        <h2>How to read the results</h2>
        <p>
          Each test case is built to test one pathway, which we call its route. A route fires when the judge finds the
          pathway&apos;s behaviors in a transcript and every condition the pathway needs is met. This is a judgment
          against a definition. It does not mean harm happened, and a sample run cannot tell you how often a system
          behaves this way with real people.
        </p>
        <p>
          Rates count only the transcripts that were judged. Transcripts that could not be judged are shown next to
          each rate as left out. Every number opens the transcripts behind it. Each transcript shows its turns, the
          judge&apos;s reasoning with the quotes it used, and the fidelity check.
        </p>
      </section>
    </div>
  );
}
