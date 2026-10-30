/** The brand footer: the copyright holder and the content's license. */
import logo from '@coralmind/brand/assets/logo/primary-color.svg';

const CONTENT_LICENSE_URL = 'https://creativecommons.org/licenses/by-nc/4.0/';

export function SiteFooter() {
  return (
    <footer className="cm-site-footer site-footer">
      <div className="cm-site-footer__inner">
        <a className="cm-site-footer__brand" href="/">
          <img src={logo} alt="CoralMind" width="150" height="36" />
        </a>
        <div className="cm-site-footer__legal">
          <p className="site-footer__copyright">
            © {new Date().getFullYear()} CoralMind Institute · Content licensed{' '}
            <a href={CONTENT_LICENSE_URL} rel="license noopener noreferrer" target="_blank">
              CC BY-NC 4.0
            </a>
          </p>
        </div>
      </div>
    </footer>
  );
}
