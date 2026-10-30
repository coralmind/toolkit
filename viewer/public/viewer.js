/*
 * Progressive enhancement for the results viewer. Every control is a working link
 * without this file; it adds quote highlighting, the strip's start position, history
 * replacement on previous/next, the rail's tablist, and records and judge calls loaded
 * when opened.
 */
(function () {
  'use strict';

  // ---------- Scrollbars show while an area scrolls ----------

  var scrollTimers = new WeakMap();
  document.addEventListener('scroll', function (event) {
    var el = event.target === document ? document.documentElement : event.target;
    if (!el.classList) return;
    el.classList.add('is-scrolling');
    clearTimeout(scrollTimers.get(el));
    scrollTimers.set(el, setTimeout(function () { el.classList.remove('is-scrolling'); }, 800));
  }, true);

  // ---------- Quote highlighting ----------

  function contentFor(turn, role) {
    var ids = role ? ['turn-' + turn + '-' + role] : ['turn-' + turn + '-user', 'turn-' + turn + '-assistant'];
    var found = [];
    ids.forEach(function (id) {
      var el = document.getElementById(id);
      var ct = el && el.querySelector('.content-text');
      if (ct) found.push(ct);
    });
    return found;
  }

  // A view of `orig` without inline markdown markers, tags and repeated spaces, with a map
  // from each kept character back to its index in `orig`.
  function normalize(orig) {
    var norm = '';
    var map = [];
    var space = false;
    for (var i = 0; i < orig.length; i++) {
      var two = orig.substr(i, 2);
      if (two === '**' || two === '__') { i += 1; continue; }
      var ch = orig.charAt(i);
      if (ch === '*' || ch === '_' || ch === '`') continue;
      if (ch === '<') {
        var close = orig.indexOf('>', i);
        if (close === -1) break;
        i = close;
        continue;
      }
      if (/\s/.test(ch)) {
        if (!space) { norm += ' '; map.push(i); space = true; }
        continue;
      }
      norm += ch;
      map.push(i);
      space = false;
    }
    return { norm: norm, map: map };
  }

  // innerHTML escapes only &, < and > in text, so a quote is escaped the same way to match.
  function escapeHtml(text) {
    return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function light(ct, quote) {
    if (ct.getAttribute('data-orig') !== null) return true;
    var orig = ct.innerHTML;
    var esc = escapeHtml(quote);
    var start = orig.indexOf(esc);
    var end;
    if (start !== -1) {
      end = start + esc.length;
    } else {
      var on = normalize(orig);
      var qn = normalize(esc).norm.trim();
      if (!qn) return false;
      var at = on.norm.indexOf(qn);
      if (at === -1) return false;
      start = on.map[at];
      end = on.map[at + qn.length - 1] + 1;
    }
    ct.setAttribute('data-orig', orig);
    ct.innerHTML = orig.slice(0, start) + '<mark class="quote-hl">' + orig.slice(start, end) +
      '</mark>' + orig.slice(end);
    return true;
  }

  function clearAll() {
    document.querySelectorAll('.content-text[data-orig]').forEach(function (ct) {
      ct.innerHTML = ct.getAttribute('data-orig');
      ct.removeAttribute('data-orig');
    });
    document.querySelectorAll('.vw-turn.is-outlined').forEach(function (el) {
      el.classList.remove('is-outlined');
    });
  }

  function highlight(link) {
    clearAll();
    var turn = link.getAttribute('data-turn');
    var quote = link.getAttribute('data-quote') || '';
    var role = link.getAttribute('data-role') || '';
    var lit = false;
    if (link.getAttribute('data-found') !== '0') {
      contentFor(turn, role).forEach(function (ct) { lit = light(ct, quote) || lit; });
    }
    if (!lit) {
      var block = document.getElementById('turn-' + turn);
      if (block) block.classList.add('is-outlined');
    }
  }

  document.querySelectorAll('.vw-cite').forEach(function (link) {
    link.addEventListener('mouseenter', function () { highlight(link); });
    link.addEventListener('focus', function () { highlight(link); });
    link.addEventListener('click', function () { highlight(link); });
  });

  // ---------- Turn strip ----------

  function hashTurn() {
    var m = /^#turn-(\d+)/.exec(window.location.hash);
    return m ? m[1] : null;
  }

  document.querySelectorAll('.vw-strip').forEach(function (strip) {
    var frame = strip.querySelector('.vw-strip__frame');
    var first = strip.querySelector('.vw-strip__cell');
    if (!frame || !first) return;
    if (first.getBoundingClientRect().width < 24) strip.classList.add('is-narrow');
    else strip.classList.remove('is-narrow');
    var start = hashTurn() || strip.getAttribute('data-start') || '1';
    var cell = strip.querySelector('.vw-strip__cell[data-turn="' + start + '"]');
    if (cell) frame.scrollLeft = Math.max(0, cell.offsetLeft - frame.clientWidth / 3);
  });

  var initial = hashTurn();
  if (initial) {
    var block = document.getElementById('turn-' + initial);
    if (block) block.classList.add('is-outlined');
  }

  // ---------- Previous / next replace the history entry ----------

  document.querySelectorAll('.vw-step').forEach(function (link) {
    link.addEventListener('click', function (event) {
      if (event.metaKey || event.ctrlKey || event.shiftKey) return;
      event.preventDefault();
      window.location.replace(link.href);
    });
  });

  // ---------- Rail tabs ----------
  // Without this file the rail is four headed sections with jump links. Here it becomes an
  // ARIA tablist: one panel at a time, arrow keys, Home and End.

  var rail = document.querySelector('.vw-case__rail');
  var railNav = rail && rail.querySelector('.vw-railtabs');
  if (rail && railNav) {
    var tabs = Array.prototype.slice.call(railNav.querySelectorAll('a[data-tab]'));
    var panels = rail.querySelectorAll('.vw-rail__panel');
    railNav.setAttribute('role', 'tablist');
    tabs.forEach(function (t) {
      var key = t.getAttribute('data-tab');
      t.setAttribute('role', 'tab');
      t.setAttribute('aria-controls', 'rail-' + key);
      var panel = document.getElementById('rail-' + key);
      if (panel) {
        panel.setAttribute('role', 'tabpanel');
        panel.setAttribute('aria-labelledby', t.id);
        panel.removeAttribute('aria-label');
      }
    });
    var show = function (key, focus) {
      panels.forEach(function (p) { p.classList.toggle('is-active', p.id === 'rail-' + key); });
      tabs.forEach(function (t) {
        var on = t.getAttribute('data-tab') === key;
        t.setAttribute('aria-selected', on ? 'true' : 'false');
        t.setAttribute('tabindex', on ? '0' : '-1');
        if (on && focus) t.focus();
      });
    };
    rail.classList.add('vw-rail--tabbed');
    var named = /^#rail-(fidelity|pathways|behaviors|records)$/.exec(window.location.hash);
    show(named ? named[1] : 'pathways', false);
    tabs.forEach(function (t, i) {
      t.addEventListener('click', function (event) {
        event.preventDefault();
        show(t.getAttribute('data-tab'), false);
      });
      t.addEventListener('keydown', function (event) {
        var to = null;
        if (event.key === 'ArrowRight') to = tabs[(i + 1) % tabs.length];
        else if (event.key === 'ArrowLeft') to = tabs[(i - 1 + tabs.length) % tabs.length];
        else if (event.key === 'Home') to = tabs[0];
        else if (event.key === 'End') to = tabs[tabs.length - 1];
        if (!to) return;
        event.preventDefault();
        show(to.getAttribute('data-tab'), true);
      });
    });
  }

  var left = document.querySelector('.vw-case__left');
  if (left) {
    var narrow = window.matchMedia('(max-width: 1100px)');
    left.open = !narrow.matches;
    if (narrow.addEventListener) {
      narrow.addEventListener('change', function () { left.open = !narrow.matches; });
    }
  }

  // ---------- Judge calls load when opened ----------

  document.querySelectorAll('details.vw-call').forEach(function (call) {
    call.addEventListener('toggle', function () {
      if (!call.open || call.getAttribute('data-loaded')) return;
      call.setAttribute('data-loaded', '1');
      [['data-prompt', '.vw-call__prompt'], ['data-response', '.vw-call__response']].forEach(function (pair) {
        var pre = call.querySelector(pair[1]);
        fetch(call.getAttribute(pair[0])).then(function (r) {
          if (!r.ok) throw new Error('status ' + r.status);
          return r.text();
        }).then(function (text) {
          pre.textContent = text;
          pre.hidden = false;
        }).catch(function () {
          pre.textContent = 'Could not load this. Close and reopen to try again.';
          pre.hidden = false;
          call.removeAttribute('data-loaded');
        });
      });
    });
  });

  // ---------- Raw records load when opened ----------

  document.querySelectorAll('details.vw-record[data-src]').forEach(function (block) {
    block.addEventListener('toggle', function () {
      if (!block.open || block.getAttribute('data-loaded')) return;
      block.setAttribute('data-loaded', '1');
      var pre = block.querySelector('.vw-record__text');
      pre.textContent = 'Loading…';
      pre.hidden = false;
      fetch(block.getAttribute('data-src')).then(function (r) {
        if (!r.ok) throw new Error('status ' + r.status);
        return r.text();
      }).then(function (text) {
        pre.textContent = text;
      }).catch(function () {
        pre.textContent = 'Could not load this. Close and reopen to try again.';
        block.removeAttribute('data-loaded');
      });
    });
  });

})();
