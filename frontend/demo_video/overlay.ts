/**
 * Browser-side helpers for the demo recording, installed with `page.addInitScript`.
 *
 * Everything here lives outside `#root`, so the app itself is untouched:
 *   - a visible cursor that follows real mouse events, with a click ripple;
 *   - a smooth zoom: a CSS transform on `#root`, re-rasterized by the browser,
 *     so zoomed text stays sharp (unlike zooming a recorded video afterwards);
 *   - a caption bar and full-screen title cards, which never zoom.
 */
export const overlayScript = `(() => {
  const W = () => window.innerWidth, H = () => window.innerHeight;
  const state = { s: 1, tx: 0, ty: 0 };

  function install() {
    if (document.getElementById("demo-overlay")) return;
    const style = document.createElement("style");
    style.textContent = \`
      #root { transform-origin: 0 0; transition: transform 1.1s cubic-bezier(.65,0,.35,1); will-change: transform; }
      #demo-overlay { direction: ltr; position: fixed; inset: 0; pointer-events: none; z-index: 2147483647; font-family: "IBM Plex Sans", system-ui, sans-serif; }
      #demo-cursor { position: absolute; left: 0; top: 0; width: 30px; height: 30px; transform: translate(-100px,-100px); filter: drop-shadow(0 2px 3px rgba(0,0,0,.35)); }
      .demo-ripple { position: absolute; width: 18px; height: 18px; margin: -9px 0 0 -9px; border-radius: 50%;
        border: 3px solid #0f766e; animation: demo-ripple .55s ease-out forwards; }
      @keyframes demo-ripple { from { transform: scale(.4); opacity: .9 } to { transform: scale(2.6); opacity: 0 } }
      #demo-caption { position: absolute; left: 50%; bottom: 44px; transform: translate(-50%, 16px); opacity: 0;
        max-width: 1500px; padding: 18px 30px; border-radius: 14px; background: rgba(17,24,24,.92); color: #fff;
        font-size: 30px; line-height: 1.35; font-weight: 500; text-align: center; letter-spacing: -.005em;
        box-shadow: 0 10px 40px rgba(0,0,0,.25); transition: opacity .45s ease, transform .45s ease; }
      #demo-caption.on { opacity: 1; transform: translate(-50%, 0); }
      #demo-caption .kicker { display: block; font-size: 17px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: #5eead4; margin-bottom: 6px; }
      #demo-card { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 22px;
        background: #111818; color: #fff; opacity: 0; transition: opacity .7s ease; text-align: center; }
      #demo-card.on { opacity: 1; }
      #demo-card .title { margin: 0; font-size: 104px; font-weight: 600; letter-spacing: -.02em; }
      #demo-card .title .ar { color: #5eead4; font-family: "IBM Plex Sans Arabic", sans-serif; margin-inline-start: 26px; }
      #demo-card p { margin: 0; font-size: 36px; color: #d4d4d4; max-width: 1400px; line-height: 1.4; }
      #demo-card .small { font-size: 26px; color: #a3a3a3; font-family: "IBM Plex Mono", monospace; }
      #demo-card .chips { display: flex; gap: 14px; margin-top: 8px; }
      #demo-card .chips span { font-size: 22px; padding: 8px 18px; border-radius: 999px; border: 1px solid #3f4a4a; color: #e5e5e5; }
    \`;
    document.head.appendChild(style);
    const overlay = document.createElement("div");
    overlay.id = "demo-overlay";
    overlay.innerHTML = \`
      <svg id="demo-cursor" viewBox="0 0 24 24"><path d="M4 2.5 L4 19 L8.6 14.9 L11.6 21.6 L14.6 20.3 L11.7 13.7 L18 13.4 Z" fill="#111" stroke="#fff" stroke-width="1.5" stroke-linejoin="round"/></svg>
      <div id="demo-caption"></div>
      <div id="demo-card"></div>\`;
    document.documentElement.appendChild(overlay);
    const cursor = overlay.querySelector("#demo-cursor");
    window.addEventListener("mousemove", (e) => { cursor.style.transform = \`translate(\${e.clientX - 5}px,\${e.clientY - 3}px)\`; }, true);
    window.addEventListener("mousedown", (e) => {
      const r = document.createElement("div");
      r.className = "demo-ripple"; r.style.left = e.clientX + "px"; r.style.top = e.clientY + "px";
      overlay.appendChild(r); setTimeout(() => r.remove(), 600);
    }, true);
  }

  function apply() {
    const root = document.getElementById("root");
    // The transform is in #root's own coordinates; correct for the page scroll so
    // the zoom is anchored to what is on screen, not to the top of the document.
    const ty = state.ty - (state.s - 1) * window.scrollY;
    const tx = state.tx - (state.s - 1) * window.scrollX;
    root.style.transform = state.s === 1 ? "none" : \`translate(\${tx}px,\${ty}px) scale(\${state.s})\`;
  }

  window.__demo = {
    install,
    /** Zoom so the element (CSS selector or rect) fills ~fill of the screen, capped at maxScale. */
    zoomTo(rect, { maxScale = 1.8, fill = 0.82, focusY = 0.47 } = {}) {
      install();
      // rect is measured on screen; undo the current zoom to get unzoomed coordinates.
      const ux = (rect.x - state.tx) / state.s, uy = (rect.y - state.ty) / state.s;
      const uw = rect.width / state.s, uh = rect.height / state.s;
      const s = Math.max(1, Math.min(maxScale, (W() * fill) / uw, (H() * fill) / uh));
      const cx = ux + uw / 2, cy = uy + uh / 2;
      let tx = W() / 2 - s * cx, ty = H() * focusY - s * cy;
      tx = Math.min(0, Math.max(W() * (1 - s), tx));
      ty = Math.min(0, Math.max(H() * (1 - s), ty));
      Object.assign(state, { s, tx, ty });
      apply();
      return s;
    },
    zoomOut() { Object.assign(state, { s: 1, tx: 0, ty: 0 }); apply(); },
    zoom() { return state.s; },
    caption(html, kicker) {
      install();
      const el = document.getElementById("demo-caption");
      if (!html) { el.classList.remove("on"); return; }
      const show = () => { el.innerHTML = (kicker ? \`<span class="kicker">\${kicker}</span>\` : "") + html; el.classList.add("on"); };
      if (el.classList.contains("on")) { el.classList.remove("on"); setTimeout(show, 380); } else show();
    },
    card(html) {
      install();
      const el = document.getElementById("demo-card");
      if (!html) { el.classList.remove("on"); return; }
      el.innerHTML = html; el.classList.add("on");
    },
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", install); else install();
})();`;
