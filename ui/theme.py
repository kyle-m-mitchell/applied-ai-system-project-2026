"""Self-contained editorial listening-room visual system."""

from __future__ import annotations

import streamlit as st


CSS = r"""
<style>
:root {
  --cadence-bg: #0D0E10;
  --cadence-panel: #15171A;
  --cadence-raised: #1C1F23;
  --cadence-border: #2B3036;
  --cadence-ink: #F3EFE7;
  --cadence-muted: #AAA49A;
  --cadence-amber: #E4A24B;
  --cadence-blue: #82AFFF;
  --cadence-lavender: #BEA7E5;
  --cadence-green: #74C69D;
  --cadence-red: #D98484;
  --cadence-radius: 14px;
}

.stApp {
  background:
    radial-gradient(900px 520px at 9% -8%, rgba(228,162,75,.12), transparent 62%),
    radial-gradient(760px 500px at 96% 15%, rgba(130,175,255,.08), transparent 58%),
    var(--cadence-bg);
  color: var(--cadence-ink);
}

.block-container { max-width: 1220px; padding-top: 2.25rem; padding-bottom: 4rem; }
h1, h2, h3 { letter-spacing: -.025em; }
h1, .cadence-wordmark { font-family: Iowan Old Style, Baskerville, Georgia, serif; }

.cadence-brand { display:flex; align-items:center; gap:16px; margin: 0 0 4px; }
.cadence-wordmark { font-size: clamp(2.35rem, 6vw, 4.8rem); line-height:.9; color:var(--cadence-ink); margin:0; }
.cadence-kicker { color:var(--cadence-amber); font-size:.72rem; letter-spacing:.19em; text-transform:uppercase; font-weight:700; }
.cadence-deck { color:var(--cadence-muted); font-size:clamp(1rem, 2vw, 1.18rem); max-width:720px; line-height:1.55; }
.cadence-compact .cadence-wordmark { font-size:2.4rem; }
.cadence-brand.cadence-compact + .cadence-deck { font-size:.95rem; }
.cadence-paused .cadence-wave span { animation:none; transform:scaleY(.55); opacity:.55; }

.cadence-wave { display:flex; align-items:center; gap:3px; height:34px; padding-left:4px; }
.cadence-wave span { display:block; width:3px; border-radius:99px; background:var(--cadence-amber); animation:cadence-pulse 1.8s ease-in-out infinite; opacity:.9; }
.cadence-wave span:nth-child(1){height:10px;animation-delay:-.4s}.cadence-wave span:nth-child(2){height:22px;animation-delay:-.9s}
.cadence-wave span:nth-child(3){height:31px;animation-delay:-.2s}.cadence-wave span:nth-child(4){height:17px;animation-delay:-1.1s}
.cadence-wave span:nth-child(5){height:26px;animation-delay:-.65s}.cadence-wave span:nth-child(6){height:12px;animation-delay:-1.35s}
@keyframes cadence-pulse { 0%,100%{transform:scaleY(.5);opacity:.45} 50%{transform:scaleY(1);opacity:1} }

.cadence-framing { font-family:Iowan Old Style,Baskerville,Georgia,serif; font-size:1.35rem; line-height:1.45; color:var(--cadence-ink); padding:.3rem 0 1.2rem; }
.cadence-eyebrow { color:var(--cadence-amber); font-size:.72rem; line-height:1.35; letter-spacing:.16em; text-transform:uppercase; font-weight:750; margin:.65rem 0 .3rem; }
.cadence-help { color:var(--cadence-muted); font-size:.82rem; line-height:1.45; }
.cadence-disclosure { border:1px solid var(--cadence-border); border-radius:12px; padding:14px 16px; color:var(--cadence-muted); font-size:.84rem; background:rgba(21,23,26,.72); }

[class*="st-key-track_card_"] { background:linear-gradient(155deg,rgba(28,31,35,.95),rgba(18,20,23,.95)); border-color:var(--cadence-border)!important; transition:transform .18s ease,border-color .18s ease; }
[class*="st-key-track_card_"]:hover { transform:translateY(-2px); border-color:#4A4F56!important; }
.cadence-track-head { display:flex; gap:14px; align-items:flex-start; }
.cadence-cover { flex:0 0 66px; width:66px; height:66px; border-radius:11px; display:grid; place-items:center; color:#0D0E10; font-size:1.05rem; font-weight:850; letter-spacing:.04em; box-shadow:inset 0 0 0 1px rgba(255,255,255,.25); }
.cadence-rank { color:var(--cadence-amber); font:700 .7rem/1.2 ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.12em; text-transform:uppercase; }
.cadence-title { color:var(--cadence-ink); font-family:inherit; letter-spacing:normal; font-size:1.18rem; font-weight:750; line-height:1.25; margin:.15rem 0; }
.cadence-artist { color:var(--cadence-muted); font-size:.9rem; }
.cadence-why { color:#D8D3CA; margin:.8rem 0 .1rem; line-height:1.45; }

.cadence-signal { margin:.55rem 0; }
.cadence-signal-row { display:flex; justify-content:space-between; gap:12px; color:var(--cadence-muted); font-size:.78rem; }
.cadence-signal-track { height:6px; overflow:hidden; border-radius:99px; background:#262A2F; margin-top:5px; }
.cadence-signal-fill { display:block; height:100%; border-radius:inherit; }
.cadence-na { color:var(--cadence-muted); font-size:.78rem; }

.cadence-pipeline { display:grid; grid-template-columns:repeat(auto-fit,minmax(108px,1fr)); gap:7px; margin:.5rem 0 1rem; }
.cadence-stage { border:1px solid var(--cadence-border); background:var(--cadence-panel); border-radius:10px; padding:10px 8px; text-align:center; color:var(--cadence-muted); font-size:.72rem; }
.cadence-stage strong { display:block; color:var(--cadence-ink); font-size:.78rem; margin-top:3px; }

.cadence-legrank { margin:.15rem 0 .35rem; padding-left:1.2rem; color:var(--cadence-ink); font-size:.85rem; line-height:1.5; }
.cadence-legrank li { margin:.12rem 0; }
.cadence-lift { color:var(--cadence-amber); font-size:.62rem; font-weight:800; text-transform:uppercase; letter-spacing:.08em; border:1px solid var(--cadence-amber); border-radius:6px; padding:0 5px; margin-left:5px; white-space:nowrap; }

.cadence-sr-only { position:absolute!important; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden; clip:rect(0,0,0,0); white-space:nowrap; border:0; }
.cadence-skip { position:absolute; left:-9999px; top:8px; background:var(--cadence-amber); color:#0D0E10; padding:8px 14px; border-radius:8px; font-weight:750; z-index:1000; text-decoration:none; }
.cadence-skip:focus { left:12px; }

div[data-testid="stForm"] { border:1px solid var(--cadence-border); border-radius:var(--cadence-radius); padding:1rem; background:rgba(21,23,26,.72); }
div[data-testid="stExpander"] { border-color:var(--cadence-border); background:rgba(21,23,26,.45); }
.stButton > button, .stFormSubmitButton > button { min-height:44px; border-radius:10px; font-weight:650; }
*:focus-visible { outline:3px solid #F2BE74!important; outline-offset:3px!important; }

@media (max-width: 720px) {
  .block-container { padding:1.25rem .9rem 3rem; }
  .cadence-wave { display:none; }
  .cadence-wordmark { font-size:3rem; }
  .cadence-pipeline { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .cadence-cover { width:54px;height:54px;flex-basis:54px; }
}

@media (max-width: 420px) {
  .cadence-signal-row { flex-direction:column; gap:2px; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration:.001ms!important; animation-iteration-count:1!important; scroll-behavior:auto!important; transition:none!important; }
}
</style>
"""


def inject_theme() -> None:
    """Inject static, self-authored CSS only—never dynamic/user content."""
    st.markdown(CSS, unsafe_allow_html=True)
