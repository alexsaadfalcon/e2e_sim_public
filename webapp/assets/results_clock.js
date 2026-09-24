/* One clock, and now ONE transport, for every animated figure on the Results tab.
 *
 * Owner, live test 2026-09-24 (Thrust 1, A/B): "default should be side by side, with
 * both frames on the same colorbar limits, and both as if the play button was hit
 * (and should loop repeatedly). Framerate looks fine over RDP".
 *
 * Plotly's own play button animates ONE graph, ONCE, and stops on the last frame.
 * Two A/B arms started by hand drift apart from the first frame and neither loops.
 * So: a dcc.Interval in webapp/app.py ticks this function (RESULTS_CLOCK_MS), and on
 * every tick EVERY animated figure inside #results-tab-content is stepped to the same
 * frame index, wrapping forever. Nothing here is a real Plotly "animation" -- each
 * step is an immediate, zero-duration redraw to a named frame, so panels cannot fall
 * behind one another and the link only ever carries one redraw per panel per tick.
 *
 * CHANGED 2026-09-24 (layout spec section 4; hostile round 10, defects 8 and 3.3):
 * the figures no longer carry per-panel `updatemenus`/`sliders` -- up to EIGHT copies
 * of a transport nobody touched, 130 px of bottom margin each. There is now ONE
 * transport, built as plain HTML in the run-identity row (webapp/app.py
 * `_transport_bar`), and this file drives it:
 *   - the toggle button pauses/resumes the clock and swaps its own glyph;
 *   - the range input scrubs, and scrubbing pauses -- and because there is one frame
 *     index for the whole screen, parking it parks BOTH arms (before, dragging a
 *     per-panel slider parked one arm and silently compared A frame 3 with B frame 5);
 *   - the label prints "frame i of n" so a photograph identifies the frame.
 *
 * Robustness the app depends on:
 *   - No state is kept per graph div. Every tick re-queries the DOM, so a new run
 *     (Dash replaces the whole results tree) and a tab switch away and back are both
 *     no-ops that simply resume.
 *   - The tab being unmounted leaves nothing to animate; the tick returns early.
 *   - The transport elements are re-created by Dash on every render, so all listeners
 *     are DELEGATED from `document` rather than bound to the nodes themselves.
 */
window.dash_clientside = window.dash_clientside || {};

(function () {
    "use strict";

    var ROOT_ID = "results-tab-content";
    var TOGGLE_ID = "results-transport-toggle";
    var SLIDER_ID = "results-transport-slider";
    var LABEL_ID = "results-transport-label";
    var PLAY_GLYPH = "▶";        // play
    var PAUSE_GLYPH = "❚❚"; // pause

    // Module state, deliberately on `window` so a hot page reload reuses it rather
    // than starting a second, competing clock.
    var S = window.__e2eResultsClockState = window.__e2eResultsClockState ||
        {paused: false, frame: 0};

    function inResults(node) {
        var root = document.getElementById(ROOT_ID);
        return !!(root && node && root.contains(node));
    }

    function ancestorWithId(node, id) {
        while (node && node !== document) {
            if (node.id === id) return node;
            node = node.parentNode;
        }
        return null;
    }

    function animatedGraphs() {
        var root = document.getElementById(ROOT_ID);
        if (!root) return [];
        var divs = root.querySelectorAll(".js-plotly-plot");
        var out = [];
        for (var i = 0; i < divs.length; i++) {
            var gd = divs[i];
            var frames = (gd._transitionData && gd._transitionData._frames) || [];
            // >1: an animation over a single frame is noise, and pipeline_runner does
            // not attach one (the detector objectness panels are deliberately pinned
            // to the last frame and carry no frames at all).
            if (frames.length > 1) out.push([gd, frames]);
        }
        return out;
    }

    function maxFrames(graphs) {
        var n = 0;
        for (var i = 0; i < graphs.length; i++) {
            if (graphs[i][1].length > n) n = graphs[i][1].length;
        }
        return n;
    }

    function paint(graphs, n) {
        var idx = n ? (S.frame % n) : 0;
        for (var i = 0; i < graphs.length; i++) {
            var gd = graphs[i][0], frames = graphs[i][1];
            var frame = frames[S.frame % frames.length];
            if (!frame || frame.name === undefined || frame.name === null) continue;
            try {
                window.Plotly.animate(gd, [String(frame.name)], {
                    mode: "immediate",
                    transition: {duration: 0},
                    frame: {duration: 0, redraw: true}
                });
            } catch (e) {
                /* A graph mid-replacement by Dash: skip it, the next tick
                   re-queries the DOM and picks up its replacement. */
            }
        }
        var label = document.getElementById(LABEL_ID);
        if (label) label.textContent = "frame " + (idx + 1) + " of " + (n || 1);
        var slider = document.getElementById(SLIDER_ID);
        if (slider) {
            if (String(slider.max) !== String(n || 1)) slider.max = String(n || 1);
            slider.value = String(idx + 1);
        }
        var btn = document.getElementById(TOGGLE_ID);
        if (btn) btn.textContent = S.paused ? PLAY_GLYPH : PAUSE_GLYPH;
    }

    document.addEventListener("click", function (ev) {
        if (!ancestorWithId(ev.target, TOGGLE_ID)) return;
        ev.preventDefault();
        S.paused = !S.paused;
        var graphs = animatedGraphs();
        paint(graphs, maxFrames(graphs));
    }, true);

    // Scrubbing means "I want to talk about THIS frame": stop the clock so it does not
    // yank the panels back 700 ms later. The toggle resumes.
    function onScrub(ev) {
        if (!ancestorWithId(ev.target, SLIDER_ID)) return;
        S.paused = true;
        var v = parseInt(ev.target.value, 10);
        if (!isNaN(v)) S.frame = v - 1;
        var graphs = animatedGraphs();
        if (window.Plotly) paint(graphs, maxFrames(graphs));
    }
    document.addEventListener("input", onScrub, true);
    document.addEventListener("change", onScrub, true);

    window.dash_clientside.e2eResultsClock = {
        tick: function (_n_intervals) {
            var noUpdate = window.dash_clientside.no_update;
            if (!window.Plotly) return noUpdate;
            var graphs = animatedGraphs();
            if (!graphs.length) return noUpdate;
            if (S.paused) return noUpdate;
            S.frame = (S.frame + 1) % 1000000;
            paint(graphs, maxFrames(graphs));
            return S.frame;
        }
    };
})();
