/* One clock for every animated figure on the Results tab.
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
 * Robustness the app depends on:
 *   - No state is kept per graph div. Every tick re-queries the DOM, so a new run
 *     (Dash replaces the whole results tree) and a tab switch away and back are both
 *     no-ops that simply resume.
 *   - The tab being unmounted leaves nothing to animate; the tick returns early.
 *   - Plotly's own frame slider follows, because plotly.js syncs a slider whose steps
 *     animate to the same frame names (webapp/pipeline_runner._add_frame_animation
 *     names its frames "0".."n-1" and its slider steps animate to those names).
 */
window.dash_clientside = window.dash_clientside || {};

(function () {
    "use strict";

    var ROOT_ID = "results-tab-content";
    var PLAY_GLYPH = "▶";          // the ▶ label pipeline_runner draws

    // Module state, deliberately on `window` so a hot page reload reuses it rather
    // than starting a second, competing clock.
    var S = window.__e2eResultsClockState = window.__e2eResultsClockState ||
        {paused: false, frame: 0};

    function ancestorWithClass(node, cls) {
        // getAttribute("class"), not classList: these are SVG <g> elements, and
        // SVGElement.classList is missing on older engines the conference laptop may
        // still be running.
        while (node && node !== document) {
            var c = (node.getAttribute && node.getAttribute("class")) || "";
            if (c.split(/\s+/).indexOf(cls) >= 0) return node;
            node = node.parentNode;
        }
        return null;
    }

    function inResults(node) {
        var root = document.getElementById(ROOT_ID);
        return !!(root && node && root.contains(node));
    }

    // Plotly's ▶/❚❚ buttons drive THIS clock instead of their own one-shot animation.
    // Capture phase + stopPropagation: the event never reaches plotly.js's own
    // handler on the button node, so there is exactly one thing stepping frames and
    // the two arms cannot drift apart. Pressing either button on ANY panel pauses or
    // resumes ALL of them, which is what "one clock" means on an A/B screen.
    document.addEventListener("click", function (ev) {
        if (!inResults(ev.target)) return;
        var btn = ancestorWithClass(ev.target, "updatemenu-button");
        if (!btn) return;
        ev.stopPropagation();
        ev.preventDefault();
        S.paused = ((btn.textContent || "").indexOf(PLAY_GLYPH) < 0);
    }, true);

    // Dragging the frame slider means "I want to talk about THIS frame": stop the
    // clock so it does not yank the panel back 700 ms later. The ▶ button resumes.
    // Not intercepted (no stopPropagation) -- the drag itself must still work.
    document.addEventListener("pointerdown", function (ev) {
        if (!inResults(ev.target)) return;
        if (ancestorWithClass(ev.target, "slider-container")) S.paused = true;
    }, true);

    function animatedGraphs() {
        var root = document.getElementById(ROOT_ID);
        if (!root) return [];
        var divs = root.querySelectorAll(".js-plotly-plot");
        var out = [];
        for (var i = 0; i < divs.length; i++) {
            var gd = divs[i];
            var frames = (gd._transitionData && gd._transitionData._frames) || [];
            // >1: a slider over a single frame is noise, and pipeline_runner does not
            // attach one (the detector objectness panels are deliberately pinned to
            // the last frame and carry no frames at all).
            if (frames.length > 1) out.push([gd, frames]);
        }
        return out;
    }

    window.dash_clientside.e2eResultsClock = {
        tick: function (_n_intervals) {
            var noUpdate = window.dash_clientside.no_update;
            if (!window.Plotly) return noUpdate;
            var graphs = animatedGraphs();
            if (!graphs.length) return noUpdate;
            if (S.paused) return noUpdate;
            S.frame = (S.frame + 1) % 1000000;
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
            return S.frame;
        }
    };
})();
