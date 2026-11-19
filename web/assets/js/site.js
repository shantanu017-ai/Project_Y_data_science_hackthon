(function () {
  "use strict";

  var root = document.documentElement;
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---------- theme ---------- */
  var KEY = "voltrelay-theme";
  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    var btn = document.querySelector(".theme-toggle");
    if (btn) {
      btn.setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
      btn.setAttribute("aria-pressed", theme === "dark" ? "true" : "false");
    }
    var meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", theme === "dark" ? "#151412" : "#f7f3ea");
  }
  var stored = null;
  try { stored = localStorage.getItem(KEY); } catch (e) {}
  applyTheme(stored === "dark" || stored === "light" ? stored : "light");

  var toggle = document.querySelector(".theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      applyTheme(next);
      try { localStorage.setItem(KEY, next); } catch (e) {}
    });
  }

  /* ---------- reading progress ---------- */
  var bar = document.querySelector(".progress span");
  function progress() {
    if (!bar) return;
    var max = document.documentElement.scrollHeight - window.innerHeight;
    var ratio = max > 0 ? Math.min(1, window.scrollY / max) : 0;
    bar.style.transform = "scaleX(" + ratio.toFixed(4) + ")";
  }
  window.addEventListener("scroll", progress, { passive: true });
  window.addEventListener("resize", progress);
  progress();

  /* ---------- count-up ---------- */
  function fmt(v, dec) {
    return v.toFixed(dec).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }
  function count(el) {
    if (el.dataset.done) return;
    el.dataset.done = "1";
    var to = parseFloat(el.dataset.to);
    var dec = parseInt(el.dataset.dec || "0", 10);
    if (!isFinite(to)) return;
    if (reduce) { el.textContent = fmt(to, dec); return; }
    var start = performance.now();
    var dur = 900;
    function tick(now) {
      var p = Math.min(1, (now - start) / dur);
      var eased = 1 - Math.pow(1 - p, 3);
      el.textContent = fmt(to * eased, dec);
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  /* ---------- reveal + count observers ---------- */
  var revealEls = document.querySelectorAll(".reveal");
  var countEls = document.querySelectorAll("[data-to]");

  if ("IntersectionObserver" in window && !reduce) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        io.unobserve(entry.target);
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
    revealEls.forEach(function (el) { io.observe(el); });

    var co = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        count(entry.target);
        co.unobserve(entry.target);
      });
    }, { threshold: 0.6 });
    countEls.forEach(function (el) { co.observe(el); });
  } else {
    revealEls.forEach(function (el) { el.classList.add("is-visible"); });
    countEls.forEach(function (el) {
      var to = parseFloat(el.dataset.to);
      if (isFinite(to)) el.textContent = fmt(to, parseInt(el.dataset.dec || "0", 10));
    });
  }

  /* ---------- active nav ---------- */
  var links = Array.prototype.slice.call(document.querySelectorAll(".nav-links a[href^='#']"));
  var sections = links.map(function (a) {
    return document.querySelector(a.getAttribute("href"));
  }).filter(Boolean);

  if ("IntersectionObserver" in window && sections.length) {
    var navIo = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        links.forEach(function (a) {
          var current = a.getAttribute("href") === "#" + entry.target.id;
          if (current) a.setAttribute("aria-current", "true");
          else a.removeAttribute("aria-current");
        });
      });
    }, { rootMargin: "-30% 0px -55% 0px", threshold: 0 });
    sections.forEach(function (s) { navIo.observe(s); });
  }

  /* ---------- footer year ---------- */
  var year = document.getElementById("year");
  if (year) year.textContent = String(new Date().getFullYear());
})();
