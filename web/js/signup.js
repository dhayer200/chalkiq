// ChalkIQ — Newsletter signup handlers
(function () {
  const API_BASE = "/api";

  // Free signup
  const freeForm = document.getElementById("freeForm");
  const freeMsg = document.getElementById("freeMsg");
  if (freeForm) {
    freeForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const email = document.getElementById("freeEmail").value.trim();
      if (!email) return;

      const btn = freeForm.querySelector("button");
      btn.disabled = true;
      btn.textContent = "Subscribing...";

      try {
        const res = await fetch(`${API_BASE}/subscribe`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, tier: "free" }),
        });
        const data = await res.json();
        if (data.ok) {
          showMsg(freeMsg, "You're in! Check your inbox.", "text-chalk-400");
          freeForm.reset();
        } else {
          showMsg(freeMsg, data.error || "Something went wrong.", "text-red-400");
        }
      } catch {
        showMsg(freeMsg, "Network error. Try again.", "text-red-400");
      } finally {
        btn.disabled = false;
        btn.textContent = "Get Free Picks";
      }
    });
  }

  // Paid signup -> Stripe Checkout
  const paidForm = document.getElementById("paidForm");
  const paidMsg = document.getElementById("paidMsg");
  if (paidForm) {
    paidForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const email = document.getElementById("paidEmail").value.trim();
      if (!email) return;

      const btn = paidForm.querySelector("button");
      btn.disabled = true;
      btn.textContent = "Redirecting...";

      try {
        const res = await fetch(`${API_BASE}/checkout`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email }),
        });
        const data = await res.json();
        if (data.url) {
          window.location.href = data.url;
        } else {
          showMsg(paidMsg, data.error || "Something went wrong.", "text-red-400");
        }
      } catch {
        showMsg(paidMsg, "Network error. Try again.", "text-red-400");
      } finally {
        btn.disabled = false;
        btn.textContent = "Start Premium";
      }
    });
  }

  function showMsg(el, text, className) {
    el.textContent = text;
    el.className = `text-sm mt-3 ${className}`;
    el.classList.remove("hidden");
  }
})();
