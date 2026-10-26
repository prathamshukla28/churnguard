"""The interactive demo page served at ``GET /``.

A single self-contained HTML page (no build step, no external JS) that lets a
visitor enter a customer's details and see the model's live churn prediction.
It calls the service's own ``POST /predict`` endpoint.
"""

from __future__ import annotations

DEMO_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChurnGuard — live demo</title>
<style>
  :root {
    --bg:#0c0f18; --surface:#141926; --surface-2:#1b2130; --fg:#eef1f8;
    --muted:#9aa3bd; --border:#273040; --accent:#8b87ff; --accent-2:#38bdf8;
    --low:#34d399; --med:#fbbf24; --high:#f87171;
    --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
    --sans:"IBM Plex Sans",system-ui,sans-serif;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);line-height:1.6}
  .wrap{max-width:760px;margin:0 auto;padding:40px max(16px,4vw)}
  .eyebrow{font-family:var(--mono);font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}
  h1{font-size:clamp(1.7rem,5vw,2.4rem);margin:.3em 0 .1em;letter-spacing:-.02em}
  .sub{color:var(--muted);max-width:60ch}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:clamp(18px,4vw,28px);margin-top:26px}
  form{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .field{display:flex;flex-direction:column;gap:6px;min-width:0}
  .field.full{grid-column:1/-1}
  label{font-size:.82rem;color:var(--muted);font-family:var(--mono);text-transform:uppercase;letter-spacing:.05em}
  input,select{background:var(--surface-2);border:1px solid var(--border);color:var(--fg);border-radius:9px;padding:10px 12px;font-size:.95rem;font-family:inherit}
  input:focus,select:focus{outline:2px solid var(--accent);outline-offset:1px}
  .check{flex-direction:row;align-items:center;gap:10px}
  .check input{width:18px;height:18px}
  button{grid-column:1/-1;background:var(--accent);color:#0c0f18;border:0;border-radius:10px;padding:13px;font-size:1rem;font-weight:700;font-family:inherit;cursor:pointer}
  button:hover{filter:brightness(1.08)}
  .result{margin-top:22px;display:none}
  .result.show{display:block}
  .gauge{height:14px;border-radius:999px;background:var(--surface-2);overflow:hidden;border:1px solid var(--border)}
  .gauge>span{display:block;height:100%;width:0;transition:width .6s ease;border-radius:999px}
  .rlabel{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px;gap:10px;flex-wrap:wrap}
  .prob{font-family:var(--mono);font-size:2rem;font-weight:600;font-variant-numeric:tabular-nums}
  .band{font-family:var(--mono);font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;padding:4px 10px;border-radius:999px;border:1px solid}
  .reasons{list-style:none;padding:0;margin:16px 0 0;display:grid;gap:8px}
  .reasons li{display:flex;justify-content:space-between;gap:12px;font-size:.9rem;background:var(--surface-2);border:1px solid var(--border);border-radius:9px;padding:9px 12px}
  .reasons .why{color:var(--muted);font-family:var(--mono);font-size:.78rem}
  .foot{margin-top:28px;font-size:.84rem;color:var(--muted)}
  .foot a{color:var(--accent-2)}
  code{font-family:var(--mono);background:var(--surface-2);padding:2px 6px;border-radius:6px}
  @media(max-width:560px){form{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
  <div class="eyebrow">Live model demo</div>
  <h1>ChurnGuard</h1>
  <p class="sub">Enter a telecom customer's details and the trained model predicts how likely they are to
    cancel. This page calls the service's own <code>/predict</code> API — the same one documented at
    <a href="/docs" style="color:var(--accent-2)">/docs</a>.</p>

  <div class="card">
    <form id="f">
      <div class="field"><label for="tenure">Tenure (months)</label>
        <input type="number" id="tenure" min="0" max="72" value="2"></div>
      <div class="field"><label for="monthly">Monthly charges ($)</label>
        <input type="number" id="monthly" min="0" step="0.05" value="89.90"></div>
      <div class="field"><label for="contract">Contract</label>
        <select id="contract"><option>Month-to-month</option><option>One year</option><option>Two year</option></select></div>
      <div class="field"><label for="internet">Internet service</label>
        <select id="internet"><option>Fiber optic</option><option>DSL</option><option>No</option></select></div>
      <div class="field"><label for="payment">Payment method</label>
        <select id="payment"><option>Electronic check</option><option>Mailed check</option><option>Bank transfer (automatic)</option><option>Credit card (automatic)</option></select></div>
      <div class="field"><label for="techsupport">Tech support</label>
        <select id="techsupport"><option>No</option><option>Yes</option></select></div>
      <div class="field check full"><input type="checkbox" id="senior"><label for="senior">Senior citizen</label></div>
      <button type="submit">Predict churn risk</button>
    </form>

    <div class="result" id="result">
      <div class="rlabel">
        <span class="prob" id="prob">—</span>
        <span class="band" id="band">—</span>
      </div>
      <div class="gauge"><span id="bar"></span></div>
      <ul class="reasons" id="reasons"></ul>
    </div>
  </div>

  <p class="foot">Model: logistic regression, ROC-AUC 0.81 on held-out data.
    Code &amp; full write-up on <a href="https://github.com/prathamshukla28/churnguard">GitHub</a>.</p>
</div>

<script>
  var COLORS = {low:"#34d399", medium:"#fbbf24", high:"#f87171"};
  document.getElementById("f").addEventListener("submit", function (e) {
    e.preventDefault();
    var tenure = Number(document.getElementById("tenure").value) || 0;
    var monthly = Number(document.getElementById("monthly").value) || 0;
    var customer = {
      tenure: tenure,
      MonthlyCharges: monthly,
      TotalCharges: Math.round(monthly * Math.max(tenure, 1) * 100) / 100,
      Contract: document.getElementById("contract").value,
      InternetService: document.getElementById("internet").value,
      PaymentMethod: document.getElementById("payment").value,
      TechSupport: document.getElementById("techsupport").value,
      SeniorCitizen: document.getElementById("senior").checked ? 1 : 0
    };
    fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({customers: [customer], explain: true})
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        var p = data.predictions[0];
        var pct = Math.round(p.churn_probability * 100);
        document.getElementById("prob").textContent = pct + "% likely to churn";
        var band = document.getElementById("band");
        band.textContent = p.risk_band + " risk";
        band.style.color = COLORS[p.risk_band];
        band.style.borderColor = COLORS[p.risk_band];
        var bar = document.getElementById("bar");
        bar.style.width = pct + "%";
        bar.style.background = COLORS[p.risk_band];
        var reasons = document.getElementById("reasons");
        reasons.innerHTML = "";
        (p.reasons || []).forEach(function (r) {
          var li = document.createElement("li");
          var left = document.createElement("span");
          left.textContent = r.feature + ": " + r.value;
          var right = document.createElement("span");
          right.className = "why";
          right.textContent = (r.impact > 0 ? "↑ " : "↓ ") + r.direction;
          right.style.color = r.impact > 0 ? COLORS.high : COLORS.low;
          li.appendChild(left); li.appendChild(right);
          reasons.appendChild(li);
        });
        document.getElementById("result").classList.add("show");
      })
      .catch(function () {
        document.getElementById("prob").textContent = "Error — is the model loaded?";
        document.getElementById("result").classList.add("show");
      });
  });
</script>
</body>
</html>
"""
