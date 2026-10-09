const STORAGE_KEY = "spam_classifier_history";
const sampleTexts = {
  spam: `URGENT! Your account has been selected for a final verification update. Click the secure link below immediately to avoid suspension and unlock a limited-time reward. This message is time-sensitive and requires your action today.`,
  ham: `Hi team, thank you for the update on the project timeline. The revised milestones look good and the new draft is ready for review. Please let me know if you want any final edits before we share it with the client.`,
};

const elements = {
  input: document.getElementById("messageInput"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  spamSampleBtn: document.getElementById("spamSampleBtn"),
  hamSampleBtn: document.getElementById("hamSampleBtn"),
  modelSelect: document.getElementById("modelSelect"),
  resultBadge: document.getElementById("resultBadge"),
  resultHeadline: document.getElementById("resultHeadline"),
  confidenceText: document.getElementById("confidenceText"),
  timingText: document.getElementById("timingText"),
  probabilityValue: document.getElementById("probabilityValue"),
  progressRing: document.getElementById("progressRing"),
  meterLabel: document.getElementById("meterLabel"),
  highlightedText: document.getElementById("highlightedText"),
  historyList: document.getElementById("historyList"),
  clearHistoryBtn: document.getElementById("clearHistoryBtn"),
  spamKeywords: document.getElementById("spamKeywords"),
  hamKeywords: document.getElementById("hamKeywords"),
};

let spamKeywordChart = null;
let hamKeywordChart = null;

function setResultState(label, probability, confidence) {
  const isSpam = label === "spam";
  elements.resultBadge.textContent = isSpam ? "Spam" : "Safe";
  elements.resultBadge.classList.toggle("spam", isSpam);
  elements.resultBadge.classList.toggle("safe", !isSpam);
  elements.resultHeadline.textContent = isSpam ? "This message looks suspicious" : "This message looks safe";
  elements.confidenceText.textContent = `Confidence: ${confidence}%`;
  elements.meterLabel.textContent = isSpam ? "High risk" : "Low risk";
  elements.probabilityValue.textContent = `${Math.round(probability * 100)}%`;
  elements.progressRing.style.setProperty("--percent", Math.round(probability * 100));
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderHighlightText(text, suspiciousWords = []) {
  if (!text || !text.trim()) {
    elements.highlightedText.textContent = "Your scanned text will appear here with risky words highlighted.";
    return;
  }

  const keywords = new Set(suspiciousWords.map((word) => word.toLowerCase()));
  const tokens = text.split(/(\s+)/);
  const html = tokens
    .map((token) => {
      const cleaned = token.replace(/[^a-zA-Z]/g, "").toLowerCase();
      if (keywords.has(cleaned)) {
        return `<span class="spam-word">${escapeHtml(token)}</span>`;
      }
      return escapeHtml(token);
    })
    .join("");

  elements.highlightedText.innerHTML = html || "No suspicious words detected.";
}

async function analyzeText(text, model = "best") {
  if (!text || !text.trim()) {
    elements.resultHeadline.textContent = "No text provided";
    elements.confidenceText.textContent = "Confidence: 0%";
    return;
  }

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text, model }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Prediction failed");
    }

    const probability = Number(payload.spam_probability || 0);
    const confidence = Number(payload.confidence || 0);
    const label = payload.label || "ham";
    const topWords = payload.top_spam_words_found || [];

    setResultState(label, probability, confidence);
    renderHighlightText(text, topWords);
    elements.timingText.textContent = `Time: ${payload.processing_time_ms || 0} ms`;

    const history = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    const entry = {
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      label,
      probability,
      text: text.slice(0, 90),
    };

    const updated = [entry, ...history].slice(0, 10);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    renderHistory();
  } catch (error) {
    elements.resultHeadline.textContent = "Prediction unavailable";
    elements.confidenceText.textContent = error.message;
  }
}

function renderHistory() {
  const history = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  if (!history.length) {
    elements.historyList.innerHTML = '<div class="history-item"><p>No recent analyses yet.</p></div>';
    return;
  }

  elements.historyList.innerHTML = history
    .map(
      (item) => `
        <div class="history-item">
          <div>
            <strong>${item.timestamp}</strong>
            <p>${escapeHtml(item.text || "No content")}</p>
          </div>
          <span class="pill ${item.label === "spam" ? "spam" : "safe"}">${item.label}</span>
        </div>
      `,
    )
    .join("");
}

async function loadMetrics() {
  try {
    const response = await fetch("/api/metrics");
    const data = await response.json();
    const modelEntries = Object.entries(data.models || {});

    const cards = document.querySelectorAll(".metric-card");
    cards.forEach((card, index) => {
      const [modelName, values] = modelEntries[index] || [];
      if (!values) return;
      card.querySelector("span").textContent = modelName;
      card.querySelector("strong").textContent = `${(values.accuracy * 100).toFixed(1)}%`;
      card.querySelector("small").textContent = `F1 ${values.f1.toFixed(3)} • FPR ${values.false_positive_rate.toFixed(3)}`;
    });

    const chartPaths = data.charts || {};
    document.getElementById("modelComparisonChart").src = chartPaths.model_comparison || "/static/charts/model_comparison.png";
    document.getElementById("classDistributionChart").src = chartPaths.class_distribution || "/static/charts/class_distribution.png";
    document.getElementById("nbMatrix").src = chartPaths.confusion_matrix_nb || "/static/charts/confusion_matrix_Naive_Bayes.png";
    document.getElementById("lrMatrix").src = chartPaths.confusion_matrix_logistic || "/static/charts/confusion_matrix_Logistic_Regression.png";
    document.getElementById("svmMatrix").src = chartPaths.confusion_matrix_svm || "/static/charts/confusion_matrix_Linear_SVM.png";
  } catch (error) {
    console.error("Failed to load model metrics:", error);
  }
}

function renderKeywordChart(canvasId, entries, label, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;

  const data = entries.map((entry) => ({
    word: entry.word || entry.label || "unknown",
    weight: Number(entry.weight || 0),
  }));

  const chartInstance = new Chart(canvas.getContext("2d"), {
    type: "bar",
    data: {
      labels: data.map((item) => item.word),
      datasets: [{
        label,
        data: data.map((item) => Number(item.weight)),
        backgroundColor: color,
        borderRadius: 8,
        borderSkipped: false,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          callbacks: {
            label: (context) => `${context.parsed.x.toFixed(4)}`,
          },
        },
      },
      scales: {
        x: {
          beginAtZero: false,
          grid: {
            color: "rgba(148, 163, 184, 0.15)",
            drawBorder: false,
          },
          ticks: {
            color: "#e2e8f0",
          },
          title: {
            display: true,
            text: "Weight",
            color: "#cbd5e1",
          },
        },
        y: {
          grid: {
            color: "rgba(148, 163, 184, 0.12)",
            drawBorder: false,
          },
          ticks: {
            color: "#e2e8f0",
          },
        },
      },
    },
  });

  return chartInstance;
}

async function loadKeywords() {
  try {
    const response = await fetch("/api/keywords");
    const data = await response.json();
    const spamWords = Array.isArray(data.spam) ? data.spam.slice(0, 15) : [];
    const hamWords = Array.isArray(data.ham) ? data.ham.slice(0, 15) : [];

    elements.spamKeywords.innerHTML = spamWords.slice(0, 10).map((item) => `<li>${item.word} <span>(${Number(item.weight).toFixed(4)})</span></li>`).join("");
    elements.hamKeywords.innerHTML = hamWords.slice(0, 10).map((item) => `<li>${item.word} <span>(${Number(item.weight).toFixed(4)})</span></li>`).join("");

    if (spamKeywordChart) spamKeywordChart.destroy();
    if (hamKeywordChart) hamKeywordChart.destroy();

    spamKeywordChart = renderKeywordChart("spamKeywordChart", spamWords, "Top Spam Keywords", "rgba(244, 114, 182, 0.85)");
    hamKeywordChart = renderKeywordChart("hamKeywordChart", hamWords, "Top Ham Keywords", "rgba(34, 211, 238, 0.85)");
  } catch (error) {
    console.error("Failed to load keywords:", error);
  }
}

function bindTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      tabButtons.forEach((item) => item.classList.toggle("active", item === button));
      tabPanels.forEach((panel) => panel.classList.toggle("active", panel.id === `${button.dataset.tab}Panel`));
    });
  });
}

function animateCounters() {
  const counters = document.querySelectorAll(".counter");
  counters.forEach((counter) => {
    const target = Number(counter.dataset.target || 0);
    const duration = 1200;
    let start = 0;
    const step = (time) => {
      if (!start) start = time;
      const progress = Math.min((time - start) / duration, 1);
      counter.textContent = `${Math.round(progress * target)}${counter.dataset.target === "96" ? "+" : ""}`;
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}

function bindEvents() {
  elements.analyzeBtn.addEventListener("click", () => {
    analyzeText(elements.input.value, elements.modelSelect.value);
  });

  elements.spamSampleBtn.addEventListener("click", () => {
    elements.input.value = sampleTexts.spam;
    analyzeText(sampleTexts.spam, elements.modelSelect.value);
  });

  elements.hamSampleBtn.addEventListener("click", () => {
    elements.input.value = sampleTexts.ham;
    analyzeText(sampleTexts.ham, elements.modelSelect.value);
  });

  elements.modelSelect.addEventListener("change", () => {
    if (elements.input.value.trim()) {
      analyzeText(elements.input.value, elements.modelSelect.value);
    }
  });

  elements.clearHistoryBtn.addEventListener("click", () => {
    localStorage.removeItem(STORAGE_KEY);
    renderHistory();
  });

  elements.input.addEventListener("input", () => {
    clearTimeout(window.analysisTimer);
    window.analysisTimer = setTimeout(() => {
      if (elements.input.value.trim()) {
        analyzeText(elements.input.value, elements.modelSelect.value);
      }
    }, 400);
  });
}

async function init() {
  bindTabs();
  bindEvents();
  renderHistory();
  loadMetrics();
  loadKeywords();
  animateCounters();
}

window.addEventListener("DOMContentLoaded", init);
