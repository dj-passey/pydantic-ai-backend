// Predictive Analytics — WebSocket Client + Chart.js Rendering

const chatStream = document.getElementById("chat-stream");
const messageInput = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const welcome = document.getElementById("welcome");

let ws = null;
let currentAssistantEl = null;
let currentToolCard = null;
let streamedText = "";
let isProcessing = false;

const CHART_COLORS = ["#5b8dd9", "#69b38a", "#e0c285", "#e06c75", "#a78bfa", "#f0a8d0"];

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

function connect() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${protocol}//${location.host}/ws`);

    ws.onopen = () => {
        statusDot.classList.add("connected");
        statusText.textContent = "Connected";
    };

    ws.onclose = () => {
        statusDot.classList.remove("connected");
        statusText.textContent = "Disconnected";
        setTimeout(connect, 3000);
    };

    ws.onerror = () => {
        statusText.textContent = "Connection error";
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleMessage(msg);
    };
}

function handleMessage(msg) {
    switch (msg.type) {
        case "start":
            hideWelcome();
            currentAssistantEl = createAssistantMessage();
            streamedText = "";
            currentToolCard = null;
            setProcessing(true);
            break;

        case "status":
            // Don't show "Running prediction in Docker..." — tool cards already indicate this
            break;

        case "text_delta":
            if (currentAssistantEl) {
                streamedText += msg.content;
                renderMarkdown(currentAssistantEl, cleanText(streamedText));
                scrollToBottom();
            }
            break;

        case "tool_call_start":
            currentToolCard = createToolCard(msg.tool_name);
            scrollToBottom();
            break;

        case "tool_args_delta":
            if (currentToolCard) {
                appendToolArgs(currentToolCard, msg.args_delta);
            }
            break;

        case "tool_start":
            if (!currentToolCard || currentToolCard.dataset.toolName !== msg.tool_name) {
                currentToolCard = createToolCard(msg.tool_name);
            }
            setToolArgs(currentToolCard, msg.args);
            setToolStatus(currentToolCard, "running");
            scrollToBottom();
            break;

        case "tool_output":
            if (currentToolCard) {
                setToolOutput(currentToolCard, msg.output);
                setToolStatus(currentToolCard, "done");
            }
            currentToolCard = null;
            scrollToBottom();
            break;

        case "chart_data":
            renderChart(msg.data);
            scrollToBottom();
            break;

        case "response":
            // Final response — ensure text is rendered
            if (currentAssistantEl && msg.content) {
                renderMarkdown(currentAssistantEl, cleanText(msg.content));
            }
            break;

        case "done":
            clearStatus();
            setProcessing(false);
            currentAssistantEl = null;
            currentToolCard = null;
            break;

        case "error":
            showError(msg.content);
            setProcessing(false);
            break;
    }
}

// ---------------------------------------------------------------------------
// UI Helpers
// ---------------------------------------------------------------------------

function hideWelcome() {
    if (welcome) welcome.style.display = "none";
}

function setProcessing(val) {
    isProcessing = val;
    sendBtn.disabled = val;
    messageInput.disabled = val;
    if (!val) messageInput.focus();
}

function clearStatus() {
    const bars = chatStream.querySelectorAll(".status-bar.active");
    bars.forEach((b) => b.remove());
}

function cleanText(text) {
    // Strip markdown image references to CHART_DATA (agent shouldn't generate these)
    return text.replace(/!\[.*?\]\(CHART_DATA.*?\)/g, "").replace(/\n{3,}/g, "\n\n").trim();
}

function showError(text) {
    const el = document.createElement("div");
    el.className = "message";
    el.innerHTML = `<div class="message-assistant" style="color: var(--error);">Error: ${escapeHtml(text)}</div>`;
    chatStream.appendChild(el);
    scrollToBottom();
}

function createUserMessage(text) {
    const el = document.createElement("div");
    el.className = "message";
    el.innerHTML = `<div class="message-user">${escapeHtml(text)}</div>`;
    chatStream.appendChild(el);
    scrollToBottom();
    return el;
}

function createAssistantMessage() {
    clearStatus();
    const el = document.createElement("div");
    el.className = "message";
    const inner = document.createElement("div");
    inner.className = "message-assistant";
    inner.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    el.appendChild(inner);
    chatStream.appendChild(el);
    scrollToBottom();
    return inner;
}

function renderMarkdown(el, text) {
    // Simple markdown: code blocks, inline code, bold, line breaks
    let html = escapeHtml(text);

    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    // Line breaks -> paragraphs
    html = html
        .split("\n\n")
        .map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`)
        .join("");

    el.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Tool Cards
// ---------------------------------------------------------------------------

function createToolCard(toolName) {
    const card = document.createElement("div");
    card.className = "tool-card";
    card.dataset.toolName = toolName;

    card.innerHTML = `
        <div class="tool-card-header">
            <span class="tool-icon">&#9654;</span>
            <span class="tool-name">${escapeHtml(toolName)}</span>
            <span class="tool-status running">running</span>
        </div>
        <div class="tool-card-body">
            <div class="tool-section-label">Arguments</div>
            <div class="tool-args"></div>
        </div>
    `;

    // Toggle expand
    const header = card.querySelector(".tool-card-header");
    const body = card.querySelector(".tool-card-body");
    const icon = card.querySelector(".tool-icon");
    header.addEventListener("click", () => {
        body.classList.toggle("show");
        icon.classList.toggle("expanded");
    });

    chatStream.appendChild(card);
    return card;
}

function appendToolArgs(card, delta) {
    const argsEl = card.querySelector(".tool-args");
    argsEl.textContent += delta;
}

function setToolArgs(card, args) {
    const argsEl = card.querySelector(".tool-args");
    if (typeof args === "object") {
        argsEl.textContent = JSON.stringify(args, null, 2);
    } else {
        argsEl.textContent = String(args);
    }
}

function setToolOutput(card, output) {
    const body = card.querySelector(".tool-card-body");

    // Add output section
    const label = document.createElement("div");
    label.className = "tool-section-label";
    label.textContent = "Output";
    body.appendChild(label);

    const outputEl = document.createElement("div");
    outputEl.className = "tool-output";
    outputEl.textContent = output;
    body.appendChild(outputEl);
}

function setToolStatus(card, status) {
    const statusEl = card.querySelector(".tool-status");
    statusEl.className = `tool-status ${status}`;
    statusEl.textContent = status === "running" ? "running" : "done";
}

// ---------------------------------------------------------------------------
// Chart Rendering
// ---------------------------------------------------------------------------

function renderChart(chartData) {
    const container = document.createElement("div");
    container.className = "chart-container";

    const title = document.createElement("h4");
    title.textContent = chartData.title;
    container.appendChild(title);

    const canvas = document.createElement("canvas");
    container.appendChild(canvas);

    // Insert chart BEFORE the current assistant message so charts appear above text
    if (currentAssistantEl && currentAssistantEl.parentElement) {
        chatStream.insertBefore(container, currentAssistantEl.parentElement);
    } else {
        chatStream.appendChild(container);
    }

    // Merge all unique x-labels across all series (Historical + Forecast may have different dates)
    const allLabels = [...new Set(chartData.series.flatMap((s) => s.data_points.map((dp) => dp.x)))].sort();

    // Build a lookup per series: x -> y
    const datasets = chartData.series.map((s, i) => {
        const lookup = new Map(s.data_points.map((dp) => [dp.x, dp.y]));
        return {
            label: s.name,
            data: allLabels.map((x) => lookup.get(x) ?? null),
            borderColor: CHART_COLORS[i % CHART_COLORS.length],
            backgroundColor: CHART_COLORS[i % CHART_COLORS.length] + "18",
            borderWidth: 2,
            tension: 0.3,
            fill: false,
            pointRadius: 3,
            pointHoverRadius: 5,
            spanGaps: false,
        };
    });

    new Chart(canvas.getContext("2d"), {
        type: "line",
        data: {
            labels: allLabels,
            datasets: datasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                mode: "index",
                intersect: false,
            },
            plugins: {
                legend: {
                    labels: {
                        color: "#e0e0e0",
                        font: { size: 12 },
                    },
                },
                tooltip: {
                    backgroundColor: "#1e1e1e",
                    borderColor: "#404040",
                    borderWidth: 1,
                    titleColor: "#e0e0e0",
                    bodyColor: "#e0e0e0",
                },
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: chartData.x_label,
                        color: "#808080",
                        font: { size: 12 },
                    },
                    ticks: { color: "#808080", maxRotation: 45 },
                    grid: { color: "#1e1e1e" },
                },
                y: {
                    title: {
                        display: true,
                        text: chartData.y_label,
                        color: "#808080",
                        font: { size: 12 },
                    },
                    ticks: { color: "#808080" },
                    grid: { color: "#1e1e1e" },
                },
            },
        },
    });
}

// ---------------------------------------------------------------------------
// Input handling
// ---------------------------------------------------------------------------

function sendMessage() {
    const text = messageInput.value.trim();
    if (!text || isProcessing || !ws || ws.readyState !== WebSocket.OPEN) return;

    createUserMessage(text);
    ws.send(JSON.stringify({ message: text }));
    messageInput.value = "";
    autoResizeTextarea();
}

sendBtn.addEventListener("click", sendMessage);

messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

messageInput.addEventListener("input", autoResizeTextarea);

function autoResizeTextarea() {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + "px";
}

// Hint buttons
document.querySelectorAll(".hint-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
        messageInput.value = btn.dataset.message;
        sendMessage();
    });
});

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatStream.scrollTop = chatStream.scrollHeight;
    });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

connect();
messageInput.focus();
