/**
 * Phishing AI Agent System — Gelişmiş Görselleştirme
 * 1. Canlı Pipeline Animasyonu
 * 2. SHAP Feature Importance Bar Chart
 * 3. Stilometrik Radar Chart
 */

// ═══════════════════════════════════════════
// 1. CANLI PIPELINE ANİMASYONU
// ═══════════════════════════════════════════

const AGENT_STEPS = [
    { id: 'ag1', name: 'Adım 1', sub: 'Ingestion', icon: '📧', color: '#c084fc' },
    { id: 'ag2', name: 'Adım 2', sub: 'Paralel Pre', icon: '⚡', color: '#c084fc' },
    { id: 'ag3', name: 'Adım 3', sub: 'Stylometric', icon: '✍️', color: '#c084fc' },
    { id: 'ag4', name: 'Adım 4', sub: 'XGBoost', icon: '🌲', color: '#c084fc' },
    { id: 'ag5', name: 'Adım 5', sub: 'Threshold', icon: '⚖️', color: '#fbbf24' },
    { id: 'ag6', name: 'Adım 6', sub: 'NLP', icon: '📝', color: '#60a5fa' },
    { id: 'ag7', name: 'Adım 7', sub: 'BiLSTM+GloVe', icon: '🧠', color: '#60a5fa' },
    { id: 'ag8', name: 'Adım 8', sub: 'Meta-Clf', icon: '🔗', color: '#60a5fa' },
    { id: 'ag9', name: 'Adım 9', sub: 'Sonuç', icon: '📤', color: '#34d399' },
];

function renderPipelineAnimation(containerId) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = '';
    el.className = 'pipeline-anim-container';

    AGENT_STEPS.forEach((step, i) => {
        const node = document.createElement('div');
        node.className = 'pipe-node';
        node.id = 'pipe-' + step.id;
        node.innerHTML = `
            <div class="pipe-icon">${step.icon}</div>
            <div class="pipe-name">${step.name}</div>
            <div class="pipe-sub">${step.sub}</div>
            <div class="pipe-pulse"></div>
        `;
        el.appendChild(node);

        if (i < AGENT_STEPS.length - 1) {
            const arrow = document.createElement('div');
            arrow.className = 'pipe-arrow';
            arrow.id = 'arrow-' + i;
            arrow.innerHTML = '→';
            el.appendChild(arrow);
        }
    });
}

async function animatePipeline(isFull) {
    const steps = isFull ? AGENT_STEPS : AGENT_STEPS.filter((_, i) => i < 5 || i === 8);

    // Reset all
    AGENT_STEPS.forEach(s => {
        const el = document.getElementById('pipe-' + s.id);
        if (el) { el.classList.remove('active', 'done', 'skipped'); }
    });

    // Skip agents 6-8 if fast path
    if (!isFull) {
        [5, 6, 7].forEach(i => {
            const el = document.getElementById('pipe-' + AGENT_STEPS[i].id);
            if (el) el.classList.add('skipped');
        });
    }

    for (const step of steps) {
        const el = document.getElementById('pipe-' + step.id);
        if (!el) continue;
        el.classList.add('active');
        await new Promise(r => setTimeout(r, 300));
        el.classList.remove('active');
        el.classList.add('done');
    }
}

// ═══════════════════════════════════════════
// 2. SHAP FEATURE IMPORTANCE BAR CHART
// ═══════════════════════════════════════════

function renderSHAPChart(containerId, features) {
    const el = document.getElementById(containerId);
    if (!el || !features) return;

    const entries = Object.entries(features).sort((a, b) => b[1] - a[1]);
    const max = Math.max(...entries.map(e => e[1]));

    el.innerHTML = `
        <div class="shap-title">🔍 Neden Bu Karar? — Feature Importance</div>
        <div class="shap-bars">
            ${entries.map(([name, val]) => {
                const pct = (val / max * 100).toFixed(1);
                const label = name.replace(/_/g, ' ');
                return `
                    <div class="shap-row">
                        <div class="shap-label">${label}</div>
                        <div class="shap-bar-bg">
                            <div class="shap-bar-fill" style="width:${pct}%"></div>
                        </div>
                        <div class="shap-val">${val.toFixed(4)}</div>
                    </div>
                `;
            }).join('')}
        </div>
    `;
}

// ═══════════════════════════════════════════
// 3. STİLOMETRİK RADAR CHART (Canvas)
// ═══════════════════════════════════════════

const RADAR_LABELS = [
    'Karakter', 'Kelime', 'Cümle', 'Büyük Harf',
    'Özel Kar.', 'Ort. Kelime', 'Kel/Kar',
    'Özel/Kar', 'Benzersiz', 'Ünlem', 'Soru', 'Ort. Cümle', 'Max Cümle'
];
const RADAR_KEYS = [
    'num_chars', 'num_words', 'num_sentences', 'num_upper_chars',
    'num_special_chars', 'avg_word_size', 'words_to_chars',
    'special_chars_to_chars', 'unique_words_to_word',
    'exclamationmark_to_chars', 'questionmark_to_chars',
    'avg_words_in_sentence', 'max_words_in_sentence'
];

// Tipik phishing profili (normalize edilmiş)
const PHISHING_PROFILE = [0.6, 0.5, 0.3, 0.8, 0.7, 0.5, 0.4, 0.7, 0.3, 0.8, 0.6, 0.4, 0.5];
const SAFE_PROFILE =     [0.4, 0.5, 0.6, 0.2, 0.3, 0.5, 0.5, 0.3, 0.7, 0.1, 0.2, 0.6, 0.4];

function normalizeFeatures(features) {
    const maxVals = [5000, 800, 50, 200, 150, 8, 0.3, 0.1, 1, 0.05, 0.02, 30, 80];
    return RADAR_KEYS.map((key, i) => {
        const v = features[key] || 0;
        return Math.min(v / maxVals[i], 1);
    });
}

function renderRadarChart(canvasId, features) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !features) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width = 420;
    const H = canvas.height = 420;
    const cx = W / 2, cy = H / 2, R = 150;
    const n = RADAR_LABELS.length;

    ctx.clearRect(0, 0, W, H);

    // BG grid circles
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    for (let r = 1; r <= 5; r++) {
        ctx.beginPath();
        ctx.arc(cx, cy, R * r / 5, 0, Math.PI * 2);
        ctx.stroke();
    }

    // Axes
    const angles = RADAR_LABELS.map((_, i) => (Math.PI * 2 * i / n) - Math.PI / 2);
    ctx.strokeStyle = 'rgba(255,255,255,0.1)';
    angles.forEach(a => {
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + R * Math.cos(a), cy + R * Math.sin(a));
        ctx.stroke();
    });

    // Labels
    ctx.fillStyle = '#8892b0';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    angles.forEach((a, i) => {
        const lx = cx + (R + 24) * Math.cos(a);
        const ly = cy + (R + 24) * Math.sin(a);
        ctx.fillText(RADAR_LABELS[i], lx, ly + 4);
    });

    function drawPolygon(values, fillColor, strokeColor) {
        ctx.beginPath();
        values.forEach((v, i) => {
            const x = cx + R * v * Math.cos(angles[i]);
            const y = cy + R * v * Math.sin(angles[i]);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fillStyle = fillColor;
        ctx.fill();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    // Safe profile (gri)
    drawPolygon(SAFE_PROFILE, 'rgba(16,185,129,0.08)', 'rgba(16,185,129,0.3)');

    // Phishing profile (kırmızı)
    drawPolygon(PHISHING_PROFILE, 'rgba(239,68,68,0.08)', 'rgba(239,68,68,0.3)');

    // Current email (mavi)
    const normalized = normalizeFeatures(features);
    drawPolygon(normalized, 'rgba(59,130,246,0.15)', 'rgba(59,130,246,0.8)');

    // Dots on current
    normalized.forEach((v, i) => {
        const x = cx + R * v * Math.cos(angles[i]);
        const y = cy + R * v * Math.sin(angles[i]);
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#3b82f6';
        ctx.fill();
    });

    // Legend
    const legends = [
        { color: '#3b82f6', label: 'Bu E-posta' },
        { color: '#ef4444', label: 'Tipik Phishing' },
        { color: '#10b981', label: 'Tipik Safe' },
    ];
    legends.forEach((l, i) => {
        const lx = 16, ly = H - 50 + i * 16;
        ctx.fillStyle = l.color;
        ctx.fillRect(lx, ly, 10, 10);
        ctx.fillStyle = '#8892b0';
        ctx.font = '11px Inter';
        ctx.textAlign = 'left';
        ctx.fillText(l.label, lx + 16, ly + 9);
    });
}
