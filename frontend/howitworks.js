/**
 * Nasil Calisir - Compact Tabbed Layout
 */

function renderHowItWorks() {
    var el = document.getElementById('page-howitworks');
    if (!el) return;

    var html = '';
    html += '<div class="hiw-tabs">';
    html += '<button class="hiw-tab active" onclick="switchHiwTab(\'overview\',this)">Genel Bakis</button>';
    html += '<button class="hiw-tab" onclick="switchHiwTab(\'pipeline\',this)">Pipeline</button>';
    html += '<button class="hiw-tab" onclick="switchHiwTab(\'nlp\',this)">NLP</button>';
    html += '<button class="hiw-tab" onclick="switchHiwTab(\'stylometric\',this)">Stilometri</button>';
    html += '<button class="hiw-tab" onclick="switchHiwTab(\'threshold\',this)">Karar</button>';
    html += '<button class="hiw-tab" onclick="switchHiwTab(\'arch\',this)">Mimari</button>';
    html += '</div>';

    html += '<div class="hiw-tab-content">';

    // GENEL BAKIS
    html += '<div id="hiw-overview" class="hiw-tab-pane active">';
    html += '<div class="card">';
    html += '<div class="card-title">Projenin Amaci</div>';
    html += '<p style="color:var(--text-secondary);line-height:1.6;font-size:0.82rem">';
    html += 'Bu sistem, e-posta iceriklerini <strong style="color:var(--accent-blue)">9 adimli cok katmanli bir pipeline</strong> kullanarak analiz eder ve ';
    html += '<strong style="color:var(--accent-red)">phishing (oltalama)</strong> saldirilarini tespit eder. ';
    html += 'Geleneksel kara liste yontemlerinin aksine, <em>yazim stili analizi (stilometri)</em> ve ';
    html += '<em>derin ogrenme (BiLSTM + GloVe)</em> kombinasyonuyla daha once gorulmemis saldirilari bile yakalayabilir.';
    html += '</p>';
    html += '<div class="hiw-highlight-row">';
    html += '<div class="hiw-highlight"><div class="hiw-highlight-icon" style="color:var(--accent-blue)">9</div><div class="hiw-highlight-label">ISLEM ADIMI</div></div>';
    html += '<div class="hiw-highlight"><div class="hiw-highlight-icon" style="color:var(--accent-green)">%96.9</div><div class="hiw-highlight-label">ACCURACY</div></div>';
    html += '<div class="hiw-highlight"><div class="hiw-highlight-icon" style="color:var(--accent-purple)">0.9944</div><div class="hiw-highlight-label">ROC-AUC</div></div>';
    html += '<div class="hiw-highlight"><div class="hiw-highlight-icon" style="color:var(--accent-cyan)">2</div><div class="hiw-highlight-label">DAL (DUAL BRANCH)</div></div>';
    html += '</div></div>';

    html += '<div class="card">';
    html += '<div class="card-title">Karar Mekanizmasi &mdash; Adaptif Esik</div>';
    html += '<div class="hiw-threshold">';
    html += '<div class="hiw-threshold-bar">';
    html += '<div class="hiw-zone safe">SAFE<br><small>P &le; 0.35</small></div>';
    html += '<div class="hiw-zone uncertain">UNCERTAIN<br><small>0.35 &lt; P &lt; 0.65</small></div>';
    html += '<div class="hiw-zone phishing">PHISHING<br><small>P &ge; 0.65</small></div>';
    html += '</div>';
    html += '<div class="hiw-threshold-legend">';
    html += '<div><span style="color:var(--accent-green)">&#9632;</span> XGBoost kesin safe &rarr; Direkt SAFE (Hizli Yol)</div>';
    html += '<div><span style="color:var(--accent-red)">&#9632;</span> XGBoost kesin phishing &rarr; Direkt PHISHING (Hizli Yol)</div>';
    html += '<div><span style="color:var(--accent-yellow)">&#9632;</span> Belirsiz &rarr; BiLSTM + Meta-Classifier devreye girer (Tam Pipeline)</div>';
    html += '</div></div></div>';
    html += '</div>';

    // PIPELINE
    html += '<div id="hiw-pipeline" class="hiw-tab-pane">';
    html += '<div class="card">';
    html += '<div class="card-title">9 Adimli Pipeline Akisi</div>';
    html += '<div class="hiw-pipeline">';
    html += renderAgentSteps();
    html += '</div></div></div>';

    // NLP
    html += '<div id="hiw-nlp" class="hiw-tab-pane">';
    html += '<div class="card">';
    html += '<div class="card-title">NLP Metin Isleme Adimlari</div>';
    html += '<p style="color:var(--text-secondary);font-size:0.72rem;margin-bottom:8px">E-posta metni BiLSTM modeline girmeden once asagidaki NLP on isleme adimlarindan gecer:</p>';
    html += '<div class="hiw-steps">';
    html += renderNLPSteps();
    html += '</div></div></div>';

    // STILOMETRI
    html += '<div id="hiw-stylometric" class="hiw-tab-pane">';
    html += '<div class="card">';
    html += '<div class="card-title">Stilometrik Feature Extraction (Adim 3)</div>';
    html += '<p style="color:var(--text-secondary);font-size:0.72rem;margin-bottom:8px">';
    html += 'Metnin <strong>nasil yazildigini</strong> analiz eder. Icerikten bagimsiz, 13 boyutlu yazim stili vektoru cikarir.';
    html += '</p>';
    html += '<div class="hiw-feature-grid">';
    html += renderFeatureTable();
    html += '</div></div></div>';

    // KARAR
    html += '<div id="hiw-threshold" class="hiw-tab-pane">';
    html += '<div class="card">';
    html += '<div class="card-title">Karar Mekanizmasi &mdash; Adaptif Esik</div>';
    html += '<div class="hiw-threshold">';
    html += '<div class="hiw-threshold-bar">';
    html += '<div class="hiw-zone safe">SAFE<br><small>P &le; 0.35</small></div>';
    html += '<div class="hiw-zone uncertain">UNCERTAIN<br><small>0.35 &lt; P &lt; 0.65</small></div>';
    html += '<div class="hiw-zone phishing">PHISHING<br><small>P &ge; 0.65</small></div>';
    html += '</div>';
    html += '<div class="hiw-threshold-legend">';
    html += '<div><span style="color:var(--accent-green)">&#9632;</span> XGBoost kesin safe &rarr; Direkt SAFE (Hizli Yol)</div>';
    html += '<div><span style="color:var(--accent-red)">&#9632;</span> XGBoost kesin phishing &rarr; Direkt PHISHING (Hizli Yol)</div>';
    html += '<div><span style="color:var(--accent-yellow)">&#9632;</span> Belirsiz &rarr; BiLSTM + Meta-Classifier devreye girer (Tam Pipeline)</div>';
    html += '</div></div></div>';

    html += '<div class="card">';
    html += '<div class="card-title">Meta-Classifier Stacking (Adim 8)</div>';
    html += '<div class="hiw-stacking">';
    html += '<div class="hiw-stack-row">';
    html += '<div class="hiw-stack-item" style="border-color:var(--accent-purple)">XGBoost<br>P1 = Stilometrik Skor</div>';
    html += '<div class="hiw-stack-item" style="border-color:var(--accent-cyan)">BiLSTM<br>P2 = Icerik Skoru</div>';
    html += '</div>';
    html += '<div style="text-align:center;color:var(--text-secondary);font-size:0.68rem;padding:6px">&darr; [P1, P2, P1&times;P2, |P1-P2|, (P1+P2)/2] &darr;</div>';
    html += '<div class="hiw-stack-meta">Logistic Regression Meta-Classifier &rarr; P_final</div>';
    html += '</div></div>';
    html += '</div>';

    // MIMARI
    html += '<div id="hiw-arch" class="hiw-tab-pane">';
    html += '<div class="card">';
    html += '<div class="card-title">Dual Branch Mimari (Adim 7)</div>';
    html += '<div class="hiw-arch">';
    html += '<div class="hiw-arch-box input">Padded Sequence [batch, 200]</div>';
    html += '<div class="hiw-arch-split">';
    html += '<div class="hiw-arch-branch glove">';
    html += '<div class="hiw-arch-box">GloVe Embedding<br><small>100d, frozen</small></div>';
    html += '<div class="hiw-arch-box">GlobalAvgPooling<br><small>[batch, 100]</small></div>';
    html += '</div>';
    html += '<div class="hiw-arch-branch bilstm">';
    html += '<div class="hiw-arch-box">BiLSTM Embedding<br><small>100d, trainable</small></div>';
    html += '<div class="hiw-arch-box">Bidirectional LSTM<br><small>128 units</small></div>';
    html += '<div class="hiw-arch-box">Self-Attention<br><small>Bahdanau</small></div>';
    html += '<div class="hiw-arch-box">Context Vector<br><small>[batch, 256]</small></div>';
    html += '</div></div>';
    html += '<div class="hiw-arch-box fusion">Concatenate [batch, 356]</div>';
    html += '<div class="hiw-arch-box">Dense(128) &rarr; Dropout(0.4) &rarr; Dense(64) &rarr; Dropout(0.3)</div>';
    html += '<div class="hiw-arch-box output">Dense(1, sigmoid) &rarr; P(phishing)</div>';
    html += '</div></div></div>';

    html += '</div>'; // close hiw-tab-content

    el.innerHTML = html;
}

function switchHiwTab(tabId, btn) {
    var panes = document.querySelectorAll('.hiw-tab-pane');
    for (var i = 0; i < panes.length; i++) { panes[i].classList.remove('active'); }
    var tabs = document.querySelectorAll('.hiw-tab');
    for (var i = 0; i < tabs.length; i++) { tabs[i].classList.remove('active'); }
    var target = document.getElementById('hiw-' + tabId);
    if (target) { target.classList.add('active'); }
    if (btn) { btn.classList.add('active'); }
    var content = document.querySelector('.hiw-tab-content');
    if (content) { content.scrollTop = 0; }
}

function renderAgentSteps() {
    var agents = [
        { num: 1, name: 'Email Ingestion', desc: 'Ham e-postayi alir, header/body ayristirir', color: 'var(--accent-purple)', icon: '&#9993;' },
        { num: 2, name: 'Parallel Preprocessing', desc: 'Metin temizleme + MD5 hash paralel calisir', color: 'var(--accent-purple)', icon: '&#9889;' },
        { num: 3, name: 'Stylometric Extraction', desc: '13 boyutlu yazim stili vektoru cikarir', color: 'var(--accent-purple)', icon: '&#9997;' },
        { num: 4, name: 'XGBoost Classifier', desc: 'Stilometrik ozelliklerle P(phishing) hesaplar', color: 'var(--accent-purple)', icon: '&#127794;' },
        { num: 5, name: 'Adaptive Threshold', desc: 'Kesin/Belirsiz ayrimi yapar (T_low / T_high)', color: 'var(--accent-yellow)', icon: '&#9878;', decision: true },
        { num: 6, name: 'NLP Preprocessing', desc: 'Tokenization + padding (maxlen=200)', color: 'var(--accent-cyan)', icon: '&#128221;', deep: true },
        { num: 7, name: 'BiLSTM + GloVe + Attention', desc: 'Dual branch derin ogrenme ile icerik analizi', color: 'var(--accent-cyan)', icon: '&#129504;', deep: true },
        { num: 8, name: 'Meta-Classifier', desc: 'XGBoost + BiLSTM skorlarini birlestirir', color: 'var(--accent-cyan)', icon: '&#128279;', deep: true },
        { num: 9, name: 'Sonuc ve Loglama', desc: 'Nihai karar + eylem + aciklama uretir', color: 'var(--accent-green)', icon: '&#128228;' }
    ];

    var out = '';
    for (var i = 0; i < agents.length; i++) {
        var a = agents[i];
        var cls = '';
        if (a.decision) cls = 'decision';
        if (a.deep) cls = 'deep';
        out += '<div class="hiw-agent ' + cls + '">';
        out += '<div class="hiw-agent-num" style="background:' + a.color + '">' + a.num + '</div>';
        out += '<div class="hiw-agent-body">';
        out += '<span class="hiw-agent-icon">' + a.icon + '</span> ';
        out += '<span class="hiw-agent-name">' + a.name + '</span> ';
        out += '<span class="hiw-agent-desc">&mdash; ' + a.desc + '</span>';
        if (a.decision) out += ' <span class="hiw-agent-badge decision">KARAR</span>';
        if (a.deep) out += ' <span class="hiw-agent-badge deep">DERIN</span>';
        out += '</div></div>';
        if (i < agents.length - 1) out += '<div class="hiw-agent-arrow">&#9660;</div>';
    }
    return out;
}

function renderNLPSteps() {
    var steps = [
        { step: 1, name: 'Lowercase', desc: 'Tum harfler kucultulur', input: '"URGENT: Click HERE"', output: '"urgent: click here"', icon: 'Aa' },
        { step: 2, name: 'URL Temizleme', desc: 'http/https linkleri kaldirilir', input: '"click http://evil.com now"', output: '"click  now"', icon: '&#128279;' },
        { step: 3, name: 'E-mail Temizleme', desc: 'user@domain adresleri kaldirilir', input: '"from admin@bank.com"', output: '"from "', icon: '&#9993;' },
        { step: 4, name: 'HTML Tag Temizleme', desc: 'HTML tagleri kaldirilir', input: '"&lt;b&gt;Urgent&lt;/b&gt; action"', output: '"Urgent action"', icon: '&#127991;' },
        { step: 5, name: 'Ozel Karakter Filtre', desc: 'Sadece [a-z] ve bosluk kalir', input: '"alert! $100 prize!!!"', output: '"alert  prize"', icon: '&#128292;' },
        { step: 6, name: 'Bosluk Normalizasyonu', desc: 'Coklu bosluklar tek bosluga indirgenir', input: '"click   here   now"', output: '"click here now"', icon: '&middot;' },
        { step: 7, name: 'Tokenization', desc: 'Kelime &rarr; sayisal indeks', input: '"click here now"', output: '[765, 2341, 890]', icon: '#' },
        { step: 8, name: 'Padding', desc: 'Dizi maxlen=200\'e tamamlanir', input: '[765, 2341, 890]', output: '[765, 2341, 890, 0, ..., 0]', icon: '&#128207;' }
    ];

    var out = '';
    for (var i = 0; i < steps.length; i++) {
        var s = steps[i];
        out += '<div class="hiw-step">';
        out += '<div class="hiw-step-num">' + s.step + '</div>';
        out += '<div class="hiw-step-content">';
        out += '<div class="hiw-step-header">';
        out += '<span class="hiw-step-icon">' + s.icon + '</span> ';
        out += '<span class="hiw-step-name">' + s.name + '</span>';
        out += '<span class="hiw-step-status">&#10004;</span>';
        out += '</div>';
        out += '<div class="hiw-step-desc">' + s.desc + '</div>';
        out += '<div class="hiw-step-example">';
        out += '<div class="hiw-step-io"><span class="hiw-io-label">In:</span> <code>' + s.input + '</code></div>';
        out += '<div class="hiw-step-io"><span class="hiw-io-label">Out:</span> <code>' + s.output + '</code></div>';
        out += '</div></div></div>';
    }
    return out;
}

function renderFeatureTable() {
    var features = [
        { group: 'Kelime', features: ['avg_word_len', 'vocab_richness'], desc: 'Ortalama kelime uzunlugu, kelime zenginligi' },
        { group: 'Cumle', features: ['avg_sent_len', 'sent_count'], desc: 'Ortalama cumle uzunlugu, cumle sayisi' },
        { group: 'Noktalama', features: ['exclaim_rate', 'question_rate'], desc: 'Unlem ve soru isareti orani' },
        { group: 'Harf', features: ['uppercase_ratio', 'capitalized_ratio'], desc: 'Buyuk harf kullanim orani' },
        { group: 'Ozel', features: ['special_char_ratio', 'whitespace_ratio'], desc: 'Ozel karakter ve bosluk orani' },
        { group: 'Sayisal', features: ['digit_ratio', 'words_to_chars'], desc: 'Rakam orani, kelime/karakter orani' }
    ];

    var out = '';
    for (var i = 0; i < features.length; i++) {
        var f = features[i];
        out += '<div class="hiw-feature-item">';
        out += '<div class="hiw-feature-group">' + f.group + '</div>';
        out += '<div class="hiw-feature-names">';
        for (var j = 0; j < f.features.length; j++) {
            out += '<code>' + f.features[j] + '</code> ';
        }
        out += '</div>';
        out += '<div class="hiw-feature-desc">' + f.desc + '</div>';
        out += '</div>';
    }
    return out;
}
