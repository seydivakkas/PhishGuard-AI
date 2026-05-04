/**
 * Nasıl Çalışır — Pipeline Görselleştirme
 * Projenin amaç, sebep-sonuç, işlem adımları ve NLP akışını gösterir.
 */

function renderHowItWorks() {
    const el = document.getElementById('page-howitworks');
    if (!el) return;
    el.innerHTML = `
    <!-- AMAÇ -->
    <div class="card">
        <div class="card-title">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="var(--accent-blue)" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4m0-4h.01"/></svg>
            Projenin Amacı
        </div>
        <p style="color:var(--text-secondary);line-height:1.8;font-size:0.92rem">
            Bu sistem, e-posta içeriklerini <strong style="color:var(--accent-blue)">9 adımlı çok katmanlı bir pipeline</strong> kullanarak analiz eder ve
            <strong style="color:var(--accent-red)">phishing (oltalama)</strong> saldırılarını tespit eder.
            Geleneksel kara liste yöntemlerinin aksine, <em>yazım stili analizi (stilometri)</em> ve
            <em>derin öğrenme (BiLSTM + GloVe)</em> kombinasyonuyla daha önce görülmemiş saldırıları bile yakalayabilir.
        </p>
        <div class="hiw-highlight-row">
            <div class="hiw-highlight">
                <div class="hiw-highlight-icon" style="color:var(--accent-blue)">9</div>
                <div class="hiw-highlight-label">İşlem Adımı</div>
            </div>
            <div class="hiw-highlight">
                <div class="hiw-highlight-icon" style="color:var(--accent-green)">%96.9</div>
                <div class="hiw-highlight-label">Accuracy</div>
            </div>
            <div class="hiw-highlight">
                <div class="hiw-highlight-icon" style="color:var(--accent-purple)">0.9944</div>
                <div class="hiw-highlight-label">ROC-AUC</div>
            </div>
            <div class="hiw-highlight">
                <div class="hiw-highlight-icon" style="color:var(--accent-cyan)">2</div>
                <div class="hiw-highlight-label">Dal (Dual Branch)</div>
            </div>
        </div>
    </div>

    <!-- ANA PIPELINE -->
    <div class="card">
        <div class="card-title">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="var(--accent-purple)" stroke-width="2"><path d="M12 2v20M2 12h20"/></svg>
            9 Adımlı Pipeline Akışı
        </div>
        <div class="hiw-pipeline">
            ${renderAgentSteps()}
        </div>
    </div>

    <!-- NLP İŞLEM ADIMLARI -->
    <div class="card">
        <div class="card-title">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="var(--accent-cyan)" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
            NLP Metin İşleme Adımları
        </div>
        <p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:16px">
            E-posta metni BiLSTM modeline girmeden önce aşağıdaki NLP ön işleme adımlarından geçer:
        </p>
        <div class="hiw-steps">
            ${renderNLPSteps()}
        </div>
    </div>

    <!-- STİLOMETRİK ANALİZ -->
    <div class="card">
        <div class="card-title">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="var(--accent-yellow)" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
            Stilometrik Feature Extraction (Adım 3)
        </div>
        <p style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:16px">
            Metnin <strong>nasıl yazıldığını</strong> analiz eder. İçerikten bağımsız, 13 boyutlu yazım stili vektörü çıkarır.
            <span style="color:var(--accent-yellow)">⚠ Bu adımda temizleme yapılmaz — ham metin kullanılır!</span>
        </p>
        <div class="hiw-feature-grid">
            ${renderFeatureTable()}
        </div>
    </div>

    <!-- KARAR MEKANİZMASI -->
    <div class="card">
        <div class="card-title">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="var(--accent-green)" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
            Karar Mekanizması — Adaptif Eşik
        </div>
        <div class="hiw-threshold">
            <div class="hiw-threshold-bar">
                <div class="hiw-zone safe">SAFE<br><small>P &le; 0.35</small></div>
                <div class="hiw-zone uncertain">UNCERTAIN<br><small>0.35 &lt; P &lt; 0.65</small></div>
                <div class="hiw-zone phishing">PHISHING<br><small>P &ge; 0.65</small></div>
            </div>
            <div class="hiw-threshold-legend">
                <div><span style="color:var(--accent-green)">■</span> XGBoost kesin safe → Direkt SAFE (Hızlı Yol)</div>
                <div><span style="color:var(--accent-red)">■</span> XGBoost kesin phishing → Direkt PHISHING (Hızlı Yol)</div>
                <div><span style="color:var(--accent-yellow)">■</span> Belirsiz → BiLSTM + Meta-Classifier devreye girer (Tam Pipeline)</div>
            </div>
        </div>
    </div>

    <!-- DUAL BRANCH MİMARİ -->
    <div class="card">
        <div class="card-title">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="var(--accent-blue)" stroke-width="2"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><path d="M8.59 13.51 15.42 17.49M15.41 6.51 8.59 10.49"/></svg>
            Dual Branch Mimari (Adım 7)
        </div>
        <div class="hiw-arch">
            <div class="hiw-arch-box input">Padded Sequence [batch, 200]</div>
            <div class="hiw-arch-split">
                <div class="hiw-arch-branch glove">
                    <div class="hiw-arch-box">GloVe Embedding<br><small>100d, frozen</small></div>
                    <div class="hiw-arch-box">GlobalAvgPooling<br><small>[batch, 100]</small></div>
                </div>
                <div class="hiw-arch-branch bilstm">
                    <div class="hiw-arch-box">BiLSTM Embedding<br><small>100d, trainable</small></div>
                    <div class="hiw-arch-box">Bidirectional LSTM<br><small>128 units ↔</small></div>
                    <div class="hiw-arch-box">Self-Attention<br><small>Bahdanau</small></div>
                    <div class="hiw-arch-box">Context Vector<br><small>[batch, 256]</small></div>
                </div>
            </div>
            <div class="hiw-arch-box fusion">Concatenate [batch, 356]</div>
            <div class="hiw-arch-box">Dense(128) → Dropout(0.4) → Dense(64) → Dropout(0.3)</div>
            <div class="hiw-arch-box output">Dense(1, sigmoid) → P(phishing)</div>
        </div>
    </div>

    <!-- STACKING -->
    <div class="card">
        <div class="card-title">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="var(--accent-purple)" stroke-width="2"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 3h-8l-2 4h12z"/></svg>
            Meta-Classifier Stacking (Adım 8)
        </div>
        <div class="hiw-stacking">
            <div class="hiw-stack-row">
                <div class="hiw-stack-item" style="border-color:var(--accent-purple)">XGBoost<br>P₁ = Stilometrik Skor</div>
                <div class="hiw-stack-item" style="border-color:var(--accent-cyan)">BiLSTM<br>P₂ = İçerik Skoru</div>
            </div>
            <div style="text-align:center;color:var(--text-secondary);font-size:0.8rem;padding:8px">
                ↓ [P₁, P₂, P₁×P₂, |P₁−P₂|, (P₁+P₂)/2] ↓
            </div>
            <div class="hiw-stack-meta">
                Logistic Regression Meta-Classifier → P_final
            </div>
        </div>
    </div>
    `;
}

function renderAgentSteps() {
    const agents = [
        { num: 1, name: 'Email Ingestion', desc: 'Ham e-postayı alır, header/body ayrıştırır', color: 'var(--accent-purple)', icon: '📧' },
        { num: 2, name: 'Parallel Preprocessing', desc: 'Metin temizleme + MD5 hash paralel çalışır', color: 'var(--accent-purple)', icon: '⚡' },
        { num: 3, name: 'Stylometric Extraction', desc: '13 boyutlu yazım stili vektörü çıkarır', color: 'var(--accent-purple)', icon: '✍️' },
        { num: 4, name: 'XGBoost Classifier', desc: 'Stilometrik özelliklerle P(phishing) hesaplar', color: 'var(--accent-purple)', icon: '🌲' },
        { num: 5, name: 'Adaptive Threshold', desc: 'Kesin/Belirsiz ayrımı yapar (T_low / T_high)', color: 'var(--accent-yellow)', icon: '⚖️', decision: true },
        { num: 6, name: 'NLP Preprocessing', desc: 'Tokenization + padding (maxlen=200)', color: 'var(--accent-cyan)', icon: '📝', deep: true },
        { num: 7, name: 'BiLSTM + GloVe + Attention', desc: 'Dual branch derin öğrenme ile içerik analizi', color: 'var(--accent-cyan)', icon: '🧠', deep: true },
        { num: 8, name: 'Meta-Classifier', desc: 'XGBoost + BiLSTM skorlarını birleştirir', color: 'var(--accent-cyan)', icon: '🔗', deep: true },
        { num: 9, name: 'Sonuç ve Loglama', desc: 'Nihai karar + eylem + açıklama üretir', color: 'var(--accent-green)', icon: '📤' },
    ];

    return agents.map((a, i) => `
        <div class="hiw-agent ${a.decision ? 'decision' : ''} ${a.deep ? 'deep' : ''}">
            <div class="hiw-agent-num" style="background:${a.color}">${a.num}</div>
            <div class="hiw-agent-body">
                <div class="hiw-agent-icon">${a.icon}</div>
                <div class="hiw-agent-name">Adım ${a.num}: ${a.name}</div>
                <div class="hiw-agent-desc">${a.desc}</div>
                ${a.decision ? '<div class="hiw-agent-badge decision">KARAR NOKTASI</div>' : ''}
                ${a.deep ? '<div class="hiw-agent-badge deep">UNCERTAIN ise çalışır</div>' : ''}
            </div>
        </div>
        ${i < agents.length - 1 ? '<div class="hiw-agent-arrow">▼</div>' : ''}
    `).join('');
}

function renderNLPSteps() {
    const steps = [
        { step: 1, name: 'Lowercase', desc: 'Tüm harfler küçültülür', input: '"URGENT: Click HERE"', output: '"urgent: click here"', icon: 'Aa', status: true },
        { step: 2, name: 'URL Temizleme', desc: 'http/https linkleri kaldırılır', input: '"click http://evil.com now"', output: '"click  now"', icon: '🔗', status: true },
        { step: 3, name: 'E-mail Temizleme', desc: 'user@domain adresleri kaldırılır', input: '"from admin@bank.com"', output: '"from "', icon: '📧', status: true },
        { step: 4, name: 'HTML Tag Temizleme', desc: '<div>, <a>, <br> vb. kaldırılır', input: '"<b>Urgent</b> action"', output: '"Urgent action"', icon: '🏷️', status: true },
        { step: 5, name: 'Özel Karakter Filtre', desc: 'Sadece [a-z] ve boşluk kalır', input: '"alert! $100 prize!!!"', output: '"alert  prize"', icon: '🔤', status: true },
        { step: 6, name: 'Boşluk Normalizasyonu', desc: 'Çoklu boşluklar tek boşluğa indirgenir', input: '"click   here   now"', output: '"click here now"', icon: '⎵', status: true },
        { step: 7, name: 'Tokenization', desc: 'Kelime → sayısal indeks (tokenizer.pkl)', input: '"click here now"', output: '[765, 2341, 890]', icon: '#️⃣', status: true },
        { step: 8, name: 'Padding', desc: 'Dizi maxlen=200\'e tamamlanır', input: '[765, 2341, 890]', output: '[765, 2341, 890, 0, 0, ..., 0]', icon: '📏', status: true },
    ];

    return steps.map(s => `
        <div class="hiw-step">
            <div class="hiw-step-num">${s.step}</div>
            <div class="hiw-step-content">
                <div class="hiw-step-header">
                    <span class="hiw-step-icon">${s.icon}</span>
                    <span class="hiw-step-name">${s.name}</span>
                    <span class="hiw-step-status">${s.status ? '✅' : '❌'}</span>
                </div>
                <div class="hiw-step-desc">${s.desc}</div>
                <div class="hiw-step-example">
                    <div class="hiw-step-io"><span class="hiw-io-label">Girdi:</span> <code>${s.input}</code></div>
                    <div class="hiw-step-io"><span class="hiw-io-label">Çıktı:</span> <code>${s.output}</code></div>
                </div>
            </div>
        </div>
    `).join('');
}

function renderFeatureTable() {
    const features = [
        { group: 'Kelime', features: ['avg_word_len', 'vocab_richness'], desc: 'Ortalama kelime uzunluğu, kelime zenginliği' },
        { group: 'Cümle', features: ['avg_sent_len', 'sent_count'], desc: 'Ortalama cümle uzunluğu, cümle sayısı' },
        { group: 'Noktalama', features: ['exclaim_rate', 'question_rate'], desc: 'Ünlem ve soru işareti oranı' },
        { group: 'Harf', features: ['uppercase_ratio', 'capitalized_ratio'], desc: 'Büyük harf kullanım oranı' },
        { group: 'Özel', features: ['special_char_ratio', 'whitespace_ratio'], desc: 'Özel karakter ve boşluk oranı' },
        { group: 'Sayısal', features: ['digit_ratio', 'words_to_chars'], desc: 'Rakam oranı, kelime/karakter oranı' },
    ];

    return features.map(f => `
        <div class="hiw-feature-item">
            <div class="hiw-feature-group">${f.group}</div>
            <div class="hiw-feature-names">${f.features.map(n => `<code>${n}</code>`).join(' ')}</div>
            <div class="hiw-feature-desc">${f.desc}</div>
        </div>
    `).join('');
}
