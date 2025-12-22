(function() {
    const form = document.getElementById('ask-form');
    const askBtn = document.getElementById('ask-btn');
    const askSpinner = document.getElementById('ask-spinner');
    const buttons = form ? form.querySelectorAll('button') : [];
    if (form && askBtn && askSpinner) {
        form.addEventListener('submit', function(ev) {
            const action = ev.submitter ? ev.submitter.value : 'ask';
            if (action === 'ask') {
                askSpinner.classList.remove('d-none');
                buttons.forEach(el => el.disabled = true);
            }
        });
    }
})();

(function() {
    const chartDataEl = document.getElementById('chart-data');
    const chartCanvas = document.getElementById('result-chart');
    if (!chartDataEl || !chartCanvas || !window.Chart) return;

    const cfg = JSON.parse(chartDataEl.textContent);
    const randomColor = () => {
        const hue = Math.floor(Math.random() * 360);
        return `hsl(${hue} 70% 55%)`;
    };
    const colors = cfg.labels.map(() => randomColor());
    const primary = randomColor();
    const pointColors = colors.length ? colors : [primary];

    new Chart(chartCanvas, {
        type: cfg.type || 'bar',
        data: {
            labels: cfg.labels,
            datasets: [{
                label: cfg.title || 'Gráfico',
                data: cfg.values,
                backgroundColor: cfg.type === 'pie' || cfg.type === 'bar' ? colors : primary,
                borderColor: cfg.type === 'pie' ? '#0b1220' : primary,
                borderWidth: 1,
                pointBackgroundColor: pointColors,
                pointBorderColor: pointColors
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: cfg.type !== 'bar', position: 'bottom' },
                tooltip: { enabled: true }
            },
            scales: cfg.type === 'pie' ? {} : {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { grid: { display: false } }
            }
        }
    });
})();

(function() {
    const languageHints = new Set([
        'python','sql','json','html','css','js','javascript','typescript',
        'bash','shell','yaml','yml','text','txt'
    ]);
    const containers = document.querySelectorAll('.bot-answer .answer-content');
    containers.forEach((container) => {
        const rawEl = container.querySelector('.answer-raw');
        if (!rawEl) return;

        const raw = rawEl.textContent || '';
        if (!raw.includes('```')) {
            rawEl.classList.add('answer-text');
            return;
        }

        const parts = raw.split('```');
        container.innerHTML = '';
        parts.forEach((part, idx) => {
            if (!part) return;
            if (idx % 2 === 0) {
                if (part.trim()) {
                    const p = document.createElement('p');
                    p.className = 'answer-text';
                    p.textContent = part.trim();
                    container.appendChild(p);
                }
                return;
            }

            let code = part.replace(/^\n+/, '').replace(/\n+$/, '');
            let lang = '';
            const inlineLang = code.match(/^([A-Za-z0-9_-]+)\s+/);
            if (inlineLang && languageHints.has(inlineLang[1].toLowerCase())) {
                lang = inlineLang[1].toLowerCase();
                code = code.slice(inlineLang[0].length);
            } else {
                const firstLineEnd = code.indexOf('\n');
                if (firstLineEnd > -1) {
                    const firstLine = code.slice(0, firstLineEnd).trim();
                    if (languageHints.has(firstLine.toLowerCase())) {
                        lang = firstLine.toLowerCase();
                        code = code.slice(firstLineEnd + 1);
                    }
                }
            }

            const pre = document.createElement('pre');
            if (lang) pre.setAttribute('data-lang', lang);
            const codeEl = document.createElement('code');
            codeEl.textContent = code;
            pre.appendChild(codeEl);
            container.appendChild(pre);
        });
    });
})();
