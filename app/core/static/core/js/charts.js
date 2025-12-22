(function() {
    const tableSelect = document.getElementById('table-select');
    if (tableSelect) {
        tableSelect.addEventListener('change', function() {
            if (tableSelect.value) {
                const form = tableSelect.closest('form');
                if (form) {
                    form.submit();
                }
            }
        });
    }
})();

(function() {
    const aggSelect = document.getElementById('agg-select');
    const valueSelect = document.getElementById('value-select');
    if (!aggSelect || !valueSelect) return;
    const toggleValue = () => {
        const isCount = aggSelect.value === 'count';
        valueSelect.disabled = isCount;
        if (isCount) {
            valueSelect.value = '';
        }
    };
    aggSelect.addEventListener('change', toggleValue);
    toggleValue();
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
