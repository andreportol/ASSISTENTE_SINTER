(function() {
    const tableSelect = document.getElementById('table-select');
    const labelValueSelect = document.getElementById('label-value-select');
    const valueValueSelect = document.getElementById('value-value-select');
    const viewModeSwitches = document.querySelectorAll('.js-view-mode-switch');
    const viewModeInput = document.getElementById('view-mode-input');
    const viewToggles = document.querySelectorAll('.js-view-toggle');
    const generateLabel = document.getElementById('generate-label');
    const generateIcon = document.getElementById('generate-icon');
    const actionInput = document.getElementById('action-input');
    const pageInput = document.getElementById('page-input');
    const limitInput = document.getElementById('limit-input');
    const generateBtn = document.getElementById('generate-btn');
    const tableColsList = document.getElementById('table-cols-list');
    const tableColsFilter = document.getElementById('table-cols-filter');
    const tableColsSelectAll = document.getElementById('table-cols-select-all');
    const tableColsClear = document.getElementById('table-cols-clear');
    const filterList = document.getElementById('filter-list');
    const addFilterBtn = document.getElementById('add-filter');
    const sourceCache = {};

    const submitForm = (el, opts = {}) => {
        const form = (el && el.closest('form')) || document.getElementById('chart-form');
        if (!form) return;
        if (actionInput) {
            actionInput.value = opts.action || 'refresh';
        }
        if (pageInput) {
            if (opts.page) {
                pageInput.value = String(opts.page);
            } else if (opts.resetPage) {
                pageInput.value = '1';
            }
        }
        form.submit();
    };

    const getSourceValues = (input) => {
        if (!input) return [];
        const sourceId = input.dataset.source;
        if (!sourceId) return [];
        if (sourceCache[sourceId]) return sourceCache[sourceId];
        const sourceEl = document.getElementById(sourceId);
        if (!sourceEl) return [];
        try {
            sourceCache[sourceId] = JSON.parse(sourceEl.textContent || '[]');
        } catch (err) {
            sourceCache[sourceId] = [];
        }
        return sourceCache[sourceId];
    };

    const bindTypeahead = (input, results, onCommit) => {
        if (!input || !results) return;
        const MAX_SUGGESTIONS = 25;

        const showSuggestions = (items) => {
            results.innerHTML = '';
            if (!items.length) {
                results.classList.add('d-none');
                return;
            }
            items.forEach((value) => {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.textContent = value;
                btn.addEventListener('click', () => {
                    input.value = value;
                    results.classList.add('d-none');
                    if (onCommit) onCommit(input);
                });
                results.appendChild(btn);
            });
            results.classList.remove('d-none');
        };

        const filterSuggestions = () => {
            const values = getSourceValues(input);
            const term = input.value.trim().toLowerCase();
            const filtered = values
                .filter((value) => !term || String(value).toLowerCase().includes(term))
                .slice(0, MAX_SUGGESTIONS);
            showSuggestions(filtered);
        };

        input.addEventListener('input', filterSuggestions);
        input.addEventListener('focus', filterSuggestions);
        input.addEventListener('change', () => {
            if (onCommit) onCommit(input);
        });
        input.addEventListener('keydown', (ev) => {
            if (ev.key === 'Enter') {
                ev.preventDefault();
                if (onCommit) onCommit(input);
            }
        });

        document.addEventListener('click', (ev) => {
            if (results.contains(ev.target) || input.contains(ev.target)) {
                return;
            }
            results.classList.add('d-none');
        });
    };

    const setupTypeahead = (inputId, resultsId, onCommit) => {
        const input = document.getElementById(inputId);
        const results = document.getElementById(resultsId);
        if (!input || !results) return null;
        bindTypeahead(input, results, onCommit);
        return input;
    };

    const updateRemoveButtons = () => {
        if (!filterList) return;
        const rows = filterList.querySelectorAll('.filter-row');
        rows.forEach((row) => {
            const btn = row.querySelector('.btn-remove-filter');
            if (btn) btn.disabled = rows.length <= 1;
        });
    };

    const renumberFilterRows = () => {
        if (!filterList) return;
        const rows = filterList.querySelectorAll('.filter-row');
        rows.forEach((row, idx) => {
            const andRadio = row.querySelector('.filter-op-radio[data-op="and"]');
            const orRadio = row.querySelector('.filter-op-radio[data-op="or"]');
            const andLabel = row.querySelector('.filter-op-label[data-op="and"]');
            const orLabel = row.querySelector('.filter-op-label[data-op="or"]');
            const opHidden = row.querySelector('.filter-op-value');
            const valueInput = row.querySelector('.filter-value-input');
            const valueList = row.querySelector('datalist');

            if (andRadio && orRadio) {
                andRadio.name = `filter_op_ui_${idx}`;
                orRadio.name = `filter_op_ui_${idx}`;
                andRadio.id = `filter-op-and-${idx}`;
                orRadio.id = `filter-op-or-${idx}`;
                andRadio.disabled = idx === 0;
                orRadio.disabled = idx === 0;
                if (idx === 0 && !andRadio.checked && !orRadio.checked) {
                    andRadio.checked = true;
                }
                if (opHidden && idx === 0) {
                    opHidden.value = 'and';
                }
            }
            if (andLabel && andRadio) {
                andLabel.setAttribute('for', andRadio.id);
            }
            if (orLabel && orRadio) {
                orLabel.setAttribute('for', orRadio.id);
            }
            if (valueInput && valueList) {
                const listId = `filter-values-${idx}`;
                valueList.id = listId;
                valueInput.setAttribute('list', listId);
            }
        });
    };

    const initFilterRow = (row) => {
        if (!row) return;
        const input = row.querySelector('.filter-col-input');
        const results = row.querySelector('.filter-col-results');
        const valueSelect = row.querySelector('.filter-value-select');
        const valueInput = row.querySelector('.filter-value-input');
        const valueList = row.querySelector('datalist');
        const removeBtn = row.querySelector('.btn-remove-filter');
        const opHidden = row.querySelector('.filter-op-value');
        const opRadios = row.querySelectorAll('.filter-op-radio');

        if (input && results) {
            bindTypeahead(input, results, () => {
                if (valueSelect) valueSelect.value = '';
                if (valueInput) valueInput.value = '';
                if (valueList) valueList.innerHTML = '';
                submitForm(input, { resetPage: true });
            });
        }

        if (input) {
            input.addEventListener('change', () => {
                if (valueSelect) valueSelect.value = '';
                if (valueInput) valueInput.value = '';
                if (valueList) valueList.innerHTML = '';
                submitForm(input, { resetPage: true });
            });
        }

        if (opHidden && opRadios.length) {
            opRadios.forEach((radio) => {
                radio.addEventListener('change', () => {
                    if (radio.checked) {
                        opHidden.value = radio.value;
                    }
                });
            });
        }

        if (removeBtn) {
            removeBtn.addEventListener('click', () => {
                row.remove();
                renumberFilterRows();
                updateRemoveButtons();
            });
        }
    };

    const labelInput = setupTypeahead('label-col-input', 'label-col-results', () => {
        if (labelValueSelect) labelValueSelect.value = '';
        submitForm(labelInput, { resetPage: true });
    });

    const valueInput = setupTypeahead('value-col-input', 'value-col-results', () => {
        if (valueValueSelect) valueValueSelect.value = '';
        submitForm(valueInput, { resetPage: true });
    });

    if (filterList) {
        filterList.querySelectorAll('.filter-row').forEach((row) => initFilterRow(row));
        renumberFilterRows();
        updateRemoveButtons();
    }

    if (addFilterBtn && filterList) {
        addFilterBtn.addEventListener('click', () => {
            const rows = filterList.querySelectorAll('.filter-row');
            if (!rows.length) return;
            const template = rows[rows.length - 1];
            const clone = template.cloneNode(true);
            const input = clone.querySelector('.filter-col-input');
            const results = clone.querySelector('.filter-col-results');
            const select = clone.querySelector('.filter-value-select');
            const valueInput = clone.querySelector('.filter-value-input');
            const valueList = clone.querySelector('datalist');
            const opHidden = clone.querySelector('.filter-op-value');
            const opRadios = clone.querySelectorAll('.filter-op-radio');
            if (input) input.value = '';
            if (results) {
                results.innerHTML = '';
                results.classList.add('d-none');
            }
            if (select) {
                select.value = '';
                while (select.options.length > 1) {
                    select.remove(1);
                }
            }
            if (valueInput) valueInput.value = '';
            if (valueList) valueList.innerHTML = '';
            if (opHidden) opHidden.value = 'and';
            if (opRadios.length) {
                opRadios.forEach((radio) => {
                    radio.checked = radio.value === 'and';
                });
            }
            filterList.appendChild(clone);
            initFilterRow(clone);
            renumberFilterRows();
            updateRemoveButtons();
        });
    }

    if (tableSelect) {
        tableSelect.addEventListener('change', function() {
            if (labelInput) labelInput.value = '';
            if (valueInput) valueInput.value = '';
            if (labelValueSelect) labelValueSelect.value = '';
            if (valueValueSelect) valueValueSelect.value = '';

            if (filterList) {
                const rows = filterList.querySelectorAll('.filter-row');
                rows.forEach((row, idx) => {
                    if (idx > 0) {
                        row.remove();
                        return;
                    }
                    const input = row.querySelector('.filter-col-input');
                    const results = row.querySelector('.filter-col-results');
                    const select = row.querySelector('.filter-value-select');
                    const valueInput = row.querySelector('.filter-value-input');
                    const valueList = row.querySelector('datalist');
                    const opHidden = row.querySelector('.filter-op-value');
                    const opRadios = row.querySelectorAll('.filter-op-radio');
                    if (input) input.value = '';
                    if (results) {
                        results.innerHTML = '';
                        results.classList.add('d-none');
                    }
                    if (select) {
                        select.value = '';
                        while (select.options.length > 1) {
                            select.remove(1);
                        }
                    }
                    if (valueInput) valueInput.value = '';
                    if (valueList) valueList.innerHTML = '';
                    if (opHidden) opHidden.value = 'and';
                    if (opRadios.length) {
                        opRadios.forEach((radio) => {
                            radio.checked = radio.value === 'and';
                        });
                    }
                });
                renumberFilterRows();
                updateRemoveButtons();
            }
            submitForm(tableSelect, { resetPage: true });
        });
    }

    if (viewModeSwitches.length && viewModeInput) {
        const setSwitches = (isChart) => {
            viewModeSwitches.forEach((sw) => {
                sw.checked = isChart;
            });
        };
        const syncViewMode = (isChart) => {
            viewModeInput.value = isChart ? 'chart' : 'table';
            viewToggles.forEach((toggle) => {
                toggle.querySelectorAll('.view-toggle__chip').forEach((chip) => {
                    const mode = chip.dataset.mode;
                    chip.classList.toggle('is-active', mode === (isChart ? 'chart' : 'table'));
                });
            });
            if (generateLabel) {
                generateLabel.textContent = isChart ? 'Gerar gráfico' : 'Gerar tabela';
            }
            if (generateIcon) {
                generateIcon.className = isChart ? 'fas fa-chart-bar me-1' : 'fas fa-table me-1';
            }
            if (limitInput) {
                limitInput.disabled = !isChart;
            }
            document.querySelectorAll('[data-view]').forEach((el) => {
                const view = el.dataset.view;
                el.classList.toggle('d-none', view !== (isChart ? 'chart' : 'table'));
            });
            if (tableColsFilter) {
                tableColsFilter.disabled = isChart;
            }
            if (tableColsSelectAll) {
                tableColsSelectAll.disabled = isChart;
            }
            if (tableColsClear) {
                tableColsClear.disabled = isChart;
            }
            if (tableColsList) {
                tableColsList.classList.toggle('is-disabled', isChart);
                tableColsList.querySelectorAll('input[type="checkbox"]').forEach((input) => {
                    input.disabled = isChart;
                });
            }
        };

        const initialIsChart = viewModeInput.value
            ? viewModeInput.value === 'chart'
            : viewModeSwitches[0].checked;
        setSwitches(initialIsChart);
        syncViewMode(initialIsChart);

        viewModeSwitches.forEach((sw) => {
            sw.addEventListener('change', () => {
                const isChart = sw.checked;
                setSwitches(isChart);
                syncViewMode(isChart);
                submitForm(sw, { resetPage: true });
            });
        });

        viewToggles.forEach((toggle) => {
            toggle.querySelectorAll('.view-toggle__chip').forEach((chip) => {
                chip.addEventListener('click', () => {
                    const mode = chip.dataset.mode;
                    const isChart = mode !== 'table';
                    setSwitches(isChart);
                    syncViewMode(isChart);
                    submitForm(toggle, { resetPage: true });
                });
            });
        });
    }

    if (generateBtn) {
        generateBtn.addEventListener('click', () => {
            submitForm(generateBtn, { action: 'generate_chart', resetPage: true });
        });
    }

    if (tableColsList && tableColsFilter) {
        const getVisibleItems = () => Array.from(tableColsList.querySelectorAll('.table-cols-item'))
            .filter((item) => !item.classList.contains('d-none'));
        const filterOptions = () => {
            const term = tableColsFilter.value.trim().toLowerCase();
            tableColsList.querySelectorAll('.table-cols-item').forEach((item) => {
                const text = item.dataset.col || '';
                const checkbox = item.querySelector('input[type="checkbox"]');
                const matches = !term || text.includes(term);
                const keepVisible = matches || (checkbox && checkbox.checked);
                item.classList.toggle('d-none', !keepVisible);
            });
        };
        const setChecked = (items, checked) => {
            items.forEach((item) => {
                const checkbox = item.querySelector('input[type="checkbox"]');
                if (checkbox && !checkbox.disabled) {
                    checkbox.checked = checked;
                }
            });
        };
        tableColsFilter.addEventListener('input', filterOptions);
        tableColsFilter.addEventListener('focus', filterOptions);
        if (tableColsSelectAll) {
            tableColsSelectAll.addEventListener('click', () => {
                setChecked(getVisibleItems(), true);
            });
        }
        if (tableColsClear) {
            tableColsClear.addEventListener('click', () => {
                setChecked(Array.from(tableColsList.querySelectorAll('.table-cols-item')), false);
            });
        }
    }

    document.querySelectorAll('.download-csv').forEach((btn) => {
        btn.addEventListener('click', (ev) => {
            ev.preventDefault();
            submitForm(btn, { action: 'download_csv' });
        });
    });

    document.querySelectorAll('.js-page-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            const page = btn.dataset.page;
            if (!page) return;
            submitForm(btn, { page, action: 'generate_chart' });
        });
    });
})();

(function() {
    const aggSelect = document.getElementById('agg-select');
    const valueInput = document.getElementById('value-col-input');
    const valueValueSelect = document.getElementById('value-value-select');
    if (!aggSelect || !valueInput) return;
    const getSourceValues = (input) => {
        const sourceId = input.dataset.source;
        if (!sourceId) return [];
        const sourceEl = document.getElementById(sourceId);
        if (!sourceEl) return [];
        try {
            return JSON.parse(sourceEl.textContent || '[]');
        } catch (err) {
            return [];
        }
    };
    const toggleValue = () => {
        const isCount = aggSelect.value === 'count';
        valueInput.disabled = false;
        valueInput.dataset.source = isCount ? 'all-columns' : 'value-columns';
        valueInput.placeholder = isCount ? 'Coluna para COUNT (opcional)' : 'Selecione ou digite...';
        if (!isCount && valueInput.value) {
            const numericValues = getSourceValues(valueInput);
            if (!numericValues.includes(valueInput.value)) {
                valueInput.value = '';
            }
        }
        if (valueValueSelect) {
            valueValueSelect.disabled = !valueInput.value;
        }
        if (isCount) {
            if (valueValueSelect) {
                valueValueSelect.value = '';
            }
        }
    };
    aggSelect.addEventListener('change', toggleValue);
    if (valueValueSelect) {
        valueInput.addEventListener('change', () => {
            valueValueSelect.value = '';
        });
    }
    valueInput.addEventListener('input', toggleValue);
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
            maintainAspectRatio: false,
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

    const downloadBtn = document.getElementById('download-chart');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
            const link = document.createElement('a');
            link.href = chartCanvas.toDataURL('image/png');
            link.download = `grafico_${Date.now()}.png`;
            link.click();
        });
    }
})();
