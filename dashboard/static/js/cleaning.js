(function () {
    const plotEl = document.getElementById("plot-data");
    
    if (!plotEl || typeof Plotly === "undefined") {
        console.error("Mancano i dati (plot-data) o Plotly");
        return;
    }
    
    const D = JSON.parse(plotEl.textContent);
    
    const jobEl = document.getElementById("job-data");
    const jobId = jobEl ? JSON.parse(jobEl.textContent).job_id : null;

    const P = D.points;
    const docs = [...new Set(P.map(p => p.doc))];
    
    const rels = {};
    const relClass = relation => String(relation || "").replace(/\s+/g, "-");

    const colors = {
        "contraddizione": "#b91c1c", 
        "complementare": "#16a34a", 
        "equivalente": "#2563eb",
        "non correlato": "#6b7280", 
        "": "#f59e0b", 
        "invalid": "#a16207"
    };


    function addLines(traces, pairs, name, color, dash, width) {
        if (!pairs.length) return;
        const x = []
        const y = []
        const z = [];

        pairs.forEach(([i, j]) => {
            if (!P[i] || !P[j]) return;

            x.push(P[i].x, P[j].x, null);
            y.push(P[i].y, P[j].y, null);
            z.push(P[i].z, P[j].z, null);
        });

        traces.push({
            type: "scatter3d", 
            mode: "lines", name, x, y, z, 
            hoverinfo: "skip", 
            line: { 
                color, width, dash 
            } 
        });
    }

    function buildTraces() {
        const traces = [];

        docs.forEach(doc => {
            
            const points = P.filter(point => point.doc === doc && !point.frag);
            
            if (!points.length) return;

            traces.push({
                type: "scatter3d", 
                mode: "markers", 
                name: doc,
                x: points.map(point => point.x), 
                y: points.map(point => point.y), 
                z: points.map(point => point.z),
                text: points.map(point => `${point.id}<br>${point.text}`),
                hoverinfo: "text",
                marker: {
                    size: points.map(point => point.group >= 0 ? 6 : 3.5), 
                    opacity: 0.9,
                    line: { 
                        width: 1, 
                        color: points.map(p => p.group >= 0 ? "#c00" : "rgba(0,0,0,0)") 
                    }
                }
            });
        });

        const fragment = P.filter(point => point.frag);
        if (fragment.length) {
            traces.push({
                type: "scatter3d", 
                mode: "markers", 
                name: "frammenti", 
                x: fragment.map(point => point.x), 
                y: fragment.map(point => point.y), 
                z: fragment.map(point => point.z),
                text: fragment.map(point => `${point.id}<br>${point.text}`), 
                hoverinfo: "text",
                marker: {
                    size: 3, 
                    symbol: "x", 
                    color: "#9ca3af"
                }
            });
        }

        // Near duplicate: arc
        addLines(traces, D.edges, "near-duplicated", "#dc2626", "solid", 4);

        // Judge's pairs: arc
        Object.entries(colors).forEach(([relation, color]) => {
            const pairs = D.cands.filter(candidate => (rels[candidate.key] || "") === relation).map(candidate => [candidate.i, candidate.j]);
            addLines(traces, pairs, relation ? `Candidate: ${relation}` : "Candidate not judge", color, "dash", 3);
        });
        return traces;
    }

    const layout = {
        margin: { l: 0, r: 0, t: 10, b: 0 }, 
        legend: { font: { size: 10 } },
        uirevision: "keep",
        scene: { 
            xaxis: { title: "PC1" }, 
            yaxis: { title: "PC2" }, 
            zaxis: { title: "PC3" } 
        }
    };

    Plotly.newPlot("plot", buildTraces(), layout,  {responsive: true});

    if (!jobId) return;

    const cards = new Map([...document.querySelectorAll(".candidate-card")].map(card => [card.dataset.key, card]));
    const applied = new Set();
    const kpi = document.getElementById("kpi-llm");

    function showJudgment(key, judgement) {
        const card = cards.get(key);
        if (!card) return;

        const slot = card.querySelector(".rel-slot");
        if (slot) {
            slot.textContent = "";

            const badge = document.createElement("span");
            badge.className = "badge tag-relation " + relClass(judgement.relation);
            badge.textContent = judgement.relation;
            slot.appendChild(badge);
        }
          
        const explanationBox = card.querySelector(".explanation-box");
        if (explanationBox && judgement.explanation) {
            explanationBox.textContent = "";

            const label = document.createElement("strong");
            label.textContent = "Judge explanation: ";
            
            explanationBox.append(label, document.createTextNode(judgement.explanation));
            explanationBox.className = "explanation-box mt-3 expl-" + relClass(judgement.relation);
            explanationBox.hidden = false;
        }
    }

    function updateKpi(status) {
        const progressEl = document.getElementById("llm-progress");
        const barEl = document.getElementById("llm-bar");
        const subEl = document.getElementById("llm-sub");
        if (!progressEl || !barEl || !subEl) return;

        const judgements = Object.values(status.items || {});
        const contradictions = judgements.filter(judgement => judgement.relation === "contraddizione").length;
       
        progressEl.textContent = `${status.done} / ${status.total}`;

        barEl.style.width = `${status.total ? (100 * status.done) / status.total : 0}%`;
        
        const contradictionShare = status.done ? `(${Math.round((100*contradictions)/status.done)}%)` : "";
        const base = `contradiction: ${contradictions}` + contradictionShare;
        
        if (status.status === "running") {
            subEl.textContent = `in corso… ${base} · ${status.elapsed} s`;
        } else if (status.status === "done") {
            subEl.textContent = `completato · ${base} · ${status.elapsed} s`;
            kpi.classList.add("done");
        } else {
            subEl.textContent = "errore: " + (status.error || status.status);
            if (kpi) kpi.classList.add("error");
        }
    }

    async function poll() {
        let status;
        try {
            const response = await fetch(`/status/${jobId}`);
            if (response.status === 404) return;
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            status = await response.json();
        } catch (error) {
            console.error("polling failed", error);
            return setTimeout(poll, 10000);
        }

        let changed = false;
        
        Object.entries(status.items || {}).forEach(([key, judgement]) => {
            if (applied.has(key)) return;
            
            applied.add(key);
            rels[key] = judgement.relation;

            showJudgment(key, judgement);
            changed = true;
        });

        if (changed) {
            Plotly.react("plot", buildTraces(), layout);
        }

        updateKpi(status);

        if (status.status === "running") {
            setTimeout(poll, 10000);
        }
    }
    poll();
}) ();