(function () {

    const el = document.getElementById("struct-data");
    if (!el || typeof Plotly === "undefined") { 
        console.error("Mancano i dati (struct-data) o Plotly"); 
        return; 
    }
    const D = JSON.parse(el.textContent);
    const COV = D.cov, GR = D.gran;

    const PAL = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f",
                 "#bcbd22","#17becf","#393b79","#e6550d","#31a354","#756bb1","#636363","#843c39"];
    
    const col = c => PAL[c % PAL.length];
    document.querySelectorAll(".cluster-dot").forEach(dot => {
        dot.style.background = col(parseInt(dot.dataset.cluster, 10));
    });
    const CFG = { 
        responsive: true, 
        displaylogo: false 
    };
    const base = { 
        margin: { 
            l: 55, r: 20, t: 10, b: 50 
        }, 
        paper_bgcolor: "rgba(0,0,0,0)", 
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { 
            size: 11 
        } 
    };
    const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
    const median = a => { 
        const s = [...a].sort((x, y) => x - y), m = s.length >> 1; 
        return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; 
    };

    /* Nuvola per cluster */
    (function () {
        const traces = [];
        COV.names.forEach((name, c) => {
            const pts = COV.points.filter(p => p.c === c);
            if (!pts.length) return;

            traces.push({ 
                type: "scatter3d", 
                mode: "markers", name, legendgroup: "c" + c,
                x: pts.map(p => p.x), y: pts.map(p => p.y), z: pts.map(p => p.z),
                text: pts.map(p => `${name}<br>${p.doc}<br>${p.text}`), 
                hoverinfo: "text",
                marker: { 
                    size: 3.5, opacity: .85, color: col(c) 
                } 
            });
        });

        const cs = COV.centroids;
        traces.push({ 
            type: "scatter3d", 
            mode: "markers+text", name: "centroidi", 
            showlegend: false,
            x: cs.map(p => p.x), y: cs.map(p => p.y), z: cs.map(p => p.z),
            text: cs.map(p => "C" + (p.cluster + 1)), 
            textposition: "top center", 
            hoverinfo: "text",
            marker: { 
                size: 8, symbol: "diamond", 
                color: cs.map(p => col(p.cluster)), 
                line: { 
                    width: 2, color: "#111" 
                } 
            } 
        });
        
        Plotly.newPlot("plot-clusters", traces, {
            margin: { 
                l: 0, r: 0, t: 0, b: 0 
            }, legend: { 
                font: { 
                    size: 10 
                } 
            },
            scene: { 
                xaxis: { 
                    title: "PC1" 
                }, 
                yaxis: { 
                    title: "PC2" 
                }, 
                zaxis: { 
                    title: "PC3"
                } 
            }
        }, CFG);
    })();

    /* Chunk per tema */
    (function () {
        const n = COV.sizes.length, tot = COV.sizes.reduce((a, b) => a + b, 0);
        Plotly.newPlot("plot-sizes", [{
            type: "bar", 
            x: COV.sizes.map((_, c) => "C" + (c + 1)), y: COV.sizes, 
            marker: { 
                color: COV.sizes.map((_, c) => col(c)) 
            },
            text: COV.sizes.map(s => (100 * s / tot).toFixed(1) + "%"), 
            textposition: "outside",
            hovertext: COV.names, hoverinfo: "text+y"
        }], 
        { ...base, 
            yaxis: { title: "chunk" }, xaxis: { title: "tema" },
            shapes: [{ 
                type: "line", xref: "paper", 
                x0: 0, x1: 1, 
                y0: COV.uniform, y1: COV.uniform,
                line: { 
                    dash: "dash", color: "#64748b", width: 1.5 
                } 
            }],
            annotations: [{ 
                xref: "paper", x: 1, 
                y: COV.uniform, yanchor: "bottom", xanchor: "right", 
                showarrow: false,
                text: "quota uniforme", 
                font: { 
                    size: 10, color: "#64748b" 
                } 
            }] 
        }, CFG);
    })();

    (function () {
        const s = COV.scan;
        Plotly.newPlot("plot-kscan", [{ 
            type: "scatter", mode: "lines+markers", name: "silhouette", x: s.map(r => r.k), y: s.map(r => r.silhouette),
            line: { 
                color: "#0ea5e9" 
            }, yaxis: "y" 
        },
        { 
            type: "scatter", mode: "lines+markers", name: "inerzia", x: s.map(r => r.k), y: s.map(r => r.inertia),
            line: { 
                color: "#f59e0b", dash: "dot" 
            }, yaxis: "y2" 
        }], 
        { ...base, margin: { 
            l: 55, r: 55, t: 10, b: 50 
        }, 
        xaxis: { 
            title: "k", dtick: 1 
        },
        yaxis: { title: "silhouette" }, yaxis2: { title: "inerzia", overlaying: "y", side: "right", showgrid: false },
        legend: { orientation: "h", y: 1.12 },
        shapes: [{ type: "line", yref: "paper", y0: 0, y1: 1, x0: COV.k, x1: COV.k, line: { color: "#dc2626", width: 2, dash: "dash" } }],
        annotations: [{ x: COV.k, yref: "paper", y: 1, text: "k scelto", showarrow: false, yanchor: "bottom", font: { color: "#dc2626", size: 10 } }] }, CFG);
    })();

    /* Heatmap tema x documento */
    (function () {
        const h = COV.heat;
        Plotly.newPlot("plot-heat", [{
            type: "heatmap", x: h.docs, y: COV.names.map((_, c) => "C" + (c + 1)), z: h.z, colorscale: "Blues",
            hovertemplate: "%{y} · %{x}<br>%{z} chunk<extra></extra>", colorbar: { title: "chunk" }
        }], { ...base, margin: { l: 50, r: 20, t: 10, b: 130 }, xaxis: { tickangle: -45, automargin: true },
              yaxis: { autorange: "reversed" } }, CFG);
    })();

    /* Istogramma lunghezze */
    (function () {
        const vals = GR.tokens, unit = "token", limit = GR.limit;
        const m = mean(vals), md = median(vals);
        const line = (x, color, dash, text, ypos) => ({
            shape: { type: "line", yref: "paper", y0: 0, y1: 1, x0: x, x1: x, line: { color, width: 2, dash } },
            ann: { x, yref: "paper", y: ypos, text, showarrow: false, xanchor: "left", font: { size: 10, color } } });
        const L = [line(m, "#0ea5e9", "solid", `media ${m.toFixed(0)}`, 1), line(md, "#16a34a", "dash", `mediana ${md.toFixed(0)}`, .92),
                   line(limit, "#dc2626", "dot", `limite modello ${limit}`, .84)];
        Plotly.newPlot("plot-hist", [{ type: "histogram", x: vals, nbinsx: 50, marker: { color: "#94a3b8" },
            hovertemplate: `%{x} ${unit}: %{y} chunk<extra></extra>` }],
            { ...base, bargap: .03, xaxis: { title: `lunghezza (${unit})` }, yaxis: { title: "chunk" },
              shapes: L.map(l => l.shape), annotations: L.map(l => l.ann) }, CFG);
    })();

    /* Anomalie */
    (function () {
        const f = [...GR.flags].reverse();
        Plotly.newPlot("plot-flags", [{
            type: "bar", orientation: "h", x: f.map(r => 100 * r.share), y: f.map(r => r.label),
            marker: { color: f.map(r => r.label === "No anomalies" ? "#22c55e" : "#f97316") },
            text: f.map(r => `${(100 * r.share).toFixed(1)}%  (${r.count})`), textposition: "auto", hoverinfo: "text"
        }], { ...base, margin: { l: 200, r: 20, t: 10, b: 45 }, xaxis: { title: "% dei chunk", range: [0, 100] }, yaxis: { automargin: true } }, CFG);
    })();

    /* Boxplot per documento */
    (function () {
        const names = Object.keys(GR.box);
        Plotly.newPlot("plot-box", names.map((d, i) => ({
            type: "box", y: GR.box[d], name: d.length > 18 ? d.slice(0, 17) + "…" : d, boxpoints: false,
            marker: { color: PAL[i % PAL.length] }, hovertemplate: d + "<br>%{y} token<extra></extra>"
        })), { ...base, margin: { l: 55, r: 20, t: 10, b: 110 }, showlegend: false, yaxis: { title: "token" }, xaxis: { tickangle: -45 },
               shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: GR.limit, y1: GR.limit, line: { color: "#dc2626", width: 1.5, dash: "dot" } }] }, CFG);
    })();
}) ();