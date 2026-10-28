const groups = [
    {titolo: "Contenuto", dims: [
        ["Copertura tematica", "Ogni area della Base è sufficientemente coperta? Esistono buchi di copertura?"],
        ["Ridondanza", "All'interno della Base sono presenti contenuti ridondanti che possono danneggiare la fase di retrieval? Sono presenti più copie dello stesso documento?"],
        ["Coerenza informativa", "“La Base presenta contenuti tra loro discordanti o contraddittori, che possono generare risposte incoerenti o non adeguatamente supportate dalle fonti?”"]
    ]},
    {titolo: "Lifecycle", dims: [
        ["Attualità", "Nella Base vi sono documenti vecchi che portano ad un crollo delle performance? Vi sono contenuti obsoleti?"],
        ["Qualità della segmentazione", "I documenti presenti nella Base sono stati suddivisi in unità complete, leggibili e autosufficienti, senza perdere dati cruciali?"],
    ]},
    {titolo: "Uso, Recupero e Interazione RAG", dims: [
        ["Recuperabilità", "Il contenuto della Base viene effettivamente recuperato o esistono aree 'morte'?"],
        ["Context Sufficiency", "Data una domanda reale, i chunk che vengono recuperati possiedono abbastanza informazioni per poter rispondere correttamente?"],
        ["Faithfulness", "Le risposte che sono state generate dall'LLM sono sostenute dai passaggi della Base che sono stati recuperati/citati?"],
    ]},   
    {titolo: "Uso reale", dims: [
        ["Qualità percepita", "Gli utenti che utilizzano questa Base, sono soddisfatti o no rispetto alle risposte ottenute?"]
    ]}
];

const book = document.getElementById("book")
const dots = document.getElementById("dots");
const prevBtn = document.getElementById("prev");
const nextBtn = document.getElementById("next");

const dlg = document.getElementById("runDialog");
const runForm = document.getElementById("runForm");
const runSubmit = document.getElementById("runSubmit");
const runNA = document.getElementById("runNA");
const runTitle = document.getElementById("runTitle");
const runSub = document.getElementById("runSub");
const runGroup = document.getElementById("runGroup");

const AVAILABLE = new Set([0, 1]);
const FORM_FIELDS_MAP = {
    0: "runContents",
    1: "runLifecycle"
}

let cur = 0;


function openRun(k) {
    const g = groups[k];
    const isAvailable = AVAILABLE.has(k);

    runTitle.textContent = `Inferenza: ${g.titolo}`;
    runSub.textContent = g.dims.map(d => d[0]).join(" · ");
    
    // Titolo del gruppo
    runGroup.value = g.titolo;
    
    // (De)Attivazione set campi specifici
    Object.entries(FORM_FIELDS_MAP).forEach(([idx, elementId]) => {
        const box = document.getElementById(elementId);
        if (!box) return;

        const isCurrentActive = isAvailable & Number(idx) === k;
        box.hidden = !isCurrentActive;

        // Disabilito gli input nascosti
        box.querySelectorAll("input, select, textarea").forEach(input => input.disabled = !isCurrentActive);
    });

    if (runNA) runNA.hidden = isAvailable;
    if (runSubmit) runSubmit.hidden = !isAvailable;
    
    dlg.showModal();
}

const closeRun = () => dlg.close();
document.getElementById("runClose").onclick = closeRun;
document.getElementById("runCancel").onclick = closeRun;

dlg.addEventListener("click", e => {
    if (e.target === dlg) {
        closeRun();
    }
});

runForm.addEventListener("submit", () => {
    if (runSubmit) {
        runSubmit.disabled = true;
        runSubmit.textContent = "Analisi in corso ....";
    }
});

// Reset stato del button con "indietro" della cronologia 
window.addEventListener("pageshow", () => {
    if (runSubmit) {
        runSubmit.disabled = false;
        runSubmit.textContent = "Avvia analisi";
    }
});

// Generazione dinamica schede del libro
const pages = groups.map((g, k) => {
    const p = document.createElement("div");
    p.className = "page";
    p.innerHTML = `
        <div class="num">Gruppo ${k + 1} / ${groups.length}</div>
        <h2>${g.titolo}</h2>
        ${g.dims.map(d => `<div class="dim"><b>${d[0]}</b><span>${d[1]}</span></div>`).join("")}
        <button class="btn btn-ck run">Lancia inferenza</button>
    `;
    p.querySelector(".run").addEventListener("click", () => openRun(k));
    
    book.appendChild(p);
    
    // Assegna la classe 'i' per accoppiarlo al CSS
    const dot = document.createElement("i");
    dot.className = "i"; 
    dot.addEventListener("click", () => go(k));
    dots.appendChild(dot);
    
    return p;
});

function render() {
    pages.forEach((p, k) => {
        // Applica il reset completo della trasformazione basato sul punto corrente
        p.style.transform = k < cur ? "rotateY(-180deg)" : "rotateY(0deg)";
        p.style.zIndex = k < cur ? k : 100 - k;
    });

    [...dots.children].forEach((d, k) => d.classList.toggle(
        "on", k === cur
    ));

    prevBtn.disabled = cur === 0;
    nextBtn.disabled = cur === pages.length - 1;
}

const go = n => {
    cur = Math.max(0, Math.min(pages.length - 1, n));
    render();
};

prevBtn.onclick = () => go(cur-1);
nextBtn.onclick = () => go(cur+1);

document.addEventListener("keydown", e => {
    if (dlg.open) {
        return;
    };
    if (e.key === "ArrowRight") {
        go(cur+1)
    };
    if (e.key === "ArrowLeft") {
        go(cur-1);
    };
});

// Drag & Swipe
let startX = null;
let dx = 0;

book.addEventListener("pointerdown", e => {
    if (e.target.closest("button")) return;
    startX = e.clientX;
    dx = 0;
    book.classList.add("dragging");
    book.setPointerCapture(e.pointerId);
});

book.addEventListener("pointermove", e => {
    if (startX === null) return;
    dx = e.clientX - startX;
    const w = book.offsetWidth || 1;
    
    if (dx < 0 && cur < pages.length - 1) {
        pages[cur].style.transform = `rotateY(${Math.max(-180, (dx / w) * 180)}deg)`;
    }
    if (dx > 0 && cur > 0) {
        pages[cur - 1].style.transform = `rotateY(${Math.min(0, -180 + (dx / w) * 180)}deg)`;
    }
});

function endDrag(e) {
    if (startX === null) return;
    startX = null;
    book.classList.remove("dragging");
    
    if(e && e.pointerId) {
        try { book.releasePointerCapture(e.pointerId); } catch(_) {}
    }

    // Determina il cambio pagina o il reset
    if (dx < -60 && cur < pages.length - 1) {
        go(cur + 1);
    } else if (dx > 60 && cur > 0) {
        go(cur - 1);
    } else {
        render();
    }
}

book.addEventListener("pointerup", endDrag);
book.addEventListener("pointercancel", endDrag);

render();


