const groups = [
    {titolo: "Contenuto", dims: [
        ["Ridondanza e duplicazione", "Quanti contenuti sono duplicati o quasi."],
        ["Coerenza interna", "Assenza di affermazioni contraddittorie."]
    ]},
    {titolo: "Struttura", dims: [
        ["Copertura tematica", "Quanto la base documentale copre gli argomenti dell'area."],
        ["Granularità e segmentazione", "Qualità della segmentazione in chunk."],
    ]},
    {titolo: "Tempo e accesso", dims: [
        ["Recuperabilità", "Quanto il motore di ricerca raggiunge i contenuti."],
        ["Qualità percepita", "Riscontro dagli utenti in esercizio."]
    ]},   
    {titolo: "Uso reale", dims: [
        ["Freschezza e obsolescenza", "Quanto sono attuali le informazioni."],
        ["Fondatezza delle risposte", "Quanto le risposte sono sostenute dalle fonti."],
    ]}
];

const book = document.getElementById("book"), dots = document.getElementById("dots");
const prevBtn = document.getElementById("prev"), nextBtn = document.getElementById("next");

const dlg = document.getElementById("runDialog");
const runForm = document.getElementById("runForm");
const runSubmit = document.getElementById("runSubmit");
const AVAILABLE = new Set([0, 1]);
let cur = 0;


function openRun(k) {
    const g = groups[k], ok = AVAILABLE.has(k);
    document.getElementById("runTitle").textContent = "Inferenza: " + g.titolo;
    document.getElementById("runSub").textContent = g.dims.map(d => d[0]).join(" · ");
    document.getElementById("runGroup").value = g.titolo;
    
    const sets = {0: "runFields", 1: "runFieldsStruct"};
    Object.entries(sets).forEach(([idx, id]) => {
        const box = document.getElementById(id);
        const on = ok && Number(idx) === k;
        box.hidden = !on;
        box.querySelectorAll("input, select").forEach(i => i.disabled = !on);
    });

    document.getElementById("runNA").hidden = ok;
    runSubmit.hidden = !ok;
    
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
    runSubmit.disabled = true;
    runSubmit.textContent = "Analisi in corso";
});

window.addEventListener("pageshow", () => {
    runSubmit.disabled = false;
    runSubmit.textContent = "Avvia analisi";
});

// Generazione pagine
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

let startX = null, dx = 0;

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
    const w = book.offsetWidth;
    
    // Gestione della rotazione interattiva
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
    
    // Rilascia esplicitamente il pointer capture se l'evento lo supporta
    if(e && e.pointerId) {
        try { book.releasePointerCapture(e.pointerId); } catch(err) {}
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

// Primo avvio
render();
