(function () {
    const fab = document.getElementById("chatFab");
    const panel = document.getElementById("chatPanel");
    if (!fab || !panel) return;
 
    const log = document.getElementById("chatLog");
    const form = document.getElementById("chatForm");
    const input = document.getElementById("chatInput");
    const send = document.getElementById("chatSend");
    const closeBtn = document.getElementById("chatClose");
    const hint = document.getElementById("chatHint");
 
    const TIMEOUT_MS = 130000;

    function toggle(open) {
        const show = typeof open === "boolean" ? open : !panel.classList.contains("open");
        panel.classList.toggle("open", show);
        if (show) input.focus();
    }

    function addMessage(kind, text) {
        if (hint) hint.remove();

        const el = document.createElement("div");
        el.classList = "chat-msg " + kind;
        el.textContent = text;
        log.appendChild(el);
        log.scrollTop = log.scrollHeight;
        return el;
    }

    fab.addEventListener("click", () => toggle());
    closeBtn.addEventListener("click", () => toggle(false));

    panel.addEventListener("keydown", e => {
        e.stopPropagation();
        if (e.key === "Escape") toggle(false);
    });

    form.addEventListener("submit", async e => {
        e.preventDefault();

        const question = input.value.trim();
        if (!question) return;

        addMessage("user", question);
        input.value = "";

        const pending = addMessage("pending", "Sto pensando...");
        input.disabled = true;
        send.disabled = true;

        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

        try {
            const response = await fetch("/ask", {
                method: "POST", 
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({question}),
                signal: controller.signal
            });
            const data = await response.json().catch(() => ({}));

            pending.remove();
            if (!response.ok) {
                addMessage("error", data.error || `Errore HTTP ${response.status}`);
            } else {
                addMessage("bot", data.answer || "(risposta vuota");
            }
        } catch (error) {
            pending.remove();
            addMessage("error", error.name === "AbortError" ? "Tempo scaduto: il modello non ha risposto." : "Impossibile contattare il server.");
        } finally {
            clearTimeout(timer);
            input.disabled = false;
            send.disabled = false;
            input.focus();
        }
    });

}) ();