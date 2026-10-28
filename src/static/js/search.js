(function () {
    const input = document.getElementById("candidate-search-input");
    const clear = document.getElementById("candidate-search-clear");
    const count = document.getElementById("candidate-search-count");
    const list = document.querySelector(".candidates-list");

    if (!input || !clear || !count || !list) return;

    const cards = Array.from(list.querySelectorAll(".candidate-card"));
    const total = cards.length;

    function normalize(value) {
        return String(value || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLocaleLowerCase("it").replace(/\s+/g, " ").trim();
    }

    function filterCandidates() {
        const query = normalize(input.value);
        let visible = 0;

        cards.forEach(card => {
            const text = normalize(card.textContent);
            const matches = !query || text.includes(query);

            card.hidden = !matches;
            card.classList.toggle("search-match", Boolean(query) && matches);
            if (matches) visible++;
        });

        if (!query) {
            count.textContent = `${total} coppie candidate`;
        } else if (visible == 0) {
            count.textContent = `Nessuna corrispondenza su ${total} coppie`;
        } else {
            count.textContent = `${visible} di ${total} coppie`;
        }
        clear.hidden = input.value.length === 0;
    }

    input.addEventListener("input", filterCandidates);

    clear.addEventListener("click", function () {
        input.value = "";
        filterCandidates();
        input.focus();
    });

    let updatePending = false;
    function scheduleFilter() {
        if (updatePending) return;

        updatePending = true;
        requestAnimationFrame(() => {
            updatePending = false;
            filterCandidates();
        });
    }
    const observer = new MutationObserver(scheduleFilter);

    cards.forEach(card => {
        card.querySelectorAll(".rel-shot, .explanation-box").forEach(element => {
            observer.observe(element, {
                childList: true,
                characterData: true,
                subtree: true,
            });
        });
    });

    filterCandidates();
})();