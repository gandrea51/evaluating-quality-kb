(function () {
    const input = document.getElementById("candidate-search-input");
    const clear = document.getElementById("candidate-search-clear");
    const count = document.getElementById("candidate-search-count");
    const list = document.querySelector(".candidates-list");

    if (!input || !clear || !count || !list) return;

    const cards = Array.from(list.querySelectorAll(".candidate-card"));
    const total = cards.length;

    function normalize(value) {
        return (value || "")
            .normalize("NFD")
            .replace(/[\u0300-\u036f]/g, "")
            .toLocaleLowerCase("it")
            .trim();
    }

    function filterCandidates() {
        const query = normalize(input.value);
        let visible = 0;

        cards.forEach(card => {
            // Legge anche giudizio e spiegazione aggiornati dal polling.
            const matches = !query || normalize(card.innerText).includes(query);

            card.hidden = !matches;

            if (matches) visible++;
        });

        count.textContent = query
            ? `${visible} di ${total} coppie`
            : `${total} coppie candidate`;

        clear.hidden = !input.value;
    }

    input.addEventListener("input", filterCandidates);

    clear.addEventListener("click", function () {
        input.value = "";
        filterCandidates();
        input.focus();
    });

    // Imposta il conteggio iniziale.
    filterCandidates();
})();