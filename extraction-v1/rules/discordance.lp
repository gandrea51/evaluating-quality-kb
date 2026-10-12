% Proprietà considerate funzionali

functional(founding_year).
functional(headquarters).
functional(website).
functional(swift_code).
functional(ifsc_example).
functional(customer_support_email).
functional(customer_support_phone).
functional(digit_length).
functional(example).
functional(bank_type).


% Gruppi da verificare

functional_group(P, E) :-
    functional(P),
    fact(P, E, _, _, _).


% Coppie di valori incompatibili

% Valori differenti: si ordina V1 e V2 per evitare di generare due volte la stessa coppia.

conflict(P, E, V1, U1, V2, U2) :-
    functional_group(P, E),
    fact(P, E, V1, U1, _),
    fact(P, E, V2, U2, _),
    V1 < V2.

% Stesso valore ma unità differenti

conflict(P, E, V, U1, V, U2) :-
    functional_group(P, E),
    fact(P, E, V, U1, _),
    fact(P, E, V, U2, _),
    U1 < U2.


% Gruppi discordanti e valori in competizione

conflicting_group(P, E) :-
    conflict(P, E, _, _, _, _).

conflict_value(P, E, V, U, S) :-
    conflicting_group(P, E),
    fact(P, E, V, U, S).


% Discordanze tra documenti distinti

cross_doc_conflict(P, E) :-
    conflict(P, E, V1, U1, V2, U2),
    fact(P, E, V1, U1, S1),
    fact(P, E, V2, U2, S2),
    chunk_doc(S1, D1),
    chunk_doc(S2, D2),
    D1 != D2.


% 6. Statistiche

n_functional_groups(N) :-
    N = #count {
        P, E : functional_group(P, E)
    }.

n_conflicting_groups(N) :-
    N = #count {
        P, E : conflicting_group(P, E)
    }.

n_cross_doc_conflicts(N) :-
    N = #count {
        P, E : cross_doc_conflict(P, E)
    }.

#show conflict/6.
#show conflict_value/5.
#show conflicting_group/2.
#show cross_doc_conflict/2.
#show n_functional_groups/1.
#show n_conflicting_groups/1.
#show n_cross_doc_conflicts/1.