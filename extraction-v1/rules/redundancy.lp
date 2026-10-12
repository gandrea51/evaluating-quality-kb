% Fatti distinti
distinct_fact(P, E, V, U) :- fact(P, E, V, U, _).

% Chunk che supportano i fatti
support(P, E, V, U, N) :-
    distinct_fact(P, E, V, U),
    N = #count {
        S : fact(P, E, V, U, S)
    }.


% Fatto presente in più chunk anche se appartengono allo stesso doc
redundant_fact(P, E, V, U, N) :-
    support(P, E, V, U, N),
    N > 1.

% Associazione fatto - documento
doc_of_fact(P, E, V, U, D) :-
    fact(P, E, V, U, S),
    chunk_doc(S, D).

% Documenti distinti che supportano il fatto
doc_support(P, E, V, U, N) :-
    distinct_fact(P, E, V, U),
    N = #count {
        D : doc_of_fact(P, E, V, U, D)
    }.


% Stesso fatto presente in più doc distinti
cross_doc_redundant(P, E, V, U, N) :-
    doc_support(P, E, V, U, N),
    N > 1.


% Statistiche

% num. occorrenze
n_occurrences(N) :-
    N = #count {
        P, E, V, U, S : fact(P, E, V, U, S)
    }.


% num. combinazioni distinte di proprietà, entità, valore e unità
n_distinct(N) :-
    N = #count {
        P, E, V, U : distinct_fact(P, E, V, U)
    }.


% num. fatti distinti ripetuti
n_redundant_facts(N) :-
    N = #count {
        P, E, V, U : redundant_fact(P, E, V, U, _)
    }.


% num. fatti distinti ripetuti tra documenti
n_cross_doc_redundant(N) :-
    N = #count {
        P, E, V, U : cross_doc_redundant(P, E, V, U, _)
    }.


#show redundant_fact/5.
#show cross_doc_redundant/5.
#show n_occurrences/1.
#show n_distinct/1.
#show n_redundant_facts/1.
#show n_cross_doc_redundant/1.

