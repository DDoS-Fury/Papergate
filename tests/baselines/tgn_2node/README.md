# Baseline TGN Vanilla a 2 Nodi (User -> Resource)

## Perché esiste questo file
Questa baseline isola l'effetto della **decomposizione ZTA a 5 nodi** ($\text{Source} \rightarrow \text{Config} \rightarrow \text{Device} \rightarrow \text{User} \rightarrow \text{Resource}$) rispetto a un Temporal Graph Network (TGN) **standard a 2 nodi** ($\text{User} \rightarrow \text{Resource}$, come nella letteratura classica di Dynamic Graph Link Prediction: Rossi et al., Euler, Argus).

## Meccanismo
- Mantiene la stessa identica macchina temporale del nostro modello: memoria ricorrente (TGNMemory GRU), encoding temporale continuo, neighbor loader temporale e link predictor.
- L'accesso HTTP è modellato unicamente come arco temporale diretto tra l'identità dell'utente (`user`) e la risorsa richiesta (`dst`), omettendo le entità intermedie di rete, configurazione TLS/JA3 e dispositivo hardware.
- Addestrato con lo stesso identico curriculum di negative sampling (InfoNCE su $K=5$ negativi casuali + anchor BCE positivo + BCE contestuale).
- Valutato con replay temporale evento per evento e soglia calibrata al 99° percentile (1% FPR) del segmento di validazione benigno.

## Risultato atteso
- Rileva bene le anomalie contestuali (grazie alle feature di messaggio) e ha una discreta comprensione delle abitudini utente-risorsa.
- È **cieco al Credential Theft e al Lateral Movement da nuove postazioni/tool**: quando un attaccante ruba le credenziali di un utente ed effettua richieste lecite (policy-clean e signal-clean), il grafo a 2 nodi vede semplicemente una richiesta valida emessa da quell'utente, non avendo a disposizione il pattern di binding anomalo $\text{IP} \rightarrow \text{JA3} \rightarrow \text{Device} \rightarrow \text{User}$.
