# Calibrazione per arco e collasso della testa strutturale

Stato al 2026-09-27, branch `fix/optimize`. Tutte le run sono in docker su GPU, con
**seed 42**, 200k eventi, 15 epoche ed `eval_batch_size=1`. I seed pre-registrati
1000–1009 non sono stati toccati, quindi nessuno di questi numeri può entrare nel paper
come risultato finale. Il contesto sui lateral è in
[`lateral_movement_findings.md`](lateral_movement_findings.md).

---

## 1. Verifica della calibrazione per arco

Config: `edge_calibration=True`, `edge_calib_tail_q=0.99`. Log: `tasks/runs/edgecal_e15.log`
(le loss di training sono identiche a `opt_train_branch.log`, quindi cambia solo lo scoring).

| classe | AP (senza → con) | recall@soglia (senza → con) |
|---|---|---|
| lateral (n=169) | 0.014 → 0.036 | 0.071 → 0.166 (a FPR 1% globale: 0.012 → 0.148) |
| cred-theft | 0.012 → 0.020 | 0.080 → 0.084 |
| contextual | 0.376 → 0.246 | 0.661 → 0.484 |
| benign-denied | 0.621 → 0.462 | 0.722 → 0.439 |
| policy | 0.406 → 0.325 | 0.733 → 0.667 |
| exfil | 1.0 → 1.0 | 1.0 → 1.0 |

- AP aggregata: 0.488 → 0.426. Precision/recall dopo il routing: 0.53/0.49 → 0.54/0.35.
- FPR sui benigni: 0.0124 → 0.0075. Lateral su dispositivi condivisi: 0.146 → 0.313.
- I numeri confermano la stima offline. Nel handoff avevo definito "lieve" il calo di
  contextual e benign-denied: non lo è, e la stima offline lo mostrava già
  (benign-denied AP 0.62 → 0.49).
- Cambia anche il significato dello score: ora è `1 - p` rispetto alla coda benigna,
  calcolato in float64. La media dei benigni in validazione è circa 0.90 e le soglie
  sono circa 0.9999.

**Raccomandazione:** tenere la calibrazione. È l'unico intervento che ha spostato davvero
lateral e cred-theft, anche se in termini assoluti restano bassi, e il prezzo si paga su
classi che il modello già vedeva.

**Da decidere:** tenerla o toglierla, e se e quando rigenerare `public/` (cambiano i numeri
di LANL, PicoDomain, UWF e delle ablation).

## 2. Cosa fa la testa strutturale

Lo score di un arco è la somma di due parti:

```
score = feat + struct
struct = struct_scale * cos( norm(struct_proj(z_src)), norm(struct_proj(z_dst)) )
struct_proj = Linear(mem, 2·mem) → ReLU → Dropout(0.1) → Linear(2·mem, mem)
struct_scale: inizializzata a 5.0, ottimizzatore AdamW con weight decay di default
```

- `feat` è un MLP (`LinkPredictor`) sulle caratteristiche dell'evento.
- `struct` proietta i due nodi (per esempio utente e risorsa) e misura quanto puntano
  nella stessa direzione.

L'idea è che due nodi che interagiscono abitualmente puntino nella stessa direzione e
due estranei no. Un movimento laterale verso una macchina mai vista dovrebbe quindi
avere coseno basso.

## 3. Cosa ha misurato il probe

Script: `tasks/tmp/diag_edgecal/struct_probe.py`. Output: `tasks/runs/struct_probe.log` e
`tasks/tmp/diag_edgecal/struct_probe.json`.

Il probe osserva l'addestramento senza modificarlo:
- avvolge `model.score` sull'istanza;
- calcola le statistiche sotto `no_grad` senza consumare RNG;
- ricalcola la proiezione senza il Dropout;
- registra i gradienti con degli hook.

La traiettoria di training è quindi la stessa di una run normale.

- **Tutti i nodi finiscono nella stessa direzione.** R è la lunghezza media del vettore
  risultante delle proiezioni normalizzate: vale 1 se coincidono tutte, circa 0 se sono
  sparse.
  - All'epoca 1 R è già circa 0.99; dall'epoca 2 in poi circa 0.999.
  - Il coseno di qualunque coppia, legittima o estranea, è circa 1.
  - La deviazione standard del coseno sulle coppie positive scende da 0.05–0.08
    a circa 0.002.
- **La testa smette di imparare.** Norma al quadrato del gradiente su `struct_proj`,
  sommata sull'epoca:

  | epoca | norma² |
  |---|---|
  | 1 | 1347 |
  | 2 | 1.0 |
  | dalla 11 | circa 0 |

  `struct_scale` passa da 4.97 a 4.58, cioè cala solo per il weight decay.
- **La parte feature continua a imparare** (norma² del gradiente tra 2e3 e 8e4), quindi
  il problema riguarda solo la testa strutturale.
- **L'InfoNCE non è saturata:** la probabilità softmax del positivo vale circa 0.80
  sull'accesso e 0.89–0.98 sui binding, quindi resterebbe gradiente da dare. È satura
  solo la BCE sui positivi (sigmoide 0.997–1.0).

## 4. Il meccanismo

1. Nella prima epoca tutte le loss (BCE sui positivi e InfoNCE) chiedono la stessa cosa:
   coseno alto per le coppie legittime.
2. La soluzione più rapida è mandare **tutti** i nodi nella stessa direzione. Così ogni
   coppia ha coseno 1, anche quelle estranee.
3. Il coseno ha il suo massimo in 1, e lì la derivata è zero. Arrivata in quel punto
   stazionario, la testa non riceve più gradiente da nessuna loss, qualunque peso le si
   dia, e resta bloccata.

Delle ipotesi iniziali regge (a), nella forma "punto stazionario". (b) il dropout e
(c) la temperatura non sono la causa.

Collassano anche gli embedding a monte: il coseno medio a coppie di `z` sale da 0.67
a 0.91 e poi a 0.96.

Nelle ultime epoche i coseni negativi si separano un po': l'AUC del solo coseno arriva
a 0.77–0.91. È solo un contorno e non prova che il modello rilevi qualcosa, perché il
contributo della testa allo score resta al massimo circa 0.5 logit, trascurabile.
Per questo oggi la testa strutturale non aiuta a vedere i lateral.

## 5. Fix proposti (non implementati, serve l'ok)

**A — BatchNorm senza parametri appresi (da provare per primo).**
- Si aggiunge `BatchNorm1d(memory_dim, affine=False)` all'uscita di `struct_proj`.
- Toglie la componente comune a tutti i nodi, quindi la scorciatoia "tutti nella stessa
  direzione" non esiste più. È il rimedio standard contro il collasso (stile SimCLR/BYOL).
- È una riga di codice. In inferenza usa le statistiche fisse (running stats), quindi lo
  scoring evento per evento e i test di parità non cambiano.
- Verifica in due passi:
  1. stesso probe (circa 15 minuti su GPU); criterio di successo: R < 0.9 e gradiente
     su `struct_proj` non nullo dopo l'epoca 2;
  2. solo se passa, la run di verifica completa, guardando per prime AP e recall a FPR
     fissato di lateral e cred-theft.

**B — Regolarizzatore di uniformità (se A non basta).** Il termine senza etichette di
Wang & Isola (2020), che spinge le proiezioni a distribuirsi sulla sfera.

**C — Rimuovere la testa (ultima spiaggia).** Oggi vale quasi niente. Prima serve
un'ablation "senza testa strutturale" con seed 42 per un confronto diretto: quella
esistente usa il seed 2000 e non ha la riga "full". Per riferimento, con il seed 2000
la variante senza testa ha lateral AP 0.011.

## 6. Cosa resta da fare

- [ ] Decidere sulla calibrazione per arco (raccomandazione: tenerla) e su quando
      rigenerare `public/`.
- [ ] Fix A: implementarlo, verificarlo con il probe e poi con la run completa. B e C
      solo se A non basta.
- [ ] Negativi più difficili senza etichette (pesati per popolarità o a 2 hop), rispettando
      la guardia di de-circolarizzazione. Obiettivo: avvicinarsi allo 0.55 della probe
      supervisionata.
- [ ] Controllo del protocollo di CyberGFM con la regola degli archi sconosciuti su LANL
      (serve `data/auth.txt.gz`).
- [ ] Ripetere le misure sui seed 1000–1009 prima di riportare qualunque numero nel paper.

## Script e log

| file | cosa contiene |
|---|---|
| `tasks/tmp/diag_edgecal/struct_probe.py` | probe per epoca della testa strutturale (R, coseni, gradienti) |
| `tasks/tmp/diag_edgecal/struct_probe.json` | output del probe, una riga per epoca |
| `tasks/tmp/diag_edgecal/diag_scores.py` | logit per arco, calibrati e combinati, sui set di valutazione |
| `tasks/runs/struct_probe.log` | log della run del probe |
| `tasks/runs/edgecal_e15.log` | run di verifica con calibrazione per arco |
| `tasks/runs/opt_train_branch.log` | run di riferimento senza calibrazione |

Fonte: [Wang & Isola, ICML 2020](https://arxiv.org/abs/2005.10242), allineamento e
uniformità sulla ipersfera.
