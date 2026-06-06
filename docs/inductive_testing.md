# Inductive Testing & Lateral Movement Validation

Questo documento descrive il rilevamento dei **movimenti laterali** tramite il
Temporal Graph Network (TGN), l'*induttività*, e — soprattutto — la
**de-circolarizzazione della valutazione** che rende i numeri qui sotto onesti e
difendibili. Sostituisce una versione precedente che riportava metriche gonfiate da
una fuga di informazione nel negative sampling.

## Il problema: valutazione circolare (corretto)

La versione precedente del training costruiva l'*hard negative* strutturale come
«src accoppiato a una risorsa **autorizzata ma non abituale**» (usando `auth_mask` e
il subset abituale del generatore), **pesato ×10** nella loss. Questa è *esattamente*
la definizione con cui il generatore sintetico crea le anomalie di **lateral movement**
di test (`etype=3`). Il modello veniva quindi allenato sulla definizione precisa
dell'anomalia che poi doveva «scoprire»: la recall riportata (~40%) misurava la
memorizzazione del curriculum, non una generalizzazione.

Analogamente, il negativo *contestuale* invertiva il 20% dei bit del messaggio — lo
stesso meccanismo con cui il generatore crea le anomalie contestuali (randomizzazione
0/1 di JA3/Snort/sonde).

**Fix (`src/train_tgn.py`):**
- Il negativo strutturale è ora una **destinazione pescata uniformemente sulle
  risorse** (`_sample_structural_negatives`), *indipendente* da `auth_mask`/abitualità.
  Niente più peso ×10. È il setup self-supervised standard (benigno vs. corruzione
  generica): **non** è zero-shot — i negativi casuali sono perlopiù non-abituali, quindi
  il modello impara comunque la nozione generica di «accesso inusuale» — ma non
  rispecchia più la regola di test.
- Il negativo contestuale usa **rumore gaussiano additivo** (meccanismo diverso dalla
  randomizzazione 0/1 del test).
- La trust score non è più mutata dalle etichette ground-truth in training (era un
  segnale auto-avverante); resta una feature statica/runtime.

Una guardia esplicita (`_sample_structural_negatives` riceve solo il range di id delle
risorse, mai `auth_mask`) impedisce la reintroduzione della circolarità.

## Induttività (hashed identity)

Il TGN è induttivo: alloca dinamicamente memoria per entità mai viste e assegna loro
un'identità tramite **hashing deterministico** della chiave (`stable_hash`, BLAKE2b),
coerente tra processi e riavvii (il precedente `hash()` builtin era salato per-processo
e rompeva la riproducibilità). Per rilevare un'anomalia *strutturale*, però, serve uno
storico (Temporal Neighborhood): un'entità a freddo, senza storia, non ha «abitudini»
da cui deviare → vedi cold-start sotto.

## Risultati onesti (stream sintetico, FPR target 1%, split cronologico 70/10/20)

Tutti i modelli: solo benigno in training, soglia calibrata sul benigno di validazione,
metriche per-classe benigno-vs-quel-tipo sul segmento di test. (`num_events=50000`,
`seed=42`.)

| Modello | Agg AUC | Agg AP | policy AUC | contextual AUC | **lateral AUC** | lateral Rec@1%FPR |
|---|---|---|---|---|---|---|
| Rule baseline (solo segnali edge) | — | — | 0.000 rec | 0.966 rec | 0.000 rec | — |
| Isolation Forest (non-relazionale) | 0.684 | 0.392 | 0.812 | 0.733 | **0.517** | 1.6% |
| One-Class SVM (non-relazionale) | 0.769 | 0.660 | 0.887 | 0.936 | **0.501** | 2.4% |
| Static GNN (stesso curriculum, no temporale) | 0.854 | 0.796 | 0.997 | 0.996 | **0.587** | 1.0% |
| **TGN (full)** | **0.895** | **0.839** | 0.990 | 0.995 | **0.711** | **16.3%** |

### Lettura dei risultati
- **Policy & contextual** sono risolte dalla sola struttura relazionale (TGN e Static
  GNN ~0.99 AUC); i detector non-relazionali (IF, OC-SVM) sono nettamente più deboli.
  Le contestuali sono inoltre **banali**: la rule baseline (solo segnali edge) le prende
  al 96.6% → il TGN qui non aggiunge nulla di sostanziale.
- **Lateral movement** è il caso genuinamente difficile (autorizzato, signal-clean):
  l'unico indizio è lo storico di interazione. È **l'unica classe in cui la macchina
  temporale conta**: lateral AUC sale da ~0.50 (non-relazionale) a 0.59 (GNN statico,
  stesso curriculum) a **0.71 (TGN completo)**. Resta comunque difficile: recall solo
  ~16% alla soglia dell'1% di FPR. Non è «risolto» — è onestamente debole ma reale.

Il valore del modello, in sintesi: rilevare **policy violation** (cieche alle regole) in
modo forte e fornire l'**unico** segnale > caso sul **lateral movement**, restando 100%
unsupervised.

## Ablation (componenti TGN-specifiche)

Stesso stream/seed, `save=False` (non tocca gli artifact). La parte temporale è ablata
a parte dalla baseline *Static GNN* (lateral AUC 0.59). Qui si togliono le due componenti
«novelty»: la **testa strutturale** (coseno) e l'**hashed identity**.

| variante | agg AUC | policy AUC | ctx AUC | **lateral AUC** | lat Rec@1%FPR |
|---|---|---|---|---|---|
| full (struct + hash) | 0.904 | 0.992 | 0.993 | **0.738** | 14.8% |
| no struct head | 0.918 | 0.987 | 0.993 | **0.781** | 5.8% |
| no hashed identity | 0.831 | 0.995 | 0.994 | **0.522** | 5.0% |

Letture oneste (e in parte controintuitive):
- **L'hashed identity è la componente critica per il laterale**: rimuoverla fa crollare
  il lateral AUC a 0.52 (≈caso). Per giudicare «questa coppia src↔dst è inusuale» il
  modello ha bisogno di identità di nodo distinguibili (incluse le risorse). Effetto
  ampio e robusto.
- **La testa strutturale a coseno è marginale/ambigua**: toglierla NON peggiora il ranking
  (lateral AUC 0.78 vs 0.74, entro la varianza run-to-run da non-determinismo CUDA);
  cambia solo il punto di lavoro (recall@thr più alta con la testa). Questo **contraddice**
  la tesi di progetto secondo cui la `struct` head sarebbe il rilevatore dedicato del
  lateral: il lavoro lo fa soprattutto la feature head condizionata dall'hashed identity.
- Policy/contextual restano saturi in tutte le varianti (la struttura relazionale basta).

> **Caveat di rigore**: i numeri sono di un singolo seed e il training su GPU è
> non-deterministico (~±0.03 sul lateral AUC tra run). Le conclusioni grandi (hashed
> identity critica; struct head marginale) sono robuste rispetto a questa varianza, ma per
> un paper vanno consolidate con **più seed** (media ± dev. std.). Riproduzione:
> `docker compose --profile ablations up`.

## Cold start & anti-poisoning (gate)

La memoria e il vicinato vengono aggiornati **solo** per eventi giudicati benigni
(predict-then-update). Due limiti noti, *intrinseci* al gate auto-deciso:
- un attaccante «stealthy» scorato benigno avvelena la baseline;
- un evento benigno scorato anomalo non viene mai appreso (starvation).
La mitigazione è demandata all'orchestrator (OPA come vero decisore via `/infer`→`/update`,
più un *grace period* breve per i nuovi nodi). Vedi `docs/orchestrator_integration.md`.

## Riprodurre

```bash
docker compose --profile training-tgn up      # TGN: training + eval onesta + rule baseline
docker compose --profile baseline-iforest up  # Isolation Forest
docker compose --profile baseline-ocsvm up     # One-Class SVM
docker compose --profile baseline-gnn up        # Static GNN (ablation, stesso curriculum)
docker compose --profile verify-tgn up          # correttezza serving-path
```

## Limite principale (validità esterna)

La valutazione è **interamente sintetica**: il generatore definisce esso stesso cosa sia
un'anomalia. La de-circolarizzazione rende il confronto onesto *all'interno* di questo
mondo, ma non sostituisce una validazione su dati reali. Il passo successivo per
solidità da paper è un dataset reale/pubblico di accessi (es. LANL auth, DARPA OpTC,
CIC-IDS) rimappato come stream ZTA. Vedi sezione *Limitazioni* nel README.
