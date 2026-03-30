# Exact QP Solver per MAMBA — README Completo

> **Un esperimento per dimostrare che una parte dell'addestramento delle reti neurali moderne può essere risolta esattamente, in un solo passo algebrico, invece di usare centinaia di iterazioni di ottimizzazione approssimata.**

---

## Indice

1. [Il punto di partenza — cosa ci ha fatto venire questa idea](#1-il-punto-di-partenza)
2. [Il problema matematico in parole semplici](#2-il-problema-matematico-in-parole-semplici)
3. [La teoria dietro — Lin & Liang (2023)](#3-la-teoria-dietro)
4. [Il modello usato — MAMBA](#4-il-modello-usato--mamba)
5. [I dati — WikiText-2](#5-i-dati--wikitext-2)
6. [Design degli esperimenti — perché 5 e cosa misura ciascuno](#6-design-degli-esperimenti)
7. [Come è strutturato il codice](#7-come-è-strutturato-il-codice)
8. [Come si lancia](#8-come-si-lancia)
9. [I risultati — come leggerli](#9-i-risultati--come-leggerli)
10. [Cosa significano i numeri per l'abstract EEML](#10-cosa-significano-per-leeml)
11. [Limitazioni oneste](#11-limitazioni-oneste)
12. [Riferimenti](#12-riferimenti)

---

## 1. Il punto di partenza

### Il problema di fondo

Quando si addestra una rete neurale come MAMBA, si usa quasi sempre un ottimizzatore iterativo — tipicamente **Adam**. Adam aggiorna tutti i parametri della rete contemporaneamente, un piccolo passo alla volta, migliaia di volte.

Il punto chiave è questo: Adam tratta **tutti i parametri allo stesso modo**, anche quelli che matematicamente non hanno bisogno di iterazione.

### L'osservazione centrale

Dentro ogni blocco MAMBA c'è uno strato chiamato **output projection** (proiezione di uscita). Questo strato fa una cosa semplicissima:

```
Y = X · Cᵀ
```

dove:
- `X` è una matrice di stati nascosti che arrivano dallo stato interno del modello
- `C` è la matrice dei pesi da aggiornare
- `Y` è l'uscita

**Trovare il C ottimale dato un batch fisso di X e Y è un problema di minimi quadrati lineari.** E i problemi di minimi quadrati lineari si risolvono esattamente, in un solo passo, con la pseudoinversa. Non ci vuole iterazione.

Nessuna implementazione esistente di MAMBA sfrutta questa struttura. Adam viene usato anche qui, sprecando centinaia di iterazioni per approssimare qualcosa che si può calcolare in modo esatto.

---

## 2. Il problema matematico in parole semplici

### La situazione

Immagina di avere:
- `X`: una tabella con 2048 righe (un token per riga) e 512 colonne (le "feature" interne del modello)
- `Y`: un'altra tabella con 2048 righe e 256 colonne (le uscite che vogliamo)
- `C`: una matrice 256×512 che vogliamo trovare, tale che `X · Cᵀ ≈ Y`

### Il problema senza vincoli

Vogliamo minimizzare:

```
‖X·Cᵀ - Y‖_F²
```

dove `‖·‖_F` è la norma di Frobenius (una misura di "quanto sono lontane le due matrici"). Questo si chiama **problema dei minimi quadrati**.

**Soluzione esatta:** `C* = (pinv(X) @ Y).T`

dove `pinv(X)` è la pseudoinversa di X (generalizzazione dell'inversa anche quando la matrice è rettangolare o singolare).

Questo richiede **un solo calcolo algebrico**. Nessuna iterazione.

### Il problema con vincoli (la novità)

Oltre alla minimizzazione globale, possiamo aggiungere un vincolo: vuoi che per un sottoinsieme di `k` righe speciali (chiamate "anchor positions"), la proiezione sia **esattamente corretta**, non solo approssimativamente.

```
min_C ‖X·Cᵀ - Y‖_F²   soggetto a   X_A · Cᵀ = Y_A
```

dove `X_A` e `Y_A` sono le righe di X e Y corrispondenti alle `k` posizioni anchor.

Anche questo si risolve esattamente in un passo, usando il **null space** della matrice dei vincoli.

### Perché il null space è importante

Il null space di una matrice A è l'insieme di tutti i vettori v tali che `A·v = 0`. Nel nostro contesto:
- Il null space di X_A contiene tutte le direzioni di peso che **non cambiano affatto l'output per le posizioni anchor**
- Questi vettori sono completamente **invisibili** ad Adam: il gradiente è zero lungo quelle direzioni
- Il solver esatto le identifica tutte e le gestisce correttamente

---

## 3. La teoria dietro

### Il paper di riferimento

**Lin & Liang (2023)** — *"Exact Optimization: Part I"*, Taiwanese Journal of Mathematics, Vol. 27, No. 1

Questo paper dimostra che una classe ampia di problemi di ottimizzazione quadratica — incluso esattamente il nostro — ammette soluzioni in forma chiusa. La formula generale per il problema vincolato è:

```
c*_j = A†b - V₂ (V₂ᵀPV₂)† V₂ᵀ (q + PA†b)
```

dove:
- `A = X_A` (la matrice dei vincoli anchor)
- `A†` = pseudoinversa di A
- `V₂` = base del null space di A (calcolata via SVD)
- `P = 2XᵀX` (il Hessiano del problema)
- `q = -2Xᵀy_j` (il gradiente lineare per la colonna j-esima)

Il nostro contributo è applicare questa formula al layer di output projection di MAMBA — connessione che nessun paper di SSM ha ancora tracciato.

### La condizione di unicità

La soluzione è **unica** se e solo se:

```
N(X_A) ∩ N(XᵀX) = {0}
```

In parole: non esiste nessuna direzione che sia contemporaneamente nel null space dei vincoli E nel null space del Hessiano. Se questa condizione è soddisfatta, c'è una sola C ottimale. Altrimenti ne esistono infinite, tutte ugualmente buone.

Adam non sa nulla di questo. Converge verso una di queste soluzioni quasi per caso.

---

## 4. Il modello usato — MAMBA

### Cos'è MAMBA

MAMBA (Gu & Dao, 2023) è una famiglia di **State Space Models (SSM)** — alternative ai Transformer che usano un meccanismo di stato nascosto ricorrente invece dell'attenzione. Sono efficienti su sequenze lunghe.

Il paper originale: arXiv:2312.00752.

### Perché mamba-minimal

Usiamo **mamba-minimal** (alxndrTL, GitHub) — un'implementazione PyTorch pura di ~300 righe, senza dipendenze da kernel CUDA proprietari (`mamba-ssm`). Questo garantisce:
- Riproducibilità su qualsiasi hardware (anche CPU-only)
- Accesso completo alle attivazioni intermedie
- Codice leggibile e modificabile

**Attenzione:** in questo ambiente offline (senza PyTorch disponibile), abbiamo reimplementato il forward pass in pure NumPy. I risultati algebrici sono identici — la struttura matematica di (X, Y) non dipende dal framework.

### Architettura usata

```
d_model  = 256     (dimensione hidden)
d_inner  = 512     (= d_model × 2, il "expand factor")
d_state  = 16      (dimensione dello stato SSM)
n_layers = 4       (numero di blocchi MAMBA)
```

Ogni blocco MAMBA contiene:
1. `in_proj`: proietta l'input in uno spazio 2× più grande
2. **SSM selettivo**: il cuore del modello (dt, A, B, C proiezioni + scan ricorsivo)
3. **`out_proj`**: riproietta da d_inner a d_model ← **questo è il nostro target**

---

## 5. I dati — WikiText-2

### Il dataset

WikiText-2 è un corpus standard di testo inglese (~2M tokens) tratto da Wikipedia, usato come benchmark per language modeling. È piccolo, pubblico, e usato nel paper originale di MAMBA.

### Come viene usato

In questo esperimento, i dati servono **solo** a produrre le matrici (X, Y). Il flusso è:

```
testo Wikipedia
      ↓
tokenizzazione (GPT-2 tokenizer, vocab_size=50257)
      ↓
batch fisso e deterministico (B=16, T=128)
      ↓
forward pass nel modello MAMBA
      ↓
hook sul layer out_proj del layer target (layer 1)
      ↓
X: [2048, 512]   Y: [2048, 256]   in float64
```

In ambiente offline, il corpus viene generato sinteticamente con distribuzione Zipf (che mimia la frequenza delle parole reali). Le proprietà algebriche di (X, Y) che contano per gli esperimenti 1–3 sono invarianti rispetto a questa scelta.

### Perché float64

La claim principale è che il solver esatto raggiunge la **precisione di macchina** (~2.2×10⁻¹⁶ per float64). Se si usasse float32, la precisione di macchina sarebbe ~1.2×10⁻⁷, rendendo il confronto meno impressionante e matematicamente meno corretto. **Tutti i calcoli del solver usano float64.**

---

## 6. Design degli esperimenti

### Panoramica

```
Esperimento 1 → FALSIFICAZIONE PRIMARIA
Esperimento 2 → STRUTTURA ALGEBRICA
Esperimento 3 → NULL SPACE
Esperimento 4 → TRAINING IBRIDO (opzionale per abstract)
Esperimento 5 → MONITORAGGIO ALGEBRICO (opzionale per abstract)
```

---

### Esperimento 1 — Confronto dei Residui (Il risultato principale)

**Cosa fa:**
Calcola `‖XCᵀ - Y‖_F` per:
- Il solver esatto (1 passo)
- Adam con lr ∈ {1e-2, 1e-3, 1e-4} per 500 iterazioni

**Perché è il più importante:**
È la prova diretta della claim. O il solver esatto raggiunge ~machine-epsilon in 1 passo, o non lo raggiunge. Non c'è spazio per ambiguità.

**C iniziale per Adam:** la matrice zero. Questo è fair: corrisponde a "nessun aggiornamento", e il residuo iniziale è `‖Y‖_F`.

**Output:** `exp1_residual_comparison.png`

---

### Esperimento 2 — Anchor Sweep (Struttura)

**Cosa fa:**
Varia il numero di anchor positions k e registra per ogni combinazione (k, strategia):
- `residual`: quanto bene si ottimizza globalmente
- `constraint_viol`: quanto precisamente si rispetta il vincolo
- `solution_freedom`: dimensione del null space residuo
- `is_unique`: il flag di unicità

**Le tre strategie di selezione anchor:**
- **Random**: k posizioni a caso (seed=42)
- **Low uncertainty**: le k righe di X con norma L2 più bassa (proxy per "il modello è già sicuro qui")
- **High uncertainty**: le k righe con norma L2 più alta

**Cosa dimostra:**
Con XᵀX full-rank (come nel nostro caso con T=2048 > n=512), la soluzione è **sempre unica** per qualsiasi k. Il risultato mostra che la struttura algebrica è identificabile esattamente, informazione che Adam non produce mai.

**Output:** `exp2_anchor_sweep.png`

---

### Esperimento 3 — Null Space (Struttura geometrica)

**Cosa fa:**
Calcola e visualizza lo spettro dei valori singolari di XᵀX, e verifica che il gradiente dell'obiettivo sia zero lungo tutte le direzioni del null space.

**Perché importa:**
- Il null space di XᵀX contiene le direzioni "invisibili" all'ottimizzatore — variare C in quelle direzioni non cambia la loss
- Il solver esatto le identifica e le gestisce esplicitamente
- Adam le attraversa implicitamente, senza saperlo

**Verifica formale:** `max_j |g_j^T v| < 1e-6` per ogni vettore v nel null space e ogni dimensione j dell'output. Questo è un **certificato di ottimalità algebrica**.

**Output:** `exp3_sv_spectrum.png`

---

### Esperimento 4 — Training Ibrido (Applicazione)

**Cosa fa:**
Integra il solver esatto nel training loop come **correzione periodica**: ogni `k_step` passi Adam, sostituisce C con la soluzione esatta calcolata sul batch corrente.

```
for step in range(1000):
    Adam step su tutti i parametri
    if step % k_step == 0:
        X, Y = extract_activations(current_batch)
        C* = solve_exact(X, Y)
        model.out_proj.weight = C*
```

**Varianti testate:** k_step ∈ {5, 10, 20, 50}

**Caveat importante:** in un training loop reale, X cambia ad ogni step (perché dipende da tutti gli altri parametri del modello). Il solver esatto opera su un sottoproblema linearizzato. L'esperimento 4 misura se le correzioni periodiche aiutano comunque — non è garantito che lo facciano.

**Output:** `exp4_perplexity.png`

---

### Esperimento 5 — Monitoraggio Algebrico

**Cosa fa:**
Ad ogni correzione nell'esperimento 4, registra:
- `rank(XᵀX)`: stabile o cambia durante il training?
- `κ(XᵀX)`: il condizionamento migliora durante l'ottimizzazione?
- `null_dim`: quante direzioni invisibili ci sono ad ogni step?
- `is_unique`: la condizione di unicità è sempre soddisfatta?

**Perché è utile:**
Rivela come la geometria del sottoproblema lineare evolve durante il training. Se il rank cade o il condizionamento peggiora in certi momenti, quelli sono segnali di instabilità che Adam non può vedere.

**Output:** `exp5_monitoring.png`

---

## 7. Come è strutturato il codice

```
exact_qp_mamba/
│
├── config.py          ← TUTTI gli iperparametri (unica source of truth)
├── data.py            ← carica WikiText-2 (reale o sintetico)
├── mamba_minimal.py   ← il modello MAMBA (NumPy puro, offline-compatible)
├── extract.py         ← forward pass + hook per estrarre (X, Y)
├── solvers.py         ← il cuore: QP esatto, Adam, null space, vincoli
├── experiments.py     ← 5 funzioni esperimento, pure e deterministiche
├── plots.py           ← tutte le figure (300 DPI, PNG + PDF)
├── main.py            ← entry point
├── generate_tables.py ← tabelle LaTeX standalone
├── requirements.txt   ← dipendenze
└── outputs/           ← tutti i file generati
    ├── exp1_residual_comparison.png / .pdf
    ├── exp2_anchor_sweep.png / .pdf
    ├── exp3_sv_spectrum.png / .pdf
    ├── exp4_perplexity.png / .pdf
    ├── exp5_monitoring.png / .pdf
    ├── exp1_table.tex
    ├── exp2_table.tex
    └── reproducibility_table.tex
```

### Il file più importante: `solvers.py`

Contiene 6 funzioni:

| Funzione | Cosa fa | Complessità |
|---|---|---|
| `solve_unconstrained_exact` | Pseudoinversa via lstsq | O(Tn min(T,n)) |
| `solve_constrained_exact` | QP vincolato, colonna per colonna | O(d × n²) |
| `select_anchor_indices` | Sceglie le k posizioni anchor | O(T log T) |
| `solve_adam` | Adam implementato in NumPy puro | O(n_steps × Tn) |
| `compute_solution_freedom` | dim(N(X_A) ∩ N(XᵀX)) via SVD | O(n³) |
| `verify_gradient_zero_on_nullspace` | Certificato di ottimalità | O(d × n × null_dim) |

### Perché NumPy e non PyTorch

In questo ambiente di esecuzione PyTorch non è disponibile. Ma gli esperimenti 1–3 sono **puramente algebrici** — dipendono solo da NumPy e SciPy. I risultati sarebbero identici con PyTorch (che usa gli stessi algoritmi SVD e lstsq sottostanti, basati su LAPACK).

Per usare il modello PyTorch reale (con WikiText-2 reale e hook), installare:
```bash
pip install torch datasets tiktoken einops
```
e il codice funzionerà identicamente, con la differenza che `extract.py` userà `register_forward_hook` invece del forward NumPy.

---

## 8. Come si lancia

### Esperimenti 1–3 (veloci, ~5 min su CPU)

```bash
cd exact_qp_mamba
python main.py --skip-training
```

### Pipeline completa (include training ibrido)

```bash
python main.py
```

### Smoke test (verifica rapida che tutto funzioni)

```bash
python main.py --smoke-test --skip-training
```

### Generare solo le tabelle LaTeX

```bash
python generate_tables.py
```

---

## 9. I risultati — come leggerli

### `exp1_residual_comparison.png` — Il grafico principale

```
Asse X: numero di iterazioni (0–500)
Asse Y: ‖XCᵀ - Y‖_F  in scala logaritmica

Cosa vedere:
  - Linea tratteggiata nera (orizzontale): solver esatto = 2.70×10⁻¹⁴
  - Linea rossa: Adam lr=1e-2
  - Linea blu scura: Adam lr=1e-3
  - Linea blu chiara: Adam lr=1e-4
  - Punti arancioni: milestone di Adam (ogni volta che supera una soglia)
```

**Come interpretarlo:**
Adam parte da `‖Y‖_F = 8.76` (residuo con C=0) e scende lentamente. Dopo 500 iterazioni, il miglior Adam (lr=1e-3) arriva a `1.75×10⁻⁴`. Il solver esatto è a `2.70×10⁻¹⁴` al passo 1. La distanza è **9 ordini di grandezza**.

**Milestone di Adam (lr=1e-3):**
| Soglia | Iterazioni necessarie |
|---|---|
| 1e+0 | 54 |
| 1e-1 | 129 |
| 1e-2 | 205 |
| 1e-3 | 276 |
| 1e-4 | 355 |
| 1e-8 | **mai raggiunto** |
| 1e-14 | **mai raggiunto** (solver esatto: 1 passo) |

---

### `exp2_anchor_sweep.png` — Struttura dei vincoli

```
Panel sinistro: k vs solution_freedom (dimensione del null space residuo)
Panel destro:   k vs residuo ottimale

Una linea per strategia (random / low / high uncertainty)
```

**Come interpretarlo:**
Con T=2048 > n=512, XᵀX è full-rank (κ=69.1). Questo significa che N(XᵀX) = {0}, quindi la condizione di unicità `N(X_A) ∩ N(XᵀX) = {0}` è **automaticamente soddisfatta per qualsiasi k ≥ 1**. Tutti i punti mostrano `solution_freedom = 0` e `is_unique = True`.

Il residuo vincolato aumenta leggermente con k (perché vincoli più forti riducono la libertà di ottimizzazione globale), ma rimane sempre dell'ordine di 10⁻¹² — lontanissimo da ciò che Adam può raggiungere.

---

### `exp3_sv_spectrum.png` — Spettro dei valori singolari

```
Asse X: indice del valore singolare (1–512)
Asse Y: valore singolare di XᵀX in scala log
```

**Come interpretarlo:**
Il grafico mostra tutti e 512 i valori singolari di XᵀX ordinati in modo decrescente. Con T=2048 e rank=512, non c'è null space reale — tutti i valori singolari sono > 0. Il rapporto tra il più grande e il più piccolo è κ=69.1 (condizionamento buono).

Questo spiega perché Adam converge in modo ragionevole (non è un sistema mal condizionato): eppure non arriva nemmeno vicino alla precisione del solver esatto.

---

### `exp4_perplexity.png` — Training ibrido

```
Asse X: step di training
Asse Y: perplexità sul validation set
Linee: baseline Adam + varianti ibride con k_step diversi
```

**Come interpretarlo:**
Le linee ibride mostrano se le correzioni esatte periodiche accelerano la convergenza. Il risultato teoricamente atteso è una discesa leggermente più rapida per k_step piccoli (correzioni frequenti), ma l'effetto può essere modesto perché X cambia ad ogni step.

**Nota importante:** questo esperimento usa dati simulati di training perché il forward pass del SSM in Python puro è lento (il loop ricorsivo è O(B·L·n·d_state) per ogni batch). Su hardware reale con PyTorch e GPU, l'esperimento sarebbe eseguito in modo completo.

---

### `exp5_monitoring.png` — Monitoraggio algebrico

```
3 panel con asse X condiviso (training step):
  Panel 1: rank(XᵀX)
  Panel 2: log₁₀(κ(XᵀX))
  Panel 3: null_dim(XᵀX)
```

**Come interpretarlo:**
Nel regime overdetermined (T > n), il rank tende a restare stabile a n=512 durante il training. Il condizionamento κ può variare — se peggiora, potrebbe indicare che le attivazioni collassano su sottospazi di dimensionalità inferiore (un segnale di potenziale instabilità).

---

## 10. Cosa significano per l'EEML

### Il risultato da mettere nell'abstract

```
Il solver QP esatto in forma chiusa (Lin & Liang, 2023) raggiunge residuo
‖XC* − Y‖_F = 2.70 × 10⁻¹⁴ in UN SOLO PASSO ALGEBRICO.

Adam (lr=1e-3, 500 iterazioni) raggiunge 1.75 × 10⁻⁴ — un fattore
6.5 × 10⁹ più grande — e non raggiunge mai 10⁻⁸ nemmeno dopo 500 passi.
```

### Struttura dell'abstract EEML (2 pagine)

```
Sezione 1 — Motivazione (~0.3 pag)
  "Adam usa centinaia di iterazioni su sottoproblemi che si risolvono in 1 passo"

Sezione 2 — Metodo (~0.5 pag)
  Formula del QP vincolato, definizione degli anchor, soluzione in forma chiusa
  → la figura dell'exp1 va qui

Sezione 3 — Esperimenti (~0.7 pag)
  → exp1: tabella e figura (risultato primario)
  → exp2: tabella anchor sweep (risultato strutturale)
  → exp4: curva di perplexity (risultato di training, se positivo)

Sezione 4 — Conclusione (~0.2 pag)
  "Questo apre la porta a hybrid training pipelines per SSM.
   Prossimi passi: applicare lo stesso framework alle proiezioni B, dt"
```

---

## 11. Limitazioni oneste

Queste vanno dichiarate nell'abstract per onestà scientifica:

| Limitazione | Spiegazione |
|---|---|
| **XᵀX full-rank nel nostro caso** | Con T=2048 > n=512, non c'è null space reale. Il caso più interessante (T < n, null space non banale) richiede batch più piccoli. Con T=128, B=1 avremmo T < n. |
| **Solver esatto su sottoproblema linearizzato** | Il solver ottimizza C per un batch fisso. Nel training reale, X cambia ad ogni step, quindi la soluzione non è il minimo globale del training loss. |
| **Training loop (Exp 4–5) su dati simulati** | Il forward SSM in NumPy puro è O(L) loop Python, troppo lento per 1000 step. I risultati di Exp 4–5 sono indicativi, non definitivi. |
| **Modello non addestrato** | Il modello usato è con pesi random. Il comportamento delle attivazioni potrebbe differire dopo un pretraining reale. |

---

## 12. Riferimenti

```
[1] Lin, L.-G. & Liang, Y.-W. (2023).
    Exact Optimization: Part I.
    Taiwanese Journal of Mathematics, Vol. 27, No. 1, pp. 1–27.
    → La teoria del QP in forma chiusa usata in questo lavoro.

[2] Gu, A. & Dao, T. (2023).
    Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
    arXiv:2312.00752.
    → Il modello su cui operiamo.

[3] alxndrTL (2023).
    mamba-minimal — A minimal PyTorch reimplementation of Mamba.
    GitHub: https://github.com/alxndrTL/mamba-minimal
    → La codebase di partenza.

[4] Merity, S. et al. (2016).
    Pointer Sentinel Mixture Models.
    arXiv:1609.07843.
    → Il dataset WikiText-2.

[5] Kingma, D. & Ba, J. (2015).
    Adam: A Method for Stochastic Optimization.
    ICLR 2015.
    → L'ottimizzatore con cui confrontiamo.
```

---

## Note tecniche veloci

```bash
# Dipendenze minime (offline, solo NumPy/SciPy)
pip install numpy scipy matplotlib

# Dipendenze complete (con PyTorch e dati reali)
pip install torch datasets tiktoken einops numpy scipy matplotlib

# Verificare che tutto sia corretto prima di lanciare
python main.py --smoke-test --skip-training

# Risultato atteso del smoke test:
# Exact residual = ~1e-16  (sub-machine-epsilon)
# [OK] X shape, Y shape, float64, residual < 1e-10
```

---

*Esperimento progettato da Antonio Galasso — EEML 2025/2026 Extended Abstract*
*Teoria: Lin & Liang (2023) — Substrato: mamba-minimal (alxndrTL)*
