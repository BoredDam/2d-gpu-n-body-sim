# 🌌 Simulazioni *n*-body su GPU tramite C e OpenCL 🔭

Questo progetto riguarda l'implementazione in C e OpenCL di **simulazioni di sistemi gravitazionali a $n$ corpi**.
Le simulazioni a $n$ corpi di sistemi gravitazionali, sono fondamentali nell'astrofisica e nella cosmologia, permettendo
di simulare l'evoluzione di sistemi di corpi celesti.


## Metodologie di simulazione e matematiche
In particolar modo si implementano le metodologie:
- naive (o bruteforce) $O(n^2)$;
- Barnes-Hut $O(n \log n)$.

Ciascuno di questi approcci è stato parallelizzato (per quanto possibile) per permettere l'utilizzo della GPU, tramite [OpenCL](https://it.wikipedia.org/wiki/OpenCL).
Per quanto riguarda l'integrazione delle grandezze rispetto al tempo, si è deciso di utilizzare il metodo di integrazione [leapfrog](https://tarini.di.unimi.it/teaching/3DVideoGames2019/05_game_physics_part3.pdf), noto per preservare maggiormente l'energia dei sistemi.

## Struttura della repository

```
2d-gpu-n-body-sim/
├── .gitignore
├── .vscode/
├── galaxies/                           # alcune configurazioni di partenza
├── images/
├── Makefile
├── README.md                           # SEI QUI! :D
├── report/
│   ├── report.bib
│   ├── report.pdf                      # relazione (non finita)
│   └── report.tex                      
├── scripts/                            # alcuni script per valutare i risultati 
│                                       # generare configurazioni
├── src/
│   ├── barnes_hut_for_testing.c        # codice host di barnes hut
│   ├── headers/
│   │   ├── ocl_boiler.h                # shutout GPGPU's teacher
│   │   └── sim-utils.h                 # alcune macro e funzioni brutte ma utili
│   ├── kernels/
│   │   ├── barnes_hut.ocl              # codice dei kernel OpenCL :o
│   │   └── naive_nbody.ocl
│   └── naive_parallel_for_testing.c    # codice host di barnes hut
└── tests/                              # i risultati dei benchmark vanno qui! 
                                        # dovrei rinominare la cartella
```

## Eseguire i codici
1. Buildare il codice con `make`
2. Per l'approccio naive, usare 
```
./naive_parallel_for_testing.out <numero di corpi> <numero di frame> <nome della configurazione con estensione> <nome output> <0 solo benchmarking, 1 benchmarking e output>
```

```
./naive_parallel_for_testing.out <numero di corpi> <numero di frame> <nome della configurazione con estensione> <nome output> <theta> <0 solo benchmarking, 1 benchmarking e output>
```
## Alcuni esempi
Sistema a 18mila corpi con metodo naive.

[![18k-bodysim](https://img.youtube.com/vi/oSdXonzxpXk/0.jpg)](https://www.youtube.com/watch?v=oSdXonzxpXk)

```
./naive_parallel_for_testing.out 180000 999 galactic_chaos_180k.csv 180ksim_naive 1
```

Sistema a 18mila corpi con metodo Barnes-Hut con $\theta = 0.1$.

[![18k-bodysim-bh](https://img.youtube.com/vi/zR9vvkl9Xt0/0.jpg)](https://www.youtube.com/watch?v=zR9vvkl9Xt0)

```
./barnes_hut_for_testing.out 180000 999 galactic_chaos_180k.csv 180ksim_BH 0.1 1
```
