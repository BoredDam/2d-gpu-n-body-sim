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


## Alcuni esempi
Sistema a 18mila corpi con metodo naive.

[![18k-bodysim](https://img.youtube.com/vi/oSdXonzxpXk/0.jpg)](https://www.youtube.com/watch?v=oSdXonzxpXk)

Sistema a 18mila corpi con metodo Barnes-Hut con $\theta = 0.1$.

[![18k-bodysim-bh](https://img.youtube.com/vi/zR9vvkl9Xt0/0.jpg)](https://www.youtube.com/watch?v=zR9vvkl9Xt0)
