"""
Runner parallelo per tutti gli script biased learning.              [biased/run_all.py]
Lancia i 4 script (logit, rf, svm, lgbm) per tutte le combinazioni
di MODEL_NUMBER (1, 2, 3) e variante (pu, pnpu).

I job girano in parallelo (MAX_PARALLEL alla volta) come sottoprocessi.
Ogni riga di output è prefissata con [script|Mn|var] per distinguere i job.
I CSV vengono salvati normalmente da ogni script figlio.

CONFIGURAZIONE: modifica SCRIPTS, MODELS, VARIANTS, MAX_PARALLEL qui sotto.
"""

import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from itertools import product

SCRIPTS  = ["logit", "svm", "lgbm", "rf"]   # script da eseguire (senza .py)
MODELS   = [1, 2, 3]                        # MODEL_NUMBER
VARIANTS = ["pnpu", "pu"]                   # varianti

# Quanti job in parallelo al massimo.
# rf e lgbm usano tutti i core; con MAX_PARALLEL=2 si limita l'uso di memoria.
# Aumentare a 3-4 su macchine con RAM abbondante.
MAX_PARALLEL = 2

SCRIPT_DIR = Path(__file__).parent
PYTHON     = sys.executable

_print_lock = threading.Lock()


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _prefix(script: str, model: int, variant: str) -> str:
    return f"[{script:5s}|M{model}|{variant:4s}]"


def _stream(proc: subprocess.Popen, prefix: str):
    """Legge stdout del processo riga per riga, stampa con prefisso."""
    for raw in proc.stdout:
        line = raw.rstrip()
        if line:
            with _print_lock:
                print(f"{prefix} {line}", flush=True)
    proc.wait()


# Meccanismo: exec_module() esegue il codice del modulo (imposta MODEL_NUMBER=3
# e PNPU=True come default). Poi sovrascriviamo i globali del modulo DOPO
# exec_module, prima di chiamare main(). Le funzioni leggono i globali da
# mod.__dict__, che è lo stesso oggetto che modifichiamo → il trick funziona.

_WRAPPER_TEMPLATE = """\
import sys, importlib.util
sys.path.insert(0, r'{parent}')

spec = importlib.util.spec_from_file_location('_biased_mod', r'{script_path}')
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)   # esegue il modulo: imposta i default

# Override DOPO exec_module — sovrascrive i globali del modulo
mod.MODEL_NUMBER = {model}
mod.PNPU         = {pnpu}

mod.main()
"""


def run_job(script: str, model: int, variant: str) -> dict:
    prefix   = _prefix(script, model, variant)
    pnpu_str = "True" if variant == "pnpu" else "False"

    wrapper = _WRAPPER_TEMPLATE.format(
        parent      = str(SCRIPT_DIR.parents[2]),
        script_path = str(SCRIPT_DIR / f"{script}.py"),
        model       = model,
        pnpu        = pnpu_str,
    )

    t0 = time.time()
    with _print_lock:
        print(f"{prefix} [{_ts()}] AVVIO", flush=True)

    proc = subprocess.Popen(
        [PYTHON, "-u", "-c", wrapper],   # -u: stdout/stderr unbuffered → output immediato
        stdout   = subprocess.PIPE,
        stderr   = subprocess.STDOUT,
        text     = True,
        encoding = "utf-8",
        errors   = "replace",
    )
    _stream(proc, prefix)

    elapsed = int(time.time() - t0)
    status  = "OK" if proc.returncode == 0 else f"ERRORE rc={proc.returncode}"
    with _print_lock:
        m, s = divmod(elapsed, 60)
        print(f"{prefix} [{_ts()}] FINE — {status} ({m}m{s:02d}s)", flush=True)

    return {
        "script": script, "model": model, "variant": variant,
        "rc": proc.returncode, "elapsed_s": elapsed,
    }


def main(scripts=None, models=None, variants=None):
    scripts  = scripts  or SCRIPTS
    models   = models   or MODELS
    variants = variants or VARIANTS
    # Crea lista job; ordine: logit prima (leggero), rf per ultimo (pesante)
    order = {"logit": 0, "svm": 1, "lgbm": 2, "rf": 3}
    jobs  = sorted(
        product(scripts, models, variants),
        key=lambda j: (order.get(j[0], 9), j[1], j[2]),
    )
    total = len(jobs)

    print(f"\n  Biased Learning — Runner parallelo  [{_ts()}]")
    print(f"  Job: {total}  ({len(scripts)} script × {len(models)} model × {len(variants)} varianti)")
    print(f"  MAX_PARALLEL={MAX_PARALLEL}  |  python={PYTHON}")

    results   = []
    sem       = threading.Semaphore(MAX_PARALLEL)
    threads   = []
    res_lock  = threading.Lock()

    def _worker(script, model, variant):
        with sem:
            res = run_job(script, model, variant)
        with res_lock:
            results.append(res)

    for script, model, variant in jobs:
        t = threading.Thread(target=_worker, args=(script, model, variant), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(0.3)  # piccolo delay per evitare burst di avvii

    for t in threads:
        t.join()

    print(f"\n  RIEPILOGO  [{_ts()}]")
    results.sort(key=lambda r: (r["script"], r["model"], r["variant"]))
    for r in results:
        m, s = divmod(r["elapsed_s"], 60)
        flag = "✓" if r["rc"] == 0 else "✗"
        print(f"  {flag} {r['script']:5s}  M{r['model']}  {r['variant']:4s}  {m}m{s:02d}s")

    ok  = sum(1 for r in results if r["rc"] == 0)
    err = total - ok
    print(f"\n  Completati: {ok}/{total}  |  Errori: {err}/{total}")
    if err:
        print("  Job falliti:")
        for r in results:
            if r["rc"] != 0:
                print(f"    {r['script']} M{r['model']} {r['variant']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scripts", nargs="+", default=SCRIPTS,
                    choices=["logit", "svm", "lgbm", "rf"])
    ap.add_argument("--models",  nargs="+", type=int, default=MODELS, choices=[1,2,3])
    ap.add_argument("--variants",nargs="+", default=VARIANTS, choices=["pnpu","pu"])
    args = ap.parse_args()
    main(scripts=args.scripts, models=args.models, variants=args.variants)
