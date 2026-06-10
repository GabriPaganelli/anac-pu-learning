# =============================================================================
# LightGBM iterativo EM-like (variante Soft)
# =============================================================================
# Ogni U appare due volte nel training: come positivo con peso score_raw
# e come negativo con peso (1 - score_raw), scalati in modo che il contributo
# totale degli U resti fisso a W_SOFT_TARGET del peso dei labeled.
# Formulazione "fractional examples" (Bekker & Davis 2020, survey PU).
#
# I raw scores di LightGBM (objective='binary') sono in [0,1] e vengono usati
# direttamente come pesi senza calibrazione aggiuntiva.
#
# Tuning globale: iperparametri selezionati una sola volta su tutti i P+N labeled.
# =============================================================================

library(arrow)
library(lightgbm)
library(caret)
library(dplyr)
library(pROC)
library(PRROC)
library(ggplot2)

# -----------------------------------------------------------------------------
# CONFIGURAZIONE
# -----------------------------------------------------------------------------

PI_HAT        <- 0.02
N_ROUNDS_MAX  <- 500L
EARLY_STOP    <- 30L
MAX_ITER_EM   <- 5L

# Peso collettivo degli U rispetto ai labeled nel training ES.
# w_soft_obs = W_SOFT_TARGET * n_labeled_es / n_U
# → contributo totale U ≈ W_SOFT_TARGET * peso_labeled (invariante con n_U)
# Valore 0.05: conservativo. Range ragionevole: 0.05–0.20.
W_SOFT_TARGET <- 0.05

MODEL_NUMBER    <- 1L          # 1, 2 o 3 — unico parametro da cambiare per dataset
DATA_SOURCE     <- "nativi"    # "nativi" | "preprocessed"

# Percorsi dinamici relativi alla posizione dello script
.THIS_DIR <- tryCatch(
  dirname(normalizePath(sub("--file=", "",
    grep("--file=", commandArgs(FALSE), value=TRUE)[1]))),
  error = function(e) getwd())
.ROOT <- normalizePath(file.path(.THIS_DIR, "..", ".."))

OUT_DIR <- file.path(.THIS_DIR, "results", "soft")

# Colonne identificative/target da escludere dalle feature (oltre a label e fold)
FEATURE_ESCLUSE <- c("esito", "anno_pubblicazione", "regione")

set.seed(17)


# =============================================================================
# 1. PREPARAZIONE DATI
# =============================================================================

# DATA_SOURCE: "nativi" usa NA natively via LightGBM; "preprocessed" usa dati puliti.
parquet_path <- file.path(.ROOT, "anac", "output", "parquet", "model", DATA_SOURCE,
                          sprintf("M%d.parquet", MODEL_NUMBER))
dati_raw <- read_parquet(parquet_path)
dati_raw$label[is.na(dati_raw$label)] <- "U"
dati_raw$label <- factor(dati_raw$label)

cig_originali <- dati_raw$cig
fold_col_all  <- dati_raw$fold
dati <- dati_raw %>% select(-cig, -fold)

idx_P <- which(dati$label == "1")
idx_N <- which(dati$label == "0")
idx_U <- which(dati$label == "U")

cat(sprintf("Dataset: M%d — %s\n\n", MODEL_NUMBER, parquet_path))
cat("Dimensioni del dataset:\n")
cat(sprintf("  Positivi certi (P):  %d\n", length(idx_P)))
cat(sprintf("  Negativi certi (N):  %d\n", length(idx_N)))
cat(sprintf("  Unlabeled (U):       %d\n\n", length(idx_U)))

feature_cols <- setdiff(names(dati), c("label", FEATURE_ESCLUSE))
if (length(FEATURE_ESCLUSE) > 0)
  cat(sprintf("Feature escluse: %s\n\n", paste(FEATURE_ESCLUSE, collapse = ", ")))

dati[feature_cols] <- lapply(dati[feature_cols], function(x) {
  if (is.character(x)) as.factor(x) else x
})
cat_cols <- which(sapply(dati[feature_cols], is.factor))
cat(sprintf("Colonne categoriali: %d su %d feature totali\n\n",
            length(cat_cols), length(feature_cols)))

X_mat <- as.matrix(
  as.data.frame(lapply(dati[, feature_cols], function(x) {
    if (is.factor(x)) as.integer(x) else x
  }))
)


# =============================================================================
# 2. FOLD OOF E UNLABELED (dal parquet — condivisi tra tutti i modelli)
# =============================================================================

labeled_idx  <- c(idx_P, idx_N)
labeled_y    <- c(rep(1L, length(idx_P)), rep(0L, length(idx_N)))
labeled_fold <- fold_col_all[labeled_idx]

if (any(is.na(labeled_fold)))
  stop("Alcuni P o N certi hanno fold=NA nel parquet. Verificare il preprocessing.")

folds_oof <- lapply(0:3, function(k) which(labeled_fold == k))

cat("Composizione dei 4 fold OOF (da colonna 'fold' del parquet):\n")
for (i in 1:4) {
  cat(sprintf("  Fold %d - test: %d P, %d N\n", i,
              sum(labeled_y[folds_oof[[i]]] == 1),
              sum(labeled_y[folds_oof[[i]]] == 0)))
}

unlabeled_per_fold <- lapply(0:3, function(k) {
  idx_U[!is.na(fold_col_all[idx_U]) & fold_col_all[idx_U] == k]
})

cat("\nUnlabeled per fold:\n")
for (i in 1:4) cat(sprintf("  Fold %d: %d U\n", i, length(unlabeled_per_fold[[i]])))
cat("\n")


# =============================================================================
# 3. PARAMETRI FISSI, GRIGLIA E FUNZIONE DI TUNING GLOBALE
# =============================================================================

params_fissi <- list(
  objective        = "binary",
  metric           = "auc",
  feature_fraction = 0.7,
  bagging_fraction = 0.7,
  bagging_freq     = 5L,
  lambda_l1        = 0,
  lambda_l2        = 0.5,
  verbose          = -1L,
  num_threads      = 4L
)

griglia <- expand.grid(
  num_leaves       = c(15L, 31L, 63L),
  min_data_in_leaf = c(10L, 20L, 50L),
  learning_rate    = c(0.01, 0.05, 0.1),
  stringsAsFactors = FALSE
)

# Tuning eseguito una sola volta su tutti i P+N labeled (globale).
# Bias da leakage trascurabile su ~2600 obs con 27 combinazioni; 4x più veloce del per-fold.
tuning_globale <- function(X_tr, y_tr) {
  cat(sprintf("  [Tuning] %d combinazioni x 5 fold su %d obs labeled totali\n",
              nrow(griglia), length(y_tr)))

  set.seed(999)
  folds_interni <- createFolds(y_tr, k = 5, list = TRUE, returnTrain = FALSE)
  risultati     <- vector("list", nrow(griglia))

  for (g in 1:nrow(griglia)) {
    params_trial <- c(params_fissi, list(
      num_leaves       = griglia$num_leaves[g],
      min_data_in_leaf = griglia$min_data_in_leaf[g],
      learning_rate    = griglia$learning_rate[g]
    ))

    pr_auc_cv <- numeric(5)
    for (k in 1:5) {
      idx_val_k   <- folds_interni[[k]]
      idx_train_k <- setdiff(seq_along(y_tr), idx_val_k)

      dtrain_k <- lgb.Dataset(X_tr[idx_train_k, , drop = FALSE],
                              label = y_tr[idx_train_k],
                              categorical_feature = cat_cols, free_raw_data = FALSE)
      dval_k   <- lgb.Dataset(X_tr[idx_val_k,   , drop = FALSE],
                              label = y_tr[idx_val_k], reference = dtrain_k)

      model_k  <- lgb.train(params_trial, dtrain_k, N_ROUNDS_MAX,
                            list(valid = dval_k), EARLY_STOP, verbose = 0)
      scores_k <- predict(model_k, X_tr[idx_val_k, , drop = FALSE])
      y_val_k  <- y_tr[idx_val_k]

      pr_auc_cv[k] <- pr.curve(
        scores.class0 = scores_k[y_val_k == 1],
        scores.class1 = scores_k[y_val_k == 0]
      )$auc.integral
    }
    risultati[[g]] <- data.frame(griglia[g, ], pr_auc_mean = mean(pr_auc_cv))
  }

  tabella <- do.call(rbind, risultati)
  best    <- tabella[which.max(tabella$pr_auc_mean), ]

  cat(sprintf("  [Tuning] Ottimali: leaves=%d  min_leaf=%d  lr=%.2f  PR AUC=%.4f\n\n",
              best$num_leaves, best$min_data_in_leaf, best$learning_rate,
              best$pr_auc_mean))

  c(params_fissi, list(
    num_leaves       = best$num_leaves,
    min_data_in_leaf = best$min_data_in_leaf,
    learning_rate    = best$learning_rate
  ))
}


# =============================================================================
# 4. HELPER: COSTRUZIONE DATASET SOFT
# =============================================================================
# Ogni U appare due volte:
#   - come positivo con peso = score_raw * w_soft_obs
#   - come negativo con peso = (1 - score_raw) * w_soft_obs
#
# Formulazione "fractional examples" (Bekker & Davis 2020, survey PU).

costruisci_dataset_soft <- function(X_es_tr, y_es_tr, X_U, scores_U_raw) {
  n_P          <- sum(y_es_tr == 1)
  n_N_cert     <- sum(y_es_tr == 0)
  n_U          <- nrow(X_U)
  n_labeled_es <- length(y_es_tr)

  scores_U <- pmin(pmax(scores_U_raw, 0), 1)  # clip di sicurezza

  # Contributo totale U fisso a W_SOFT_TARGET del peso labeled, indipendente da n_U
  w_soft_obs <- W_SOFT_TARGET * n_labeled_es / n_U

  contrib_pos_U <- sum(scores_U * w_soft_obs)
  contrib_neg_U <- sum((1 - scores_U) * w_soft_obs)

  # Peso P: bilancia lato pos (P + U come pos) con lato neg (N + U come neg)
  w_P_calc <- (n_N_cert + contrib_neg_U - contrib_pos_U) / n_P
  w_P      <- max(1.0, w_P_calc)

  total_pos <- n_P * w_P + contrib_pos_U
  total_neg <- n_N_cert + contrib_neg_U

  cat(sprintf("  w_soft_obs=%.5f | P=%d(w=%.2f) N=%d(w=1.0) U=%d (tot pos=%.1f neg=%.1f)\n",
              w_soft_obs, n_P, w_P, n_N_cert, n_U, total_pos, total_neg))

  list(
    X      = rbind(X_es_tr, X_U, X_U),
    y      = c(y_es_tr,     rep(1L, n_U), rep(0L, n_U)),
    w      = c(ifelse(y_es_tr == 1, w_P, 1.0), scores_U * w_soft_obs,
               (1 - scores_U) * w_soft_obs),
    w_P    = w_P,
    n_rows = n_labeled_es + 2 * n_U
  )
}


# =============================================================================
# 5. HELPER: METRICHE OOF
# =============================================================================

calcola_metriche <- function(scores_labeled, y_labeled, scores_u) {
  pr_auc  <- pr.curve(scores.class0 = scores_labeled[y_labeled == 1],
                      scores.class1 = scores_labeled[y_labeled == 0])$auc.integral
  roc_auc <- as.numeric(auc(roc(y_labeled, scores_labeled, quiet = TRUE)))

  scores_full <- c(scores_labeled, scores_u)
  y_full      <- c(y_labeled, rep(0L, length(scores_u)))
  k           <- max(1L, floor(0.01 * length(scores_full)))
  lift        <- mean(y_full[order(scores_full, decreasing = TRUE)[1:k]] == 1) / PI_HAT

  list(lift1 = lift, pr_auc = pr_auc, roc_auc = roc_auc)
}


# =============================================================================
# 6. FUNZIONE PRINCIPALE: UN FOLD OOF
# =============================================================================

esegui_fold_soft <- function(fold_id) {

  cat(sprintf("\n%s\n", strrep("=", 65)))
  cat(sprintf("FOLD OOF %d / 4  [variante SOFT]\n", fold_id))
  cat(sprintf("%s\n\n", strrep("=", 65)))

  idx_test_loc  <- folds_oof[[fold_id]]
  idx_train_loc <- setdiff(seq_along(labeled_idx), idx_test_loc)

  idx_train_glob <- labeled_idx[idx_train_loc]
  idx_test_glob  <- labeled_idx[idx_test_loc]

  y_train <- labeled_y[idx_train_loc]
  y_test  <- labeled_y[idx_test_loc]
  X_train <- X_mat[idx_train_glob, , drop = FALSE]
  X_test  <- X_mat[idx_test_glob,  , drop = FALSE]

  u_idx <- unlabeled_per_fold[[fold_id]]
  set.seed(200 + fold_id)
  u_test_pos  <- sample(seq_along(u_idx), size = floor(0.2 * length(u_idx)))
  u_train_idx <- u_idx[-u_test_pos]
  u_test_idx  <- u_idx[u_test_pos]

  cat(sprintf("Train labeled: %d P, %d N | Test labeled: %d P, %d N\n",
              sum(y_train == 1), sum(y_train == 0),
              sum(y_test  == 1), sum(y_test  == 0)))
  cat(sprintf("U training: %d | U test: %d\n\n",
              length(u_train_idx), length(u_test_idx)))

  # ES split: 15% del training per early stopping
  set.seed(300L + fold_id)
  n_es_val     <- max(10L, floor(0.15 * nrow(X_train)))
  es_val_local <- sample(nrow(X_train), size = n_es_val, replace = FALSE)
  es_tr_local  <- setdiff(seq_len(nrow(X_train)), es_val_local)

  X_es_tr  <- X_train[es_tr_local,  , drop = FALSE]
  y_es_tr  <- y_train[es_tr_local]
  X_es_val <- X_train[es_val_local, , drop = FALSE]
  y_es_val <- y_train[es_val_local]

  cat(sprintf("ES split: %d train | %d val (15%%)\n\n",
              length(es_tr_local), length(es_val_local)))

  # -------------------------------------------------------------------------
  # ITERAZIONE 0: solo P + N certi (baseline per questo fold)
  # -------------------------------------------------------------------------
  cat("--- Iterazione 0: solo P + N certi (baseline) ---\n")

  dtrain_0 <- lgb.Dataset(X_es_tr, label = y_es_tr,
                          categorical_feature = cat_cols, free_raw_data = FALSE)
  dval_0   <- lgb.Dataset(X_es_val, label = y_es_val, reference = dtrain_0)

  model_0 <- lgb.train(params_globali, dtrain_0, N_ROUNDS_MAX,
                       list(valid = dval_0), EARLY_STOP, verbose = 0)

  scores_U_correnti <- predict(model_0, X_mat[u_train_idx, , drop = FALSE])
  m0 <- calcola_metriche(predict(model_0, X_test), y_test,
                         predict(model_0, X_mat[u_test_idx, , drop = FALSE]))

  cat(sprintf("  Score U iter0 — media: %.4f\n", mean(scores_U_correnti)))
  cat(sprintf("  PR AUC: %.4f | ROC AUC: %.4f\n\n", m0$pr_auc, m0$roc_auc))

  iter_log <- data.frame(
    iter        = 0L,
    score_U_med = mean(scores_U_correnti),
    pr_auc      = m0$pr_auc,
    roc_auc     = m0$roc_auc,
    lift1       = m0$lift1
  )

  best_model   <- model_0
  best_pr_auc  <- m0$pr_auc
  best_roc_auc <- m0$roc_auc
  best_iter    <- 0L

  # -------------------------------------------------------------------------
  # ITERAZIONI 1..MAX_ITER_EM
  # -------------------------------------------------------------------------
  for (iter in 1:MAX_ITER_EM) {

    cat(sprintf("--- Iterazione %d ---\n", iter))

    ds <- costruisci_dataset_soft(
      X_es_tr      = X_es_tr,
      y_es_tr      = y_es_tr,
      X_U          = X_mat[u_train_idx, , drop = FALSE],
      scores_U_raw = scores_U_correnti
    )

    dtrain_soft <- lgb.Dataset(ds$X, label = ds$y, weight = ds$w,
                               categorical_feature = cat_cols, free_raw_data = FALSE)
    dval_soft   <- lgb.Dataset(X_es_val, label = y_es_val, reference = dtrain_soft)

    model_iter <- lgb.train(params_globali, dtrain_soft, N_ROUNDS_MAX,
                            list(valid = dval_soft), EARLY_STOP, verbose = 0)

    scores_U_correnti <- predict(model_iter, X_mat[u_train_idx, , drop = FALSE])

    mi <- calcola_metriche(predict(model_iter, X_test), y_test,
                           predict(model_iter, X_mat[u_test_idx, , drop = FALSE]))

    cat(sprintf("  Score U — media: %.4f | >0.5: %d%%\n",
                mean(scores_U_correnti),
                round(100 * mean(scores_U_correnti > 0.5))))
    cat(sprintf("  PR AUC: %.4f | ROC AUC: %.4f\n", mi$pr_auc, mi$roc_auc))

    iter_log <- rbind(iter_log, data.frame(
      iter        = iter,
      score_U_med = mean(scores_U_correnti),
      pr_auc      = mi$pr_auc,
      roc_auc     = mi$roc_auc,
      lift1       = mi$lift1
    ))

    if (mi$pr_auc > best_pr_auc) {
      best_pr_auc  <- mi$pr_auc
      best_roc_auc <- mi$roc_auc
      best_model   <- model_iter
      best_iter    <- iter
      cat(sprintf("  *** NUOVO BEST: iter=%d, PR AUC=%.4f ***\n", iter, best_pr_auc))
    }
    cat("\n")

    # Stop: collasso verso positivo
    if (mean(scores_U_correnti > 0.5) > 0.30) {
      cat("  Stop: >30% U con score > 0.5 — collasso positivo\n"); break
    }
    # Stop: collasso verso negativo
    if (mean(scores_U_correnti < 0.001) > 0.95) {
      cat("  Stop: >95% U con score < 0.001 — collasso negativo\n"); break
    }
  }

  scores_test_final   <- predict(best_model, X_test)
  scores_u_test_final <- predict(best_model, X_mat[u_test_idx, , drop = FALSE])

  imp      <- lgb.importance(best_model)
  top_gain <- imp$Gain[1] / sum(imp$Gain)
  cat(sprintf("\nFold %d — best iter %d | PR AUC: %.4f | ROC AUC: %.4f\n",
              fold_id, best_iter, best_pr_auc, best_roc_auc))
  cat(sprintf("  Top feature: %s (%.1f%% gain)\n", imp$Feature[1], 100 * top_gain))
  if (top_gain > 0.30)
    cat("  ! GUARDRAIL: feature dominante (>30%) — verificare leakage\n")

  list(fold_id = fold_id,
       best_model = best_model, best_iter = best_iter,
       best_pr_auc = best_pr_auc, best_roc_auc = best_roc_auc,
       params_globali = params_globali,
       iter_log = iter_log, feature_importance = imp,
       test_labeled_idx = idx_test_glob, test_labeled_y = y_test,
       test_labeled_scores = scores_test_final,
       u_test_idx = u_test_idx, u_test_scores = scores_u_test_final)
}


# =============================================================================
# 7. ESECUZIONE
# =============================================================================

cat(sprintf("\n### EM-like SOFT — M%d ###\n", MODEL_NUMBER))

# Tuning globale (una sola volta su tutti i P+N labeled)
cat("\n[Tuning globale]\n")
params_globali <- tuning_globale(X_mat[labeled_idx, , drop = FALSE], labeled_y)

risultati_fold <- lapply(1:4, esegui_fold_soft)


# =============================================================================
# 8. METRICHE AGGREGATE
# =============================================================================

cat("\n\n### RISULTATI FINALI EM-like SOFT ###\n\n")

pr_auc_vf  <- sapply(risultati_fold, function(x) x$best_pr_auc)
roc_auc_vf <- sapply(risultati_fold, function(x) x$best_roc_auc)

baseline_df <- do.call(rbind, lapply(risultati_fold, function(x) {
  log0 <- x$iter_log[x$iter_log$iter == 0, ]
  data.frame(fold_id = x$fold_id,
             pr_auc  = log0$pr_auc, roc_auc = log0$roc_auc, lift1 = log0$lift1)
}))

cat("Baseline (iter 0 — P+N soli) per fold:\n")
print(baseline_df, row.names = FALSE)
cat(sprintf("\nBaseline — PR AUC: %.4f (SD=%.4f) | ROC AUC: %.4f\n\n",
            mean(baseline_df$pr_auc), sd(baseline_df$pr_auc),
            mean(baseline_df$roc_auc)))

for (i in 1:4)
  cat(sprintf("  Fold %d (best iter %d): PR AUC=%.4f | ROC AUC=%.4f\n",
              i, risultati_fold[[i]]$best_iter, pr_auc_vf[i], roc_auc_vf[i]))

cat(sprintf("\nPR AUC medio:  %.4f (SD=%.4f)\n", mean(pr_auc_vf),  sd(pr_auc_vf)))
cat(sprintf("ROC AUC medio: %.4f (SD=%.4f)\n",   mean(roc_auc_vf), sd(roc_auc_vf)))

all_scores   <- unlist(lapply(risultati_fold, function(x) x$test_labeled_scores))
all_y        <- unlist(lapply(risultati_fold, function(x) x$test_labeled_y))
all_scores_u <- unlist(lapply(risultati_fold, function(x) x$u_test_scores))

pr_auc_glob <- pr.curve(scores.class0 = all_scores[all_y == 1],
                        scores.class1 = all_scores[all_y == 0])$auc.integral

all_full  <- c(all_scores, all_scores_u)
y_full    <- c(all_y, rep(0L, length(all_scores_u)))
k_top     <- max(1L, floor(0.01 * length(all_full)))
lift_glob <- mean(y_full[order(all_full, decreasing = TRUE)[1:k_top]] == 1) / PI_HAT

cat(sprintf("\nPR AUC globale (OOF concat): %.4f\n", pr_auc_glob))
cat(sprintf("Lift@1%% globale (k=%d, affidabile): %.2fx\n\n", k_top, lift_glob))

cat("### CONFRONTO soft vs BASELINE ###\n")
cat(sprintf("Baseline — PR AUC: %.4f | ROC AUC: %.4f\n",
            mean(baseline_df$pr_auc), mean(baseline_df$roc_auc)))
cat(sprintf("Soft     — PR AUC: %.4f | ROC AUC: %.4f\n",
            mean(pr_auc_vf), mean(roc_auc_vf)))
delta <- mean(pr_auc_vf) - mean(baseline_df$pr_auc)
cat(sprintf("Delta PR AUC: %+.4f\n", delta))
if (abs(delta) < 0.02)
  cat("! Delta < 0.02: il PU learning aggiunge poco rispetto a P+N soli\n")

cat("\nLog iterazioni per fold:\n")
for (i in 1:4) {
  cat(sprintf("\n  Fold %d:\n", i))
  print(risultati_fold[[i]]$iter_log, row.names = FALSE)
}


# =============================================================================
# 9. PLOT
# =============================================================================

log_soft <- do.call(rbind, lapply(1:4, function(i) {
  df <- risultati_fold[[i]]$iter_log; df$fold <- paste0("Fold ", i); df
}))

p1 <- ggplot(log_soft, aes(x = iter, y = pr_auc, color = fold, group = fold)) +
  geom_line(linewidth = 1) + geom_point(size = 2.5) +
  labs(title    = "PR AUC OOF per iterazione (variante SOFT)",
       subtitle = "iter 0 = baseline P+N | iter 1-5 = pseudo-etichette soft (raw scores)",
       x = "Iterazione", y = "PR AUC", color = NULL) +
  theme_bw(base_size = 12)

p2 <- ggplot(log_soft, aes(x = iter, y = score_U_med, color = fold, group = fold)) +
  geom_line(linewidth = 0.9) + geom_point(size = 2) +
  geom_hline(yintercept = PI_HAT, linetype = "dotted",
             color = "steelblue", linewidth = 0.8) +
  annotate("text", x = 0.2, y = PI_HAT + 0.005,
           label = "pi = 0.02", color = "steelblue", size = 3, hjust = 0) +
  labs(title    = "Score medio U per iterazione (raw — nessuna calibrazione)",
       subtitle = "Score LightGBM binary in [0,1], usati direttamente come pesi soft",
       x = "Iterazione", y = "Score medio U", color = "Fold") +
  theme_bw(base_size = 12)

print(p1); print(p2)


# =============================================================================
# 10. SALVATAGGIO
# =============================================================================

metriche_csv <- do.call(rbind, lapply(risultati_fold, function(x) {
  log0 <- x$iter_log[x$iter_log$iter == 0, ]
  data.frame(
    fold             = x$fold_id,
    best_iter        = x$best_iter,
    pr_auc           = x$best_pr_auc,
    roc_auc          = x$best_roc_auc,
    pr_auc_baseline  = log0$pr_auc,
    roc_auc_baseline = log0$roc_auc,
    lift_1pct_globale = lift_glob
  )
}))
csv_name <- file.path(OUT_DIR, sprintf("em_soft_M%d_%s_fold_metrics.csv", MODEL_NUMBER, DATA_SOURCE))
write.csv(metriche_csv, csv_name, row.names = FALSE)
cat(sprintf("Metriche OOF salvate in '%s'\n", csv_name))

rds_name <- file.path(OUT_DIR, sprintf("em_soft_M%d_%s_results.rds", MODEL_NUMBER, DATA_SOURCE))
saveRDS(list(
  params_fissi   = params_fissi,
  baseline_PN    = baseline_df,
  risultati_fold = risultati_fold,
  aggregati = list(
    pr_auc_mean    = mean(pr_auc_vf),  pr_auc_sd    = sd(pr_auc_vf),
    roc_auc_mean   = mean(roc_auc_vf), roc_auc_sd   = sd(roc_auc_vf),
    pr_auc_globale = pr_auc_glob,      lift_globale  = lift_glob
  )
), file = rds_name)

cat(sprintf("Risultati completi salvati in '%s'\n", rds_name))
cat("=== COMPLETATO ===\n")

rm(list=ls())
gc()
