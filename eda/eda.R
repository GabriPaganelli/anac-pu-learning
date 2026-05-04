# EDA Appalti — Analisi esplorativa completa
# Script autonomo: carica M1, M2, M3 nativi e produce tutti
# i grafici esplorativi del progetto.
#
#  01. Funnel del dataset M1 → M2 → M3
#  02. Feature signature (Cliff's δ e log₂-ratio binari)
#  03. Geografia della corruzione (regionale + contesto)
#  04. Importi e tipo procedura (ridge + heatmap)
#  05. Dinamica competitiva dei bandi aggiudicati (M2)
#  06. Anomalie di esecuzione accumulate sui condannati (M3)
#  07. Missingness come segnale (confronto P vs N)
#  08. Evoluzione temporale e mix procedurale
#  09. Leakage watch (flag_delega e importo per fase)
#
#  10. Benford's Law su importo_lotto
#  11. Threshold bunching alle soglie del Codice Appalti
#  12. Corruption Risk Indicators (CRI) di Fazekas et al.
#  13. Bid-rigging indicators (OECD 2009 / Imhof et al.)
#  14. Settore SANITARIO: esposizione corruttiva documentata
#  15. Distribuzione temporale dei contratti labeled
#  16. Missing data heatmap per feature M1
#  17. Tasso condanne per regione
#
#  18. UMAP + Gower — griglia 3×2 (etichetta + tipo procedura)
#  ++  umap_landscape.pdf — figura 4×3 per il report LaTeX
#
# Referenze:
# - Cliff (1993) — Dominance statistics
# - Fazekas, Toth, King (2016) — European J. Crim. Pol.
# - OECD (2009) — Guidelines for Fighting Bid Rigging
# - Nigrini (2012) — Benford's Law: Forensic Accounting
# - Imhof, Karagok, Rutz (2018) — Screening for Bid Rigging
# - Coviello & Mariniello (2014) — Publicity Requirements
#
# Lancio: Rscript eda/eda.R
#         oppure source() da RStudio con working dir = root progetto

suppressPackageStartupMessages({
  library(arrow); library(dplyr); library(tidyr); library(ggplot2)
  library(patchwork); library(scales); library(forcats); library(stringr)
  library(ggridges); library(ggrepel)
})
if (!requireNamespace("uwot",    quietly = TRUE))
  stop("Installa uwot: install.packages('uwot')")
if (!requireNamespace("cluster", quietly = TRUE))
  stop("Installa cluster: install.packages('cluster')")

set.seed(17)

ROOT <- local({
  args      <- commandArgs(trailingOnly = FALSE)
  file_flag <- grep("^--file=", args, value = TRUE)
  if (length(file_flag) > 0)
    return(normalizePath(file.path(dirname(sub("^--file=", "", file_flag)), "..")))
  ofile <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (!is.null(ofile) && nzchar(ofile))
    return(normalizePath(file.path(dirname(ofile), "..")))
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    ctx <- rstudioapi::getActiveDocumentContext()$path
    if (nzchar(ctx)) return(normalizePath(file.path(dirname(ctx), "..")))
  }
  message("[WARN] Path script non rilevato — assumo che il working dir sia la root del progetto.")
  normalizePath(".")
})

NATIVI_PATH <- file.path(ROOT, "anac", "output", "parquet", "model", "nativi")
OUTDIR      <- file.path(ROOT, "eda", "plots")
CACHEDIR    <- file.path(ROOT, "eda", "cache")
dir.create(OUTDIR,   recursive = TRUE, showWarnings = FALSE)
dir.create(CACHEDIR, recursive = TRUE, showWarnings = FALSE)

message("ROOT:     ", ROOT)
message("OUTDIR:   ", OUTDIR)
message("CACHEDIR: ", CACHEDIR)

save_plot <- function(p, nome, w = 14, h = 9) {
  ggsave(file.path(OUTDIR, paste0(nome, ".png")), p,
         width = w, height = h, dpi = 150, bg = "white")
  message("  salvato: ", nome, ".png")
}

wilson_ci <- function(k, n, z = 1.96) {
  if (n == 0) return(c(NA_real_, NA_real_))
  p <- k / n; denom <- 1 + z^2 / n
  centro <- (p + z^2 / (2 * n)) / denom
  half   <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / denom
  c(centro - half, centro + half)
}

prima_cifra <- function(x) {
  x <- x[!is.na(x) & x > 0]
  as.integer(substr(format(x, scientific = FALSE, trim = TRUE), 1, 1))
}

cliff_delta <- function(x_P, x_N) {
  x_P <- na.omit(as.numeric(x_P)); x_N <- na.omit(as.numeric(x_N))
  if (length(x_P) < 2 || length(x_N) < 2) return(NA_real_)
  m <- length(x_P); n <- length(x_N)
  U <- suppressWarnings(wilcox.test(x_P, x_N, exact = FALSE))$statistic
  as.numeric(2 * U / (m * n) - 1)
}

col_N <- "#2E86AB"; col_P <- "#C73E1D"; col_U <- "#B0BEC5"
col_models <- c("M1 (ex ante)"        = "#E69F00",
                "M2 (aggiudicazione)" = "#009E73",
                "M3 (esecuzione)"     = "#CC79A7")

theme_paper <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(plot.title       = element_text(face = "bold", size = rel(1.1)),
          plot.subtitle    = element_text(color = "grey30", size = rel(0.9)),
          plot.caption     = element_text(color = "grey40", size = rel(0.8), hjust = 0),
          panel.grid.minor = element_blank(),
          panel.border     = element_rect(color = "grey90", fill = NA, linewidth = 0.3),
          strip.text       = element_text(face = "bold"),
          legend.position  = "top")
}

scale_lab <- scale_fill_manual(
  values = c("Assolto (N)" = col_N, "Condannato (P)" = col_P), name = NULL)

message("Carico M1 nativi (9.47M righe)...")
m1 <- read_parquet(file.path(NATIVI_PATH, "M1.parquet"))

message("Carico M2 nativi (3.69M righe)...")
m2 <- read_parquet(file.path(NATIVI_PATH, "M2.parquet"))

message("Carico M3 nativi (1.13M righe)...")
m3 <- read_parquet(file.path(NATIVI_PATH, "M3.parquet"))

message("Carico metadata (anno_pubblicazione, regione)...")
meta <- read_parquet(
  file.path(ROOT, "anac", "output", "parquet", "bando_cig_all.parquet"),
  col_select = c("cig", "anno_pubblicazione", "regione")
)
m1 <- left_join(m1, meta, by = "cig")
m2 <- left_join(m2, meta, by = "cig")
m3 <- left_join(m3, meta, by = "cig")
rm(meta)
message("  Metadata uniti a M1/M2/M3.")

make_labeled <- function(df) {
  df |> filter(!is.na(label)) |>
    mutate(label_fct = factor(label, levels = c(0, 1),
                               labels = c("Assolto (N)", "Condannato (P)")))
}
labeled_m1 <- make_labeled(m1)
labeled_m2 <- make_labeled(m2)
labeled_m3 <- make_labeled(m3)

message(sprintf("Labeled: M1=%d (%dP/%dN)  M2=%d (%dP/%dN)  M3=%d (%dP/%dN)",
  nrow(labeled_m1), sum(labeled_m1$label == 1), sum(labeled_m1$label == 0),
  nrow(labeled_m2), sum(labeled_m2$label == 1), sum(labeled_m2$label == 0),
  nrow(labeled_m3), sum(labeled_m3$label == 1), sum(labeled_m3$label == 0)))

FEAT_M1_NUM <- c(
  "importo_lotto", "importo_complessivo_gara", "n_lotti_componenti",
  "finestra_offerta_giorni", "lag_perfezionamento_giorni",
  "importo_sicurezza_pct", "pct_riserva_base",
  "tasso_disoccupazione", "reddito_irpef_procapite", "tasso_omicidi_100k"
)
FEAT_M1_CAT <- c(
  "tipo_scelta_4cls", "cpv_macro_categoria", "natura_giuridica_SA",
  "cod_modalita_realizzazione", "sezione_regionale", "tipo_lavorazione_macro"
)
FEAT_M1_BIN <- c(
  "flag_urgenza", "flag_accordo_quadro", "flag_ripetizioni",
  "settore_speciale", "flag_appalto_riservato"
)
FEAT_M2_EXTRA_NUM <- c(
  "ribasso_aggiudicazione", "num_imprese_offerenti",
  "numero_offerte_ammesse", "numero_offerte_escluse", "pct_offerte_escluse",
  "lag_aggiudicazione_giorni", "lag_comunicazione_esito_giorni",
  "importo_aggiudicazione", "ribasso_spread"
)
FEAT_M2_EXTRA_CAT <- c("tipo_soggetto_agg", "flag_vince_minimo", "flag_progettazione_esterna")
FEAT_M2_EXTRA_BIN <- c("asta_elettronica", "flag_scomputo", "flag_proc_accelerata", "flag_subappalto")
FEAT_M3_EXTRA_NUM <- c(
  "lag_stipula_aggiudicazione_giorni", "durata_pianificata_giorni",
  "n_varianti", "n_sospensioni", "durata_totale_sospensioni_gg",
  "pct_durata_sospesa", "n_sal"
)
FEAT_M3_EXTRA_CAT <- c("consegna_frazionata", "consegna_sotto_riserva", "flag_in_ritardo")
FEAT_M3_EXTRA_BIN <- c(
  "flag_variante_sostanziale", "flag_variante_oltre_termine",
  "flag_sospensione", "flag_sosp_giudiziaria", "flag_proroga"
)
# binarie con flag_delega (per la feature signature)
FEAT_M1_BIN_EXT <- c(FEAT_M1_BIN, "flag_delega")

# Il tasso P cresce nelle fasi più avanzate: selection bias
# (solo i bandi aggiudicati/eseguiti sopravvivono).
message("01 - Funnel del dataset")

n_tot <- nrow(m1)

funnel_df <- tibble(
  modello = factor(c("M1 (ex ante)", "M2 (aggiudicazione)", "M3 (esecuzione)"),
                   levels = c("M1 (ex ante)", "M2 (aggiudicazione)", "M3 (esecuzione)")),
  n_P = c(sum(labeled_m1$label == 1), sum(labeled_m2$label == 1), sum(labeled_m3$label == 1)),
  n_N = c(sum(labeled_m1$label == 0), sum(labeled_m2$label == 0), sum(labeled_m3$label == 0))
) |> mutate(tot = n_P + n_N, pct_P = n_P / tot * 100)

funnel_long <- funnel_df |>
  pivot_longer(c(n_N, n_P), names_to = "gruppo", values_to = "n") |>
  mutate(gruppo = factor(
    ifelse(gruppo == "n_P", "Condannato (P)", "Assolto (N)"),
    levels = c("Assolto (N)", "Condannato (P)")
  ))

p01a <- ggplot(funnel_long, aes(x = modello, y = n, fill = gruppo)) +
  geom_col(width = 0.55, alpha = 0.9) +
  geom_text(data = funnel_df,
            aes(x = modello, y = tot,
                label = sprintf("%s\n(%.1f%% P)", format(tot, big.mark = "'"), pct_P)),
            inherit.aes = FALSE, vjust = -0.3, fontface = "bold", size = 4.2) +
  scale_lab +
  scale_y_continuous(expand = expansion(mult = c(0, 0.18)), labels = label_comma()) +
  labs(title = "Funnel del dataset: M1 → M2 → M3",
       subtitle = "Il tasso di condannati cresce nelle fasi più avanzate (selection bias)",
       x = NULL, y = "N° appalti labeled") +
  theme_paper() + theme(axis.text.x = element_text(size = 11))

n_lab    <- nrow(labeled_m1)
donut_df <- tibble(
  gruppo = c("Assolto (N)", "Condannato (P)", "Unlabeled (U)"),
  n      = c(sum(labeled_m1$label == 0), sum(labeled_m1$label == 1), n_tot - n_lab)
)

p01b <- ggplot(donut_df, aes(x = 2, y = n, fill = gruppo)) +
  geom_col(width = 1, color = "white", linewidth = 0.6) +
  coord_polar(theta = "y") + xlim(0.3, 2.5) +
  annotate("text", x = 0.3, y = 0,
           label = sprintf("%.2f%%\nlabeled", n_lab / n_tot * 100),
           size = 5.5, fontface = "bold", color = "grey20", hjust = 0.5) +
  scale_fill_manual(values = c("Assolto (N)" = col_N, "Condannato (P)" = col_P,
                                "Unlabeled (U)" = col_U), name = "gruppo") +
  labs(title    = "Composizione del dataset completo",
       subtitle = sprintf("Totale: %s appalti", format(n_tot, big.mark = "'"))) +
  theme_paper() +
  theme(axis.text = element_blank(), axis.title = element_blank(),
        panel.grid = element_blank(), panel.border = element_blank(),
        legend.position = "bottom")

p01 <- (p01a | p01b) +
  plot_annotation(
    caption = "M1 = tutti i labeled · M2 = con importo di aggiudicazione · M3 = M2 ∩ durata esecuzione presente",
    theme   = theme(plot.caption = element_text(color = "grey40", size = rel(0.85), hjust = 0))
  )

save_plot(p01, "01_panoramica_funnel", w = 16, h = 7)

# Cliff's δ (numeriche) e log₂(P/N) (binarie).
# |δ| > 0.147 = piccolo, > 0.33 = medio, > 0.474 = grande.
message("02 - Feature signature")

compute_cliff <- function(df, feats, model_name) {
  feats_ok <- intersect(feats, names(df))
  do.call(bind_rows, lapply(feats_ok, function(feat) {
    tibble(feature = feat, modello = model_name,
           delta   = cliff_delta(df[[feat]][df$label == 1],
                                  df[[feat]][df$label == 0]))
  }))
}

cliff_df <- bind_rows(
  compute_cliff(labeled_m1, FEAT_M1_NUM, "M1 (ex ante)"),
  compute_cliff(labeled_m2, c(FEAT_M1_NUM, FEAT_M2_EXTRA_NUM), "M2 (aggiudicazione)"),
  compute_cliff(labeled_m3, c(FEAT_M1_NUM, FEAT_M2_EXTRA_NUM, FEAT_M3_EXTRA_NUM),
                "M3 (esecuzione)")
) |> filter(!is.na(delta)) |>
  mutate(modello = factor(modello, levels = names(col_models)))

feat_order_num <- cliff_df |>
  group_by(feature) |> summarise(med = median(abs(delta)), .groups = "drop") |>
  arrange(med) |> pull(feature)
cliff_df <- cliff_df |> mutate(feature = factor(feature, levels = feat_order_num))

p02a <- ggplot(cliff_df, aes(x = delta, y = feature, color = modello)) +
  geom_vline(xintercept = c(-0.147, 0.147), linetype = "dashed",
             color = "grey70", linewidth = 0.4) +
  geom_vline(xintercept = 0, color = "grey40", linewidth = 0.5) +
  geom_point(size = 2.8, alpha = 0.9) +
  scale_color_manual(values = col_models, name = NULL) +
  scale_x_continuous(limits = c(-0.9, 0.9), breaks = c(-0.6, -0.3, 0, 0.3, 0.6)) +
  labs(title    = "Variabili numeriche — Cliff's δ (P vs N)",
       subtitle = "Positivo = valore più alto nei condannati  |  |δ| > 0.15 = segnale rilevante",
       x = "Cliff's δ", y = NULL) +
  theme_paper() + theme(axis.text.y = element_text(size = 8.5))

compute_log2r <- function(df, feats, model_name) {
  feats_ok <- intersect(feats, names(df))
  do.call(bind_rows, lapply(feats_ok, function(feat) {
    v     <- suppressWarnings(as.numeric(as.character(df[[feat]])))
    v_bin <- ifelse(v %in% c(0, 1), v, NA_real_)
    pct_P <- mean(v_bin[df$label == 1] == 1, na.rm = TRUE)
    pct_N <- mean(v_bin[df$label == 0] == 1, na.rm = TRUE)
    log2r <- if (!is.na(pct_P) && !is.na(pct_N) && pct_P > 0 && pct_N > 0)
      log2(pct_P / pct_N) else NA_real_
    tibble(feature = feat, modello = model_name, log2r = log2r)
  }))
}

log2r_df <- bind_rows(
  compute_log2r(labeled_m1, FEAT_M1_BIN_EXT, "M1 (ex ante)"),
  compute_log2r(labeled_m2, c(FEAT_M1_BIN_EXT, FEAT_M2_EXTRA_BIN), "M2 (aggiudicazione)"),
  compute_log2r(labeled_m3, c(FEAT_M1_BIN_EXT, FEAT_M2_EXTRA_BIN, FEAT_M3_EXTRA_BIN),
                "M3 (esecuzione)")
) |> filter(!is.na(log2r)) |>
  mutate(modello = factor(modello, levels = names(col_models)))

feat_order_bin <- log2r_df |>
  group_by(feature) |> summarise(med = median(log2r), .groups = "drop") |>
  arrange(med) |> pull(feature)
log2r_df <- log2r_df |> mutate(feature = factor(feature, levels = feat_order_bin))

p02b <- ggplot(log2r_df, aes(x = log2r, y = feature, color = modello)) +
  geom_vline(xintercept = 0, color = "grey40", linewidth = 0.5) +
  geom_point(size = 2.8, alpha = 0.9) +
  scale_color_manual(values = col_models, name = NULL, guide = "none") +
  labs(title    = "Flag binari — log₂(P/N) delle proporzioni",
       subtitle = "log₂(P/N) = 0 → nessuna differenza",
       x = "log₂(P/N)", y = NULL) +
  theme_paper() + theme(axis.text.y = element_text(size = 8.5))

p02 <- (p02a | p02b) +
  plot_annotation(
    title    = "Feature signature: quali variabili separano condannati da assolti",
    subtitle = "Ordinamento per effect size — Color = fase temporale della feature",
    theme    = theme(plot.title = element_text(size = 14, face = "bold"))
  )

save_plot(p02, "02_feature_signature", w = 18, h = 10)

# Tasso condanne per regione + scatter vs contesto
# socioeconomico (omicidi, disoccupazione).
# Fonte: M1 labeled.
message("03 - Geografia della corruzione")

reg_stats <- labeled_m1 |>
  filter(!is.na(regione)) |>
  group_by(regione) |>
  summarise(n = n(), k = sum(label == 1), .groups = "drop") |>
  filter(n >= 10) |>
  rowwise() |>
  mutate(pct   = k / n * 100,
         ci_lo = wilson_ci(k, n)[1] * 100,
         ci_hi = wilson_ci(k, n)[2] * 100) |>
  ungroup()

pct_naz <- mean(labeled_m1$label == 1) * 100

p03a <- ggplot(reg_stats, aes(x = fct_reorder(regione, pct), y = pct)) +
  geom_hline(yintercept = pct_naz, linetype = "dashed",
             color = "grey40", linewidth = 0.7) +
  geom_col(aes(fill = pct), alpha = 0.9, width = 0.72) +
  geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi),
                width = 0.25, color = "grey30", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%.0f%% (n=%d)", pct, n)),
            hjust = -0.12, size = 2.8) +
  scale_fill_gradient2(low = "#2E86AB", mid = "#FFF3E0", high = "#C73E1D",
                       midpoint = pct_naz, guide = "none") +
  coord_flip() +
  scale_y_continuous(labels = function(x) paste0(x, "%"),
                     limits = c(0, max(reg_stats$ci_hi) * 1.3), expand = c(0, 0)) +
  labs(title    = "Tasso di condanne per regione",
       subtitle = sprintf("Barre + IC 95%% Wilson — media nazionale: %.1f%%", pct_naz),
       x = NULL, y = "% condannati") +
  theme_paper()

reg_eco <- m1 |>
  filter(!is.na(regione), !is.na(tasso_omicidi_100k), !is.na(tasso_disoccupazione)) |>
  group_by(regione) |>
  summarise(omicidi_med = median(tasso_omicidi_100k,   na.rm = TRUE),
            disoc_med   = median(tasso_disoccupazione, na.rm = TRUE),
            .groups = "drop")

reg_scatter <- reg_stats |> left_join(reg_eco, by = "regione") |> filter(!is.na(omicidi_med))

p03b <- ggplot(reg_scatter, aes(x = omicidi_med, y = pct, size = n, color = disoc_med)) +
  geom_smooth(method = "lm", se = TRUE, color = "grey60",
              linewidth = 0.7, alpha = 0.15, show.legend = FALSE) +
  geom_point(alpha = 0.85) +
  geom_text_repel(aes(label = regione), size = 2.8, color = "grey20",
                  max.overlaps = 25, show.legend = FALSE) +
  scale_color_gradient(low = "#FFF3E0", high = "#BF360C",
                       name = "Disoccupazione\n(mediana %)") +
  scale_size_continuous(range = c(2, 9), name = "N labeled", guide = "none") +
  labs(title    = "Tasso condanne regionale vs contesto territoriale",
       subtitle = "Dimensione punto = N labeled — colore = tasso disoccupazione mediano",
       x = "Tasso omicidi / 100k abitanti (mediano)", y = "% condannati") +
  theme_paper() + theme(legend.position = "right")

p03 <- (p03a | p03b) +
  plot_annotation(
    title   = "Geografia della corruzione negli appalti",
    caption = "Fonte: M1 nativi. Regioni con n < 10 escluse dal pannello sinistro.",
    theme   = theme(plot.title = element_text(size = 14, face = "bold"))
  )

save_plot(p03, "03_territorio_socioeconomico", w = 17, h = 9)

# Ridge plot + heatmap procedura × quartile importo.
# Fonte: M1 labeled.
message("04 - Importi e tipo procedura")

ridge_df <- labeled_m1 |>
  filter(!is.na(importo_lotto), importo_lotto > 0, !is.na(tipo_scelta_4cls))

p04a <- ggplot(ridge_df, aes(x = importo_lotto, y = tipo_scelta_4cls,
                              fill = label_fct, color = label_fct)) +
  geom_density_ridges(alpha = 0.55, scale = 0.9, rel_min_height = 0.01, bandwidth = 0.25) +
  scale_x_log10(labels = label_number(scale_cut = cut_short_scale()),
                breaks  = c(1e3, 1e4, 1e5, 1e6, 1e7, 1e8)) +
  scale_fill_manual(values  = c("Assolto (N)" = col_N, "Condannato (P)" = col_P), name = NULL) +
  scale_color_manual(values = c("Assolto (N)" = col_N, "Condannato (P)" = col_P), guide = "none") +
  labs(title = "Distribuzione importo lotto per procedura",
       subtitle = "Ridge plot — asse x in scala log",
       x = "Importo lotto (€, log)", y = NULL) +
  theme_paper()

heat_df <- labeled_m1 |>
  filter(!is.na(importo_lotto), importo_lotto > 0, !is.na(tipo_scelta_4cls)) |>
  mutate(q_imp = cut(importo_lotto,
                     breaks = quantile(importo_lotto, probs = c(0, .25, .5, .75, 1), na.rm = TRUE),
                     labels = c("Q1 (piccoli)", "Q2", "Q3", "Q4 (grandi)"),
                     include.lowest = TRUE)) |>
  filter(!is.na(q_imp)) |>
  group_by(tipo_scelta_4cls, q_imp) |>
  summarise(n = n(), pct_P = mean(label == 1) * 100, .groups = "drop")

p04b <- ggplot(heat_df, aes(x = tipo_scelta_4cls, y = q_imp, fill = pct_P)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.0f%%\n(n=%d)", pct_P, n)), size = 3.2, color = "grey15") +
  scale_fill_gradient(low = "#E3F2FD", high = "#B71C1C",
                      name = "% condannati", labels = function(x) paste0(x, "%")) +
  scale_y_discrete(limits = c("Q4 (grandi)", "Q3", "Q2", "Q1 (piccoli)")) +
  labs(title = "Tasso condanne: procedura × quartile d'importo",
       subtitle = "Dove cresce il rischio fuori dalla diagonale attesa?",
       x = "Tipo procedura", y = "Quartile importo") +
  theme_paper() +
  theme(axis.text.x = element_text(angle = 15, hjust = 1), panel.grid = element_blank())

p04 <- (p04a | p04b) +
  plot_annotation(
    title   = "Dove si concentra il rischio: importi e procedura di scelta",
    caption = "Fonte: M1 labeled. Quartili calcolati sull'intero dataset labeled M1.",
    theme   = theme(plot.title = element_text(size = 14, face = "bold"))
  )

save_plot(p04, "04_importi_procedura", w = 16, h = 8)

# Scatter ribasso vs offerenti con densità marginali.
# Cover bidding: molti offerenti, spread compresso.
# Fonte: M2 labeled (campione 1.5k).
message("05 - Dinamica competitiva M2")

rigg_df <- labeled_m2 |>
  filter(!is.na(num_imprese_offerenti), !is.na(ribasso_aggiudicazione),
         num_imprese_offerenti >= 1, num_imprese_offerenti <= 50,
         ribasso_aggiudicazione > -10, ribasso_aggiudicazione < 110)
rigg_df <- rigg_df |> slice_sample(n = min(1500L, nrow(rigg_df)))
n_rg    <- nrow(rigg_df)

p05_main <- ggplot(rigg_df,
                   aes(x = num_imprese_offerenti, y = ribasso_aggiudicazione,
                       color = label_fct)) +
  geom_jitter(alpha = 0.6, size = 1.8, width = 0.3, height = 0) +
  geom_smooth(method = "loess", se = TRUE, linewidth = 1.1, alpha = 0.18, span = 0.7) +
  scale_color_manual(values = c("Assolto (N)" = col_N, "Condannato (P)" = col_P), name = NULL) +
  labs(title    = "Ribasso vs numero di imprese offerenti (M2)",
       subtitle = sprintf(
         "Cartello tipico: molte offerte ma ribassi bassi — monopolio: poche offerte, ribasso basso\n(n=%d, campione M2 labeled)", n_rg),
       x = "N° imprese offerenti", y = "Ribasso aggiudicazione (%)") +
  theme_paper() + theme(legend.position = "top")

p05_top <- ggplot(rigg_df, aes(x = num_imprese_offerenti, fill = label_fct)) +
  geom_density(alpha = 0.55, linewidth = 0.3) +
  scale_fill_manual(values = c("Assolto (N)" = col_N, "Condannato (P)" = col_P), guide = "none") +
  theme_void() + theme(plot.background = element_rect(fill = "white", color = NA))

p05_right <- ggplot(rigg_df, aes(x = ribasso_aggiudicazione, fill = label_fct)) +
  geom_density(alpha = 0.55, linewidth = 0.3) +
  scale_fill_manual(values = c("Assolto (N)" = col_N, "Condannato (P)" = col_P), guide = "none") +
  coord_flip() +
  theme_void() + theme(plot.background = element_rect(fill = "white", color = NA))

p05 <- (p05_top + plot_spacer()) / (p05_main + p05_right) +
  plot_layout(heights = c(1, 3.5), widths = c(3.5, 1)) +
  plot_annotation(
    title   = "Dinamica competitiva dei bandi aggiudicati (M2)",
    caption = sprintf("Scatter + densità marginali — campione M2 labeled (n=%d)", n_rg),
    theme   = theme(plot.title = element_text(size = 14, face = "bold"))
  )

save_plot(p05, "05_competizione_M2", w = 13, h = 10)

# Severity score (0-6) e frequenza singole anomalie.
# Fonte: M3 labeled.
message("06 - Anomalie M3")

labeled_m3_an <- labeled_m3 |>
  mutate(
    an_variante    = as.integer(
      replace_na(flag_variante_sostanziale == 1, FALSE) |
      replace_na(flag_variante_oltre_termine == 1, FALSE)),
    an_sospensione = as.integer(replace_na(flag_sospensione    == 1, FALSE)),
    an_proroga     = as.integer(replace_na(flag_proroga        == 1, FALSE)),
    an_subappalto  = as.integer(replace_na(flag_subappalto     == 1, FALSE)),
    an_ritardo     = as.integer(replace_na(flag_in_ritardo     == 1, FALSE)),
    an_giudiziaria = as.integer(replace_na(flag_sosp_giudiziaria == 1, FALSE)),
    severity       = an_variante + an_sospensione + an_proroga +
                     an_subappalto + an_ritardo + an_giudiziaria
  )

n_m3 <- nrow(labeled_m3_an)

sev_df <- labeled_m3_an |>
  count(label_fct, severity) |>
  group_by(label_fct) |>
  mutate(pct = n / sum(n) * 100) |>
  ungroup()

p06a <- ggplot(sev_df, aes(x = severity, y = pct, fill = label_fct)) +
  geom_col(position = "dodge", alpha = 0.9, width = 0.72) +
  geom_text(aes(label = sprintf("%.0f%%", pct)),
            position = position_dodge(0.72), vjust = -0.35, size = 3) +
  scale_lab +
  scale_x_continuous(breaks = 0:6) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12)),
                     labels = function(x) paste0(x, "%")) +
  labs(title    = "Severity score: quante anomalie esecutive accumula un appalto?",
       subtitle = "Somma di: variante (sost. o fuori termine), sospensione, proroga, subappalto, ritardo SAL, sospensione giudiziaria",
       x        = "Severity score (0 = nessuna anomalia, 6 = tutte)",
       y        = "% appalti M3") +
  theme_paper()

anom_long <- labeled_m3_an |>
  group_by(label_fct) |>
  summarise(
    "Subappalto"               = mean(an_subappalto  == 1) * 100,
    "Variante (sost./fuori term.)" = mean(an_variante == 1) * 100,
    "Sospensione"              = mean(an_sospensione == 1) * 100,
    "Ritardo SAL"              = mean(an_ritardo     == 1) * 100,
    "Proroga"                  = mean(an_proroga     == 1) * 100,
    "Sospens. giudiziaria"     = mean(an_giudiziaria == 1) * 100,
    .groups = "drop"
  ) |>
  pivot_longer(-label_fct, names_to = "anomalia", values_to = "pct")

anom_order <- anom_long |>
  filter(label_fct == "Condannato (P)") |> arrange(pct) |> pull(anomalia)
anom_long <- anom_long |> mutate(anomalia = factor(anomalia, levels = anom_order))

p06b <- ggplot(anom_long, aes(x = pct, y = anomalia, fill = label_fct)) +
  geom_col(position = "dodge", alpha = 0.9, width = 0.65) +
  geom_text(aes(label = sprintf("%.1f%%", pct)),
            position = position_dodge(0.65), hjust = -0.1, size = 3) +
  scale_lab +
  scale_x_continuous(labels = function(x) paste0(x, "%"),
                     limits = c(0, max(anom_long$pct) * 1.25), expand = c(0, 0)) +
  labs(title = "Frequenza delle singole anomalie",
       x = "% appalti M3 con anomalia attiva", y = NULL) +
  theme_paper() + theme(legend.position = "none")

p06 <- p06a / p06b +
  plot_annotation(
    title   = sprintf("Anomalie di esecuzione si accumulano sui condannati (M3, n=%d)", n_m3),
    caption = "Fonte: M3 nativi.",
    theme   = theme(plot.title = element_text(size = 14, face = "bold"))
  )

save_plot(p06, "06_anomalie_M3", w = 14, h = 12)

# Dumbbell % NA per feature (P vs N) — test di MNAR.
# Fonte: M3 labeled (tutte le feature disponibili).
message("07 - Missingness come segnale")

all_feats <- unique(c(FEAT_M1_NUM, FEAT_M2_EXTRA_NUM, FEAT_M3_EXTRA_NUM,
                       FEAT_M1_BIN_EXT, FEAT_M2_EXTRA_BIN, FEAT_M3_EXTRA_BIN))

miss_p <- labeled_m3 |> filter(label == 1) |>
  summarise(across(all_of(intersect(all_feats, names(labeled_m3))),
                   ~ mean(is.na(.)) * 100)) |>
  pivot_longer(everything(), names_to = "feature", values_to = "pct_P")

miss_n <- labeled_m3 |> filter(label == 0) |>
  summarise(across(all_of(intersect(all_feats, names(labeled_m3))),
                   ~ mean(is.na(.)) * 100)) |>
  pivot_longer(everything(), names_to = "feature", values_to = "pct_N")

miss_df <- left_join(miss_p, miss_n, by = "feature") |>
  mutate(delta = pct_P - pct_N) |>
  filter(!is.na(pct_P), !is.na(pct_N), pct_P > 0 | pct_N > 0) |>
  mutate(feature = fct_reorder(feature, abs(delta)))

miss_long <- miss_df |>
  pivot_longer(c(pct_P, pct_N), names_to = "gruppo", values_to = "pct") |>
  mutate(gruppo = factor(ifelse(gruppo == "pct_P", "Condannato (P)", "Assolto (N)"),
                          levels = c("Assolto (N)", "Condannato (P)")))

p07 <- ggplot() +
  geom_segment(data = miss_df,
               aes(x = pct_N, xend = pct_P, y = feature, yend = feature),
               color = "grey70", linewidth = 0.5) +
  geom_point(data = miss_long,
             aes(x = pct, y = feature, color = gruppo), size = 3, alpha = 0.9) +
  geom_text(data = miss_df,
            aes(x = pmax(pct_P, pct_N), y = feature,
                label = sprintf("Δ = %.1f pp", delta)),
            hjust = -0.15, size = 2.7, color = "grey30") +
  scale_color_manual(values = c("Assolto (N)" = col_N, "Condannato (P)" = col_P), name = NULL) +
  scale_x_continuous(labels = function(x) paste0(x, "%"),
                     expand = expansion(mult = c(0, 0.25))) +
  labs(title    = "Il missingness stesso è informativo: % NA per feature (P vs N)",
       subtitle = "Dumbbell plot — top feature ordinate per |Δ|",
       x = "% valori mancanti", y = NULL,
       caption  = "Δ = 0 ⇒ la feature manca allo stesso modo in P e N. Feature M3 (massima copertura).") +
  theme_paper() + theme(axis.text.y = element_text(size = 8.5))

save_plot(p07, "07_missingness_signal", w = 14, h = 10)

# % condannati nel tempo (Wilson CI) + mix procedurale annuale.
# Fonte: M1.
message("08 - Evoluzione temporale")

time_lab <- labeled_m1 |>
  filter(!is.na(anno_pubblicazione),
         anno_pubblicazione >= 2008, anno_pubblicazione <= 2024) |>
  group_by(anno_pubblicazione) |>
  summarise(n = n(), k = sum(label == 1), .groups = "drop") |>
  rowwise() |>
  mutate(pct   = k / n * 100,
         ci_lo = wilson_ci(k, n)[1] * 100,
         ci_hi = wilson_ci(k, n)[2] * 100) |>
  ungroup()

p08a <- ggplot(time_lab, aes(x = anno_pubblicazione)) +
  geom_ribbon(aes(ymin = ci_lo, ymax = ci_hi), fill = col_P, alpha = 0.18) +
  geom_line(aes(y = pct), color = col_P, linewidth = 1.2) +
  geom_point(aes(y = pct, size = n), color = col_P, alpha = 0.85) +
  geom_vline(xintercept = 2016.5, linetype = "dashed", color = "grey40", linewidth = 0.7) +
  annotate("label", x = 2016.5, y = Inf,
           label = "D.Lgs. 50/2016", vjust = 1.3, size = 3,
           fill = "white", color = "grey30", label.size = 0.3) +
  scale_size_continuous(range = c(2, 9), name = "N labeled") +
  scale_x_continuous(breaks = seq(2008, 2024, 2)) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(title = "% condannati per anno di pubblicazione bando",
       subtitle = "Bande + IC 95% Wilson — dimensione punto = N labeled",
       x = "Anno pubblicazione bando", y = "% condannati") +
  theme_paper() + theme(legend.position = "right")

proc_time <- m1 |>
  filter(!is.na(anno_pubblicazione), !is.na(tipo_scelta_4cls),
         anno_pubblicazione >= 2008, anno_pubblicazione <= 2024) |>
  count(anno_pubblicazione, tipo_scelta_4cls) |>
  group_by(anno_pubblicazione) |>
  mutate(pct = n / sum(n) * 100) |>
  ungroup()

p08b <- ggplot(proc_time, aes(x = anno_pubblicazione, y = pct, fill = tipo_scelta_4cls)) +
  geom_area(alpha = 0.88, position = "stack") +
  scale_fill_brewer(palette = "Set2", name = "Procedura") +
  scale_x_continuous(breaks = seq(2008, 2024, 2)) +
  scale_y_continuous(labels = function(x) paste0(x, "%"), expand = c(0, 0)) +
  labs(title = "Mix procedurale nel dataset completo",
       subtitle = "Affidamento diretto domina dopo la riforma del 2016",
       x = "Anno pubblicazione", y = "% appalti") +
  theme_paper()

p08 <- p08a / p08b +
  plot_annotation(
    title   = "Evoluzione temporale: tasso condanne e mix procedurale",
    caption = "Fonte: M1 nativi.",
    theme   = theme(plot.title = element_text(size = 14, face = "bold"))
  )

save_plot(p08, "08_timeline", w = 14, h = 11)

# flag_delega (91.7% MISSING) e importo_complessivo_gara
# distribuito per fase — variabili a rischio di leakage.
# Fonte: M1 labeled (flag_delega); M1/M2/M3 labeled (importo).
message("09 - Leakage watch")

delega_df <- labeled_m1 |>
  mutate(cat_delega = case_when(
    is.na(flag_delega) | as.character(flag_delega) == "MISSING" ~ "MISSING",
    as.character(flag_delega) == "1" ~ "Delegati",
    TRUE ~ "Non-delegati"
  )) |>
  count(cat_delega, label_fct) |>
  group_by(cat_delega) |>
  mutate(pct = n / sum(n) * 100, tot = sum(n)) |>
  ungroup() |>
  mutate(cat_delega = factor(cat_delega, levels = c("Delegati", "MISSING", "Non-delegati")))

p09a <- ggplot(delega_df, aes(x = cat_delega, y = pct, fill = label_fct)) +
  geom_col(alpha = 0.9, width = 0.6) +
  geom_text(aes(label = sprintf("%.0f%%", pct)),
            position = position_stack(vjust = 0.5),
            color = "white", fontface = "bold", size = 4) +
  geom_text(data = delega_df |> distinct(cat_delega, tot),
            aes(x = cat_delega, y = 103,
                label = sprintf("n=%s", format(tot, big.mark = "'"))),
            inherit.aes = FALSE, size = 3.2, color = "grey30") +
  scale_lab +
  scale_y_continuous(labels = function(x) paste0(x, "%"),
                     expand = expansion(mult = c(0, 0.08))) +
  labs(title    = "flag_delega — composizione P/N per categoria",
       subtitle = "91.7% della colonna è MISSING: le 3 categorie hanno tasso P diverso?",
       x = NULL, y = "% appalti") +
  theme_paper()

cigs_m2 <- labeled_m2$cig
cigs_m3 <- labeled_m3$cig

imp_df <- bind_rows(
  labeled_m1 |>
    filter(!is.na(importo_complessivo_gara), importo_complessivo_gara > 0,
           !(cig %in% cigs_m2)) |>
    mutate(fase = "Solo M1 (nessuna agg.)"),
  labeled_m2 |>
    filter(!is.na(importo_complessivo_gara), importo_complessivo_gara > 0,
           !(cig %in% cigs_m3)) |>
    mutate(fase = "M2 (solo agg.)"),
  labeled_m3 |>
    filter(!is.na(importo_complessivo_gara), importo_complessivo_gara > 0) |>
    mutate(fase = "M3 (esecuzione)")
) |> mutate(fase = factor(fase, levels = c("Solo M1 (nessuna agg.)", "M2 (solo agg.)", "M3 (esecuzione)")))

p09b <- ggplot(imp_df, aes(x = importo_complessivo_gara, fill = label_fct, color = label_fct)) +
  geom_density(alpha = 0.45, linewidth = 0.4) +
  facet_wrap(~ fase, ncol = 1, scales = "free_y") +
  scale_x_log10(labels = label_number(scale_cut = cut_short_scale()),
                breaks  = c(1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9)) +
  scale_fill_manual(values  = c("Assolto (N)" = col_N, "Condannato (P)" = col_P), name = NULL) +
  scale_color_manual(values = c("Assolto (N)" = col_N, "Condannato (P)" = col_P), guide = "none") +
  labs(title    = "importo_complessivo_gara — distribuzione per fase",
       subtitle = "Se le distribuzioni P/N si separano di più in M3, è sospetto di leakage ex-cost",
       x = "Importo complessivo gara (€, log)", y = "Densità") +
  theme_paper() + theme(legend.position = "top")

p09 <- (p09a | p09b) +
  plot_annotation(
    title   = "Leakage watch: le due feature dominanti nel guardrail",
    caption = "flag_delega (91.7% MISSING) e importo_complessivo_gara (dominante in 3/4 fold)",
    theme   = theme(plot.title = element_text(size = 14, face = "bold"))
  )

save_plot(p09, "09_leakage_watch", w = 16, h = 10)

# Deviazioni da Benford = importi "costruiti" (Nigrini 2012).
# CAVEAT: P ha ~781 osservazioni — MAD con CI ampi.
# Fonte: M1.
message("10 - Benford's Law")

benford_atteso <- tibble(cifra = 1:9, pct_atteso = log10(1 + 1 / (1:9)) * 100)

u_vals <- m1 |>
  filter(is.na(label), !is.na(importo_lotto), importo_lotto > 0) |>
  slice_sample(n = 2e6) |> pull(importo_lotto)

bench_df <- bind_rows(
  tibble(gruppo = "Assolto (N)",
         cifra  = prima_cifra(labeled_m1$importo_lotto[labeled_m1$label == 0])),
  tibble(gruppo = "Condannato (P)",
         cifra  = prima_cifra(labeled_m1$importo_lotto[labeled_m1$label == 1])),
  tibble(gruppo = "Unlabeled (campione 2M)", cifra = prima_cifra(u_vals))
) |>
  count(gruppo, cifra) |> filter(cifra %in% 1:9) |>
  group_by(gruppo) |>
  mutate(pct = n / sum(n) * 100, n_gruppo = sum(n)) |>
  ungroup() |>
  left_join(benford_atteso, by = "cifra") |>
  mutate(deviazione = pct - pct_atteso)

mad_tab <- bench_df |>
  group_by(gruppo, n_gruppo) |>
  summarise(MAD = mean(abs(deviazione), na.rm = TRUE) / 100, .groups = "drop") |>
  mutate(
    giudizio  = case_when(MAD < 0.006 ~ "conforme", MAD < 0.012 ~ "accettabile",
                          MAD < 0.015 ~ "marginalmente accett.", TRUE ~ "non conforme"),
    etichetta = sprintf("%s\nn=%s — MAD=%.4f (%s)",
                        gruppo, format(n_gruppo, big.mark = "'"), MAD, giudizio)
  )

etic     <- setNames(mad_tab$etichetta, mad_tab$gruppo)
bench_df <- bench_df |> mutate(faceta = factor(etic[gruppo], levels = etic))

p10 <- ggplot(bench_df, aes(x = factor(cifra))) +
  geom_col(aes(y = pct, fill = gruppo), alpha = 0.85, width = 0.75) +
  geom_line(data = benford_atteso,
            aes(x = factor(cifra), y = pct_atteso, group = 1),
            color = "grey20", linewidth = 0.8, linetype = "dashed") +
  geom_point(data = benford_atteso, aes(x = factor(cifra), y = pct_atteso),
             color = "grey20", size = 2) +
  facet_wrap(~ faceta, nrow = 1) +
  scale_fill_manual(values = c("Assolto (N)" = col_N, "Condannato (P)" = col_P,
                                "Unlabeled (campione 2M)" = col_U), guide = "none") +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(title    = "Legge di Benford sulla prima cifra di importo_lotto",
       subtitle = "Barre = osservato — linea tratteggiata = atteso log10(1+1/d) — MAD Nigrini: <0.006 conforme, >0.015 non conforme",
       x = "Prima cifra di importo_lotto", y = "% osservazioni",
       caption  = "CAVEAT: P ha ~781 osservazioni — i MAD hanno CI ampi. Fonte: M1 nativi.") +
  theme_paper()

save_plot(p10, "10_benford_law", w = 15, h = 6)
cat("MAD Benford per gruppo:\n"); print(mad_tab)

# Bunching estimator (Chetty et al. 2011) applicato alle
# soglie 40k (pre-2020) e 150k (post-2020).
# Fonte: M1.
message("11 - Threshold bunching")

imp_df <- m1 |>
  filter(!is.na(importo_lotto), importo_lotto > 1e3, importo_lotto < 2e6,
         !is.na(anno_pubblicazione)) |>
  mutate(periodo = case_when(
    anno_pubblicazione <= 2019 ~ "Pre-2020 (soglia 40k)",
    anno_pubblicazione >= 2021 ~ "Post-2020 (soglia 150k)",
    TRUE ~ "2020 (transizione)"
  )) |>
  filter(periodo != "2020 (transizione)")

p11a <- ggplot(imp_df |> filter(importo_lotto < 1e5), aes(x = importo_lotto)) +
  geom_histogram(binwidth = 1000, fill = "#5E81AC", color = "white",
                 linewidth = 0.1, alpha = 0.92) +
  geom_vline(xintercept = 40000, linetype = "dashed", color = "#BF616A", linewidth = 0.8) +
  annotate("label", x = 40000, y = Inf,
           label = "40k — soglia\naffidamento diretto",
           vjust = 1.2, size = 3, fill = "white", color = "#BF616A",
           label.size = 0.3, fontface = "bold") +
  facet_wrap(~ periodo, nrow = 1, scales = "free_y") +
  scale_x_continuous(labels = function(x) paste0(x / 1000, "k"),
                     breaks = c(10000, 20000, 30000, 40000, 50000, 60000, 80000, 100000)) +
  labs(title = "Bunching intorno alla soglia dei 40k EUR",
       subtitle = "Bin = 1k EUR — spike appena sotto 40k è segnale di splitting artificioso (solo pre-2020)",
       x = "Importo lotto (EUR)", y = "N. appalti") +
  theme_paper()

p11b <- ggplot(imp_df |> filter(importo_lotto > 5e4, importo_lotto < 3e5),
               aes(x = importo_lotto)) +
  geom_histogram(binwidth = 2000, fill = "#A3BE8C", color = "white",
                 linewidth = 0.1, alpha = 0.92) +
  geom_vline(xintercept = 150000, linetype = "dashed", color = "#BF616A", linewidth = 0.8) +
  annotate("label", x = 150000, y = Inf,
           label = "150k — nuova\nsoglia diretto",
           vjust = 1.2, size = 3, fill = "white", color = "#BF616A",
           label.size = 0.3, fontface = "bold") +
  facet_wrap(~ periodo, nrow = 1, scales = "free_y") +
  scale_x_continuous(labels = function(x) paste0(x / 1000, "k"),
                     breaks = c(75000, 100000, 125000, 150000, 175000, 200000, 250000)) +
  labs(title = "Bunching intorno alla soglia dei 150k EUR",
       subtitle = "Bin = 2k EUR — atteso: spike appena sotto 150k visibile solo post-2020",
       x = "Importo lotto (EUR)", y = "N. appalti") +
  theme_paper()

calc_ratio_bunch <- function(data, soglia, window = 0.1) {
  lo    <- soglia * (1 - window); hi <- soglia * (1 + window)
  sotto <- sum(data$importo_lotto > lo  & data$importo_lotto <= soglia)
  sopra <- sum(data$importo_lotto > soglia & data$importo_lotto <= hi)
  sprintf("%.2fx (sotto %d / sopra %d)", sotto / max(sopra, 1), sotto, sopra)
}

ratio_40  <- imp_df |> filter(periodo == "Pre-2020 (soglia 40k)")   |> calc_ratio_bunch(40000)
ratio_150 <- imp_df |> filter(periodo == "Post-2020 (soglia 150k)") |> calc_ratio_bunch(150000)
cat(sprintf("Bunching ratio ±10%% soglia 40k  (pre-2020):  %s\n", ratio_40))
cat(sprintf("Bunching ratio ±10%% soglia 150k (post-2020): %s\n", ratio_150))

p11 <- p11a / p11b +
  plot_annotation(
    title    = "Threshold bunching: frazionamento artificioso sotto le soglie?",
    subtitle = sprintf("Ratio 10%% ±soglia — 40k: %s — 150k: %s", ratio_40, ratio_150),
    caption  = "Codice Appalti: D.Lgs. 50/2016 soglia 40k — L.120/2020 e D.Lgs. 36/2023 soglia 150k (lavori). Fonte: M1 nativi.",
    theme    = theme(plot.title = element_text(size = 15, face = "bold"),
                     plot.subtitle = element_text(color = "grey30"))
  )

save_plot(p11, "11_threshold_bunching", w = 15, h = 10)

# 5 componenti binari: single bid, non-open, short window,
# urgency, low ribasso. Fonte: M2.
message("12 - Corruption Risk Indicators")

med_finestra <- m2 |>
  group_by(cpv_macro_categoria) |>
  summarise(med_fin = median(finestra_offerta_giorni, na.rm = TRUE), .groups = "drop")

q25_ribasso <- m2 |>
  group_by(cpv_macro_categoria) |>
  summarise(q25 = quantile(ribasso_aggiudicazione, 0.25, na.rm = TRUE), .groups = "drop")

cri_m2 <- m2 |>
  left_join(med_finestra, by = "cpv_macro_categoria") |>
  left_join(q25_ribasso,  by = "cpv_macro_categoria") |>
  mutate(
    CRI_single     = as.integer(replace_na(num_imprese_offerenti <= 1,        FALSE)),
    CRI_nonopen    = as.integer(replace_na(tipo_scelta_4cls != "APERTA",      FALSE)),
    CRI_short      = as.integer(replace_na(finestra_offerta_giorni < med_fin, FALSE)),
    CRI_urgenza    = as.integer(replace_na(flag_urgenza == 1,                 FALSE)),
    CRI_lowribasso = as.integer(replace_na(ribasso_aggiudicazione < q25,      FALSE)),
    CRI_score      = CRI_single + CRI_nonopen + CRI_short + CRI_urgenza + CRI_lowribasso
  )

cri_lab <- cri_m2 |>
  filter(!is.na(label)) |>
  mutate(label_fct = factor(label, levels = c(0, 1),
                             labels = c("Assolto (N)", "Condannato (P)")))

p12a <- cri_lab |>
  count(label_fct, CRI_score) |>
  group_by(label_fct) |>
  mutate(pct = n / sum(n) * 100) |>
  ggplot(aes(x = CRI_score, y = pct, fill = label_fct)) +
  geom_col(position = "dodge", alpha = 0.92) +
  geom_text(aes(label = sprintf("%.0f%%", pct)),
            position = position_dodge(0.9), vjust = -0.4, size = 2.8) +
  scale_lab +
  scale_x_continuous(breaks = 0:5) +
  scale_y_continuous(labels = function(x) paste0(x, "%"),
                     expand = expansion(mult = c(0, 0.1))) +
  labs(title    = "Distribuzione CRI score (Fazekas 2016)",
       subtitle = "5 componenti binari: single bid — non-open — short window — urgency — low ribasso",
       x = "CRI score (0 = nessun red flag, 5 = tutti)", y = "% appalti M2") +
  theme_paper()

cri_geo <- cri_m2 |>
  filter(!is.na(regione), !is.na(cpv_macro_categoria)) |>
  group_by(regione, cpv_macro_categoria) |>
  summarise(cri_mean = mean(CRI_score, na.rm = TRUE), n = n(), .groups = "drop") |>
  filter(n >= 200)

p12b <- ggplot(cri_geo,
               aes(x = cpv_macro_categoria,
                   y = fct_reorder(regione, cri_mean, .fun = mean),
                   fill = cri_mean)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%.2f", cri_mean)), size = 2.7, color = "grey15") +
  scale_fill_gradient(low = "#E3F2FD", high = "#B71C1C", name = "CRI medio") +
  labs(title    = "CRI medio per regione × categoria CPV (M2 intero)",
       subtitle = "Celle con n ≥ 200 — rosso = più red flag procedurali",
       x = "Categoria CPV", y = NULL) +
  theme_paper() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1), panel.grid = element_blank())

p12 <- p12a + p12b + plot_layout(widths = c(1, 1.3)) +
  plot_annotation(
    title    = "Corruption Risk Indicators (Fazekas, Toth & King 2016)",
    subtitle = "Indice composito ex-ante/procedurale — calcolato su M2 (appalti aggiudicati)",
    caption  = "Fonte: M2 nativi. CRI_nonopen e CRI_urgenza: NA trattati come 0 (assenza red flag).",
    theme    = theme(plot.title = element_text(size = 15, face = "bold"),
                     plot.subtitle = element_text(color = "grey30"))
  )

save_plot(p12, "12_CRI_fazekas", w = 16, h = 9)

# NOTA: ribasso_spread ha ~88.8% missing nel dataset totale.
# Fonte: M2 labeled.
message("13 - Bid-rigging indicators")

rigging <- labeled_m2 |>
  filter(!is.na(num_imprese_offerenti), !is.na(ribasso_aggiudicazione),
         !is.na(ribasso_spread),
         num_imprese_offerenti >= 2, num_imprese_offerenti <= 30,
         ribasso_spread >= 0, ribasso_spread < 50,
         ribasso_aggiudicazione > -10, ribasso_aggiudicazione < 80)

n_riggings <- nrow(rigging); n_P_rigg <- sum(rigging$label == 1); n_N_rigg <- sum(rigging$label == 0)

rigging <- rigging |>
  mutate(zona = case_when(
    num_imprese_offerenti <= 2 & ribasso_aggiudicazione < 5 ~
      "Monopolio (<=2 offerenti, ribasso basso)",
    num_imprese_offerenti >= 5 & ribasso_spread < 2 ~
      "Cover bidding (>=5 offerenti, spread<2)",
    num_imprese_offerenti >= 5 & ribasso_spread >= 2 ~
      "Competitiva (>=5 offerenti, spread>=2)",
    TRUE ~ "Intermedia"
  ))

zona_stats <- rigging |>
  group_by(zona) |>
  summarise(n = n(), pct_P = mean(label == 1) * 100, .groups = "drop") |>
  mutate(zona = factor(zona, levels = c("Competitiva (>=5 offerenti, spread>=2)",
                                         "Intermedia",
                                         "Cover bidding (>=5 offerenti, spread<2)",
                                         "Monopolio (<=2 offerenti, ribasso basso)")))

pct_P_m2 <- mean(labeled_m2$label == 1) * 100

p13a <- ggplot(rigging, aes(x = num_imprese_offerenti, y = ribasso_spread, color = label_fct)) +
  geom_jitter(alpha = 0.55, size = 1.6, width = 0.25) +
  annotate("rect", xmin = 5, xmax = 30, ymin = 0, ymax = 2,
           fill = "#C73E1D", alpha = 0.08) +
  annotate("text", x = 20, y = 1,
           label = "Zona cover-bidding\n(molti offerenti, spread compresso)",
           size = 3, color = "#C73E1D", fontface = "bold") +
  scale_color_manual(values = c("Assolto (N)" = col_N, "Condannato (P)" = col_P), name = NULL) +
  labs(title    = "Compressione ribassi vs numero offerenti",
       subtitle = sprintf("n = %d appalti M2 con ribasso_spread disponibile (%dP, %dN) — zona rossa = cover bidding OECD",
                          n_riggings, n_P_rigg, n_N_rigg),
       x = "N. imprese offerenti", y = "Spread ribassi (max - aggiudicatario)") +
  theme_paper()

p13b <- ggplot(zona_stats, aes(x = fct_reorder(zona, pct_P), y = pct_P)) +
  geom_col(aes(fill = pct_P), alpha = 0.92, width = 0.7) +
  geom_hline(yintercept = pct_P_m2, linetype = "dashed", color = "grey40", linewidth = 0.7) +
  geom_text(aes(label = sprintf("%.1f%%\n(n=%d)", pct_P, n)), hjust = -0.1, size = 3) +
  scale_fill_gradient(low = "#FFF3E0", high = "#B71C1C", guide = "none") +
  coord_flip() +
  scale_y_continuous(labels = function(x) paste0(x, "%"),
                     limits = c(0, max(zona_stats$pct_P) * 1.3), expand = c(0, 0)) +
  labs(title    = "% condannati per zona di rischio",
       subtitle = sprintf("Zone OECD/Imhof  |  ancoraggio M2: %.1f%% (linea tratteggiata)", pct_P_m2),
       x = NULL, y = "% condannati") +
  theme_paper()

p13 <- p13a / p13b + plot_layout(heights = c(1.3, 1)) +
  plot_annotation(
    title   = "Indicatori di bid-rigging (OECD 2009 — Imhof et al. 2018)",
    caption = "NOTA: ribasso_spread ha ~88.8% missing nel dataset totale — campione ridotto. Fonte: M2 nativi.",
    theme   = theme(plot.title = element_text(size = 15, face = "bold"))
  )

save_plot(p13, "13_bid_rigging", w = 13, h = 11)
cat("\nZone bid-rigging:\n"); print(zona_stats)

# Tasso condanne per CPV, mix procedurale SANITA vs altri,
# serie storica urgenza e negoziate.
# Fonte: M1.
message("14 - Settore SANITARIO")

cpv_stats <- labeled_m1 |>
  filter(!is.na(cpv_macro_categoria)) |>
  group_by(cpv_macro_categoria) |>
  summarise(n = n(), k = sum(label == 1), .groups = "drop") |>
  rowwise() |>
  mutate(pct   = k / n * 100,
         ci_lo = wilson_ci(k, n)[1] * 100,
         ci_hi = wilson_ci(k, n)[2] * 100,
         is_san = cpv_macro_categoria == "SANITA") |>
  ungroup()

p14a <- ggplot(cpv_stats, aes(x = fct_reorder(cpv_macro_categoria, pct), y = pct, fill = is_san)) +
  geom_col(alpha = 0.92, width = 0.7) +
  geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.2, color = "grey30", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%.0f%% (n=%d)", pct, n)), hjust = -0.1, size = 3) +
  scale_fill_manual(values = c("TRUE" = "#C73E1D", "FALSE" = "#90A4AE"), guide = "none") +
  coord_flip() +
  scale_y_continuous(labels = function(x) paste0(x, "%"),
                     limits = c(0, max(cpv_stats$ci_hi) * 1.25), expand = c(0, 0)) +
  labs(title = "Tasso condanne per categoria CPV",
       subtitle = "IC 95% Wilson — SANITA evidenziato in rosso",
       x = NULL, y = "% condannati") +
  theme_paper()

proc_mix <- m1 |>
  filter(!is.na(cpv_macro_categoria), !is.na(tipo_scelta_4cls)) |>
  mutate(settore = ifelse(cpv_macro_categoria == "SANITA", "SANITA", "Altri settori")) |>
  count(settore, tipo_scelta_4cls) |>
  group_by(settore) |>
  mutate(pct = n / sum(n) * 100, tot = sum(n)) |>
  ungroup()

p14b <- ggplot(proc_mix, aes(x = settore, y = pct, fill = tipo_scelta_4cls)) +
  geom_col(alpha = 0.9, width = 0.65) +
  geom_text(aes(label = sprintf("%.0f%%", pct)),
            position = position_stack(vjust = 0.5), color = "white", fontface = "bold", size = 3.5) +
  geom_text(data = proc_mix |> distinct(settore, tot),
            aes(x = settore, y = 102, label = sprintf("n=%s", format(tot, big.mark = "'"))),
            inherit.aes = FALSE, size = 3, color = "grey30") +
  scale_fill_brewer(palette = "Set2", name = "Procedura") +
  scale_y_continuous(labels = function(x) paste0(x, "%"),
                     expand = expansion(mult = c(0, 0.1))) +
  labs(title = "Mix procedurale: SANITA vs altri settori",
       subtitle = "M1 intero — SANITA fa più ricorso a procedure non-aperte?",
       x = NULL, y = NULL) +
  theme_paper()

urg_tempo <- m1 |>
  filter(!is.na(cpv_macro_categoria), !is.na(anno_pubblicazione),
         anno_pubblicazione >= 2012, anno_pubblicazione <= 2024) |>
  mutate(settore = ifelse(cpv_macro_categoria == "SANITA", "SANITA", "Altri settori")) |>
  group_by(settore, anno_pubblicazione) |>
  summarise(pct_urg = mean(flag_urgenza == 1,               na.rm = TRUE) * 100,
            pct_neg = mean(tipo_scelta_4cls == "NEGOZIATA", na.rm = TRUE) * 100,
            n = n(), .groups = "drop") |>
  pivot_longer(c(pct_urg, pct_neg), names_to = "metrica", values_to = "pct") |>
  mutate(metrica = recode(metrica, pct_urg = "% flag_urgenza", pct_neg = "% procedura NEGOZIATA"))

p14c <- ggplot(urg_tempo, aes(x = anno_pubblicazione, y = pct, color = settore, linetype = settore)) +
  geom_line(linewidth = 1.1) + geom_point(size = 2) +
  facet_wrap(~ metrica, scales = "free_y") +
  scale_color_manual(values = c("SANITA" = "#C73E1D", "Altri settori" = "#455A64"), name = NULL) +
  scale_linetype_manual(values = c("SANITA" = "solid", "Altri settori" = "dashed"), name = NULL) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(title = "Urgenza e procedura negoziata nel tempo",
       subtitle = "Se SANITA usa sistematicamente più urgenza/negoziate, conferma l'esposizione strutturale",
       x = "Anno pubblicazione", y = "% appalti") +
  theme_paper()

p14 <- (p14a + p14b) / p14c + plot_layout(heights = c(1.1, 1)) +
  plot_annotation(
    title    = "Settore SANITARIO: esposizione corruttiva documentata",
    subtitle = "CPV SANITA è noto in letteratura per alto ricorso a negoziate/urgenza (Gnaldi & Del Sarto 2018)",
    caption  = "Fonte: M1 nativi.",
    theme    = theme(plot.title = element_text(size = 15, face = "bold"),
                     plot.subtitle = element_text(color = "grey30"))
  )

save_plot(p14, "14_sanita_deepdive", w = 16, h = 11)

# Cluster temporali → operazioni giudiziarie specifiche.
# Sbilanciamento temporale → rischio leakage temporale.
# Fonte: M1 labeled.
message("15 - Distribuzione temporale dei labeled")

temp_lab <- labeled_m1 |>
  filter(!is.na(anno_pubblicazione),
         anno_pubblicazione >= 2008, anno_pubblicazione <= 2024) |>
  count(anno_pubblicazione, label_fct)

p15 <- ggplot(temp_lab, aes(x = anno_pubblicazione, y = n, fill = label_fct)) +
  geom_col(position = "dodge", alpha = 0.9, width = 0.8) +
  geom_text(aes(label = n), position = position_dodge(0.8), vjust = -0.4, size = 2.8) +
  scale_lab +
  scale_x_continuous(breaks = 2008:2024) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(title    = "Distribuzione temporale dei contratti labeled",
       subtitle = "Condannati e assolti per anno di pubblicazione bando",
       x = "Anno pubblicazione", y = "N. contratti",
       caption  = paste0("Cluster temporali suggeriscono effetti di operazioni giudiziarie specifiche. ",
                         "Un campione temporalmente sbilanciato può indurre leakage temporale nel modello. ",
                         "Fonte: M1 nativi.")) +
  theme_paper() + theme(axis.text.x = element_text(angle = 45, hjust = 1))

save_plot(p15, "15_labeled_temporal_distribution", w = 14, h = 7)

# Pattern MNAR: missing differenziale per label è informativo.
# Fonte: M1 (labeled + campione U 5k).
message("16 - Missing data heatmap per feature M1")

feat_m1_heatmap <- c("importo_lotto", "importo_complessivo_gara", "n_lotti_componenti",
                      "finestra_offerta_giorni", "lag_perfezionamento_giorni",
                      "importo_sicurezza_pct", "pct_riserva_base",
                      "flag_urgenza", "flag_accordo_quadro", "flag_ripetizioni",
                      "settore_speciale", "flag_appalto_riservato",
                      "tasso_disoccupazione", "reddito_irpef_procapite", "tasso_omicidi_100k")

feat_disp <- intersect(feat_m1_heatmap, names(m1))
u_sample  <- m1 |> filter(is.na(label)) |> slice_sample(n = 5000)

groups_df <- bind_rows(
  labeled_m1 |> filter(label == 1) |> mutate(gruppo = "Condannato (P)"),
  labeled_m1 |> filter(label == 0) |> mutate(gruppo = "Assolto (N)"),
  u_sample                          |> mutate(gruppo = "Unlabeled (campione)")
)

miss_hm <- groups_df |>
  select(gruppo, all_of(feat_disp)) |>
  group_by(gruppo) |>
  summarise(across(everything(), ~ mean(is.na(.x)) * 100), .groups = "drop") |>
  pivot_longer(-gruppo, names_to = "feature", values_to = "pct_missing") |>
  mutate(gruppo = factor(gruppo, levels = c("Condannato (P)", "Assolto (N)", "Unlabeled (campione)")))

p16 <- ggplot(miss_hm,
              aes(x = gruppo, y = fct_reorder(feature, pct_missing, .fun = mean),
                  fill = pct_missing)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%.0f%%", pct_missing)),
            size  = 2.8,
            color = ifelse(miss_hm$pct_missing > 50, "white", "grey15")) +
  scale_fill_gradient2(low = "#E3F2FD", mid = "#FFF3E0", high = "#B71C1C",
                       midpoint = 40, name = "% missing") +
  labs(title    = "% valori mancanti per feature M1 e gruppo",
       subtitle = "Pattern MNAR: se P o N hanno missing diverso da U, il missing è potenzialmente informativo",
       x = NULL, y = NULL,
       caption  = "Feature ordinate per % missing media crescente. Solo feature M1. Fonte: M1 nativi.") +
  theme_paper() +
  theme(axis.text.x = element_text(face = "bold"), panel.grid = element_blank())

save_plot(p16, "16_missing_heatmap_m1", w = 10, h = 8)

# IC Wilson per regione — check di validità esterna vs
# indici regionali di corruzione (Transparency Italia).
# Fonte: M1 labeled.
message("17 - Tasso condanne per regione")

reg_stats17 <- labeled_m1 |>
  filter(!is.na(regione)) |>
  group_by(regione) |>
  summarise(n = n(), k = sum(label == 1), .groups = "drop") |>
  filter(n >= 10) |>
  rowwise() |>
  mutate(pct   = k / n * 100,
         ci_lo = wilson_ci(k, n)[1] * 100,
         ci_hi = wilson_ci(k, n)[2] * 100) |>
  ungroup()

pct_nazionale <- mean(labeled_m1$label == 1) * 100

p17 <- ggplot(reg_stats17, aes(x = fct_reorder(regione, pct), y = pct)) +
  geom_hline(yintercept = pct_nazionale, linetype = "dashed",
             color = "grey40", linewidth = 0.7) +
  geom_col(aes(fill = pct), alpha = 0.92, width = 0.75) +
  geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi),
                width = 0.25, color = "grey30", linewidth = 0.4) +
  geom_text(aes(label = sprintf("%.0f%% (n=%d)", pct, n)), hjust = -0.1, size = 2.8) +
  scale_fill_gradient2(low = "#2E86AB", mid = "#FFF3E0", high = "#C73E1D",
                       midpoint = pct_nazionale, guide = "none") +
  coord_flip() +
  scale_y_continuous(labels = function(x) paste0(x, "%"),
                     limits = c(0, max(reg_stats17$ci_hi) * 1.3), expand = c(0, 0)) +
  labs(title    = "Tasso condanne per regione (labeled M1)",
       subtitle = sprintf("IC 95%% Wilson  |  media nazionale: %.1f%% (linea tratteggiata)", pct_nazionale),
       x = NULL, y = "% condannati",
       caption  = paste0("Regioni con n < 10 escluse. ",
                         "Confrontare con indici regionali di corruzione percepita (Transparency Italia). ",
                         "Fonte: M1 nativi.")) +
  theme_paper()

save_plot(p17, "17_tasso_condanne_regione", w = 12, h = 9)

# Pannelli: etichetta (P/N/U) × tipo procedura
# per M1, M2, M3. Coordinate salvate in cache/ per il
# landscape PDF generato subito dopo.
#
# Gower su feature miste → k-NN esatti → UMAP precomputed.
# Campione: tutti i labeled + 10.000 U per modello.
# Tempi attesi: ~3-5 min/modello — RAM picco ~1.2 GB.

N_U_UMAP <- 10000L; UMAP_SEED <- 42L; UMAP_K <- 15L; UMAP_EPOCHS <- 200L

pal_proc_umap <- c(
  "APERTA"              = "#4C9A6E", "NEGOZIATA"           = "#E07B39",
  "RISTRETTA"           = "#7B5EA7", "AFFIDAMENTO_DIRETTO" = "#3A7CBF"
)
lab_proc_umap <- c(
  "APERTA"              = "Aperta",  "NEGOZIATA"           = "Negoziata",
  "RISTRETTA"           = "Ristretta", "AFFIDAMENTO_DIRETTO" = "Affidamento diretto"
)

theme_umap <- function(base = 17) {
  theme_minimal(base_size = base) +
    theme(plot.title      = element_text(face = "bold", size = base + 1, hjust = 0.5),
          plot.background = element_rect(fill = "white", color = NA),
          panel.grid      = element_blank(),
          axis.text       = element_blank(),
          axis.ticks      = element_blank(),
          axis.title      = element_text(color = "grey60", size = base - 2),
          legend.position = "bottom", legend.title = element_blank(),
          legend.text     = element_text(size = base - 1),
          legend.key.size = unit(0.85, "lines"),
          panel.border    = element_rect(color = "grey88", fill = NA, linewidth = 0.4))
}

run_umap_panels <- function(df_full, labeled_df,
                             feat_num, feat_cat, feat_bin,
                             model_name, col_title,
                             n_u = N_U_UMAP, umap_k = UMAP_K,
                             umap_ep = UMAP_EPOCHS, umap_seed = UMAP_SEED) {

  feat_present <- intersect(c(feat_num, feat_cat, feat_bin), names(df_full))
  feat_log     <- intersect(c("importo_lotto", "importo_complessivo_gara",
                               "importo_aggiudicazione"), feat_present)

  message(sprintf("[%s] %d feature — campionamento...", model_name, length(feat_present)))
  set.seed(umap_seed)
  df_s <- bind_rows(labeled_df,
                    df_full |> filter(is.na(label)) |> slice_sample(n = n_u)) |>
    mutate(gruppo = factor(
      case_when(label == 1 ~ "Condannato (P)", label == 0 ~ "Assolto (N)",
                TRUE ~ "Non etichettato"),
      levels = c("Non etichettato", "Assolto (N)", "Condannato (P)")
    ))

  n_P <- sum(df_s$gruppo == "Condannato (P)")
  n_N <- sum(df_s$gruppo == "Assolto (N)")
  n_U <- sum(df_s$gruppo == "Non etichettato")
  message(sprintf("[%s] %d P + %d N + %d U", model_name, n_P, n_N, n_U))

  X_g <- df_s |>
    select(all_of(feat_present)) |>
    mutate(across(all_of(feat_log), ~ log1p(pmax(., 0, na.rm = FALSE))),
           across(all_of(intersect(c(feat_cat, feat_bin), feat_present)),
                  ~ factor(as.character(.))))

  message(sprintf("[%s] Gower...", model_name))
  t0    <- proc.time()
  gdist <- cluster::daisy(X_g, metric = "gower")
  tg    <- round((proc.time() - t0)[["elapsed"]])
  message(sprintf("[%s] Gower: %ds — RAM: %s", model_name, tg,
                  format(object.size(gdist), units = "MB")))

  message(sprintf("[%s] k=%d NN esatti...", model_name, umap_k))
  t0      <- proc.time()
  gmat    <- as.matrix(gdist); rm(gdist)
  n_pts   <- nrow(gmat)
  nn_idx  <- t(apply(gmat, 1, function(d) order(d)[2L:(umap_k + 1L)]))
  nn_dist <- t(sapply(seq_len(n_pts), function(i) gmat[i, nn_idx[i, ]]))
  rm(gmat)
  message(sprintf("[%s] NN estratti in %ds.", model_name,
                  round((proc.time() - t0)[["elapsed"]])))

  message(sprintf("[%s] UMAP (epochs=%d)...", model_name, umap_ep))
  t0   <- proc.time()
  set.seed(umap_seed)
  umat <- uwot::umap(
    X         = matrix(0.0, nrow = n_pts, ncol = 1),
    nn_method = list(idx = nn_idx, dist = nn_dist),
    init      = "random", min_dist = 0.1,
    n_epochs  = umap_ep, n_threads = 4, verbose = FALSE, ret_model = FALSE
  )
  message(sprintf("[%s] UMAP: %ds", model_name, round((proc.time() - t0)[["elapsed"]])))

  df_s <- df_s |> mutate(U1 = umat[, 1], U2 = umat[, 2])

  saveRDS(
    df_s |> select(any_of(c("cig", "U1", "U2", "gruppo",
                              "tipo_scelta_4cls", "cpv_macro_categoria",
                              "natura_giuridica_SA", "sezione_regionale",
                              "anno_pubblicazione", "importo_lotto",
                              "flag_urgenza", "flag_accordo_quadro"))),
    file.path(CACHEDIR, paste0("coords_", model_name, ".rds"))
  )
  message(sprintf("[%s] coordinate → cache/coords_%s.rds", model_name, model_name))

  df_plot <- df_s |> arrange(gruppo)
  ann_counts <- sprintf("P=%d  N=%d  U=%d", n_P, n_N, n_U)

  pa <- ggplot(df_plot, aes(x = U1, y = U2, color = gruppo, size = gruppo, alpha = gruppo)) +
    geom_point(shape = 16) +
    scale_color_manual(
      values = c("Non etichettato" = col_U, "Assolto (N)" = col_N, "Condannato (P)" = col_P),
      guide  = guide_legend(override.aes = list(size = 3.5, alpha = 1), nrow = 1)) +
    scale_size_manual(values  = c("Non etichettato" = 0.7, "Assolto (N)" = 1.4,
                                   "Condannato (P)" = 1.4), guide = "none") +
    scale_alpha_manual(values = c("Non etichettato" = 0.28, "Assolto (N)" = 0.80,
                                   "Condannato (P)" = 1.0), guide = "none") +
    annotate("text", x = Inf, y = -Inf, label = ann_counts,
             hjust = 1.05, vjust = -0.5, size = 2.8, color = "grey55", fontface = "italic") +
    labs(title = col_title, x = "UMAP 1", y = "UMAP 2") + theme_umap()

  proc_vals <- sort(unique(na.omit(df_s$tipo_scelta_4cls)))
  col_p <- pal_proc_umap[proc_vals]; col_p[is.na(col_p)] <- "#9E9E9E"; names(col_p) <- proc_vals
  lbl_p <- lab_proc_umap[proc_vals]; lbl_p[is.na(lbl_p)] <- proc_vals[is.na(lbl_p)]; names(lbl_p) <- proc_vals

  pc <- ggplot(df_s |> filter(!is.na(tipo_scelta_4cls)),
               aes(x = U1, y = U2, color = tipo_scelta_4cls)) +
    geom_point(alpha = 0.32, size = 0.55, shape = 16) +
    scale_color_manual(values = col_p, labels = lbl_p,
                       guide  = guide_legend(override.aes = list(size = 3.5, alpha = 1), nrow = 2)) +
    labs(title = NULL, x = "UMAP 1", y = "UMAP 2") + theme_umap()

  list(pa = pa, pc = pc)
}

message("\n=== UMAP M1 ===")
pnl1 <- run_umap_panels(m1, labeled_m1, FEAT_M1_NUM, FEAT_M1_CAT, FEAT_M1_BIN,
                         "M1", "M1 — Ante-aggiudicazione")

message("\n=== UMAP M2 ===")
pnl2 <- run_umap_panels(m2, labeled_m2,
                         c(FEAT_M1_NUM, FEAT_M2_EXTRA_NUM[-length(FEAT_M2_EXTRA_NUM)]), # escluso ribasso_spread (89% NA)
                         c(FEAT_M1_CAT, FEAT_M2_EXTRA_CAT),
                         c(FEAT_M1_BIN, FEAT_M2_EXTRA_BIN),
                         "M2", "M2 — Durante l'aggiudicazione")

message("\n=== UMAP M3 ===")
pnl3 <- run_umap_panels(m3, labeled_m3,
                         c(FEAT_M1_NUM, FEAT_M2_EXTRA_NUM[-length(FEAT_M2_EXTRA_NUM)], FEAT_M3_EXTRA_NUM),
                         c(FEAT_M1_CAT, FEAT_M2_EXTRA_CAT, FEAT_M3_EXTRA_CAT),
                         c(FEAT_M1_BIN, FEAT_M2_EXTRA_BIN, FEAT_M3_EXTRA_BIN),
                         "M3", "M3 — Post-aggiudicazione")

message("18 - UMAP grid 3x2")
top_row <- (pnl1$pa | pnl2$pa | pnl3$pa) +
  plot_layout(guides = "collect") & theme(legend.position = "bottom")
bot_row <- (pnl1$pc | pnl2$pc | pnl3$pc) +
  plot_layout(guides = "collect") & theme(legend.position = "bottom")

p18 <- top_row / bot_row +
  plot_annotation(theme = theme(plot.background = element_rect(fill = "white", color = NA)))

save_plot(p18, "18_umap_grid", w = 20, h = 13)

# Landscape 4×3 ad alta risoluzione: righe = Etichetta /
# CPV / Procedura / Importo — colonne = M1 / M2 / M3.
message("UMAP landscape PDF...")

col_N_land <- "#1B998B"; col_P_land <- "#C73E1D"; col_U_land <- "#CFD8DC"

pal_label_l <- c("Non etichettato" = col_U_land, "Assolto (N)" = col_N_land,
                  "Condannato (P)"  = col_P_land)
pal_cpv_l   <- c("Sanità" = "#C73E1D", "Lavori" = "#E07B39", "Servizi" = "#3A7CBF",
                  "Forniture" = "#7B5EA7", "Informatica" = "#4C9A6E",
                  "Servizi tecnici e professionali" = "#1D7A5F")
pal_proc_l  <- c("Negoziata" = "#E07B39", "Affidamento diretto" = "#3A7CBF",
                  "Aperta"   = "#4C9A6E", "Ristretta" = "#7B5EA7")
pal_imp_l   <- c("> 1M" = "#1D3557", "150k–1M" = "#2E86AB",
                  "40k–150k" = "#7FBEEB", "< 40k" = "#CFD8DC")

theme_land <- function(base = 25) {
  theme_minimal(base_size = base) +
    theme(plot.title        = element_text(face = "bold", size = 25, hjust = 0.5),
          plot.background   = element_rect(fill = "white", color = NA),
          plot.margin       = margin(2, 4, 2, 4),
          panel.grid        = element_blank(),
          axis.text         = element_blank(), axis.ticks = element_blank(),
          axis.title        = element_blank(),
          legend.position   = "bottom", legend.title = element_blank(),
          legend.text       = element_text(size = 25),
          legend.key.size   = unit(1.1, "lines"),
          legend.margin     = margin(0, 0, 0, 0),
          legend.box.margin = margin(-25, 0, 2, 0),
          panel.border      = element_rect(color = "grey88", fill = NA, linewidth = 0.4))
}

load_coords <- function(model_name) {
  path <- file.path(CACHEDIR, paste0("coords_", model_name, ".rds"))
  if (!file.exists(path))
    stop("coordinate non trovate: ", path, "\nEseguire prima il blocco UMAP sopra.")
  df <- readRDS(path)
  if ("importo_lotto" %in% names(df))
    df <- df |> mutate(fascia_importo = cut(
      importo_lotto, breaks = c(0, 40000, 150000, 1000000, Inf),
      labels = c("< 40k", "40k–150k", "150k–1M", "> 1M"),
      right = FALSE, include.lowest = TRUE))
  if ("cpv_macro_categoria" %in% names(df))
    df <- df |> mutate(cpv_macro_categoria = recode(cpv_macro_categoria,
      SANITA = "Sanità", LAVORI = "Lavori", SERVIZI = "Servizi",
      FORNITURE = "Forniture", IT = "Informatica", ING_PROF = "Servizi tecnici e professionali"))
  if ("tipo_scelta_4cls" %in% names(df))
    df <- df |> mutate(tipo_scelta_4cls = recode(tipo_scelta_4cls,
      AFFIDAMENTO_DIRETTO = "Affidamento diretto",
      APERTA = "Aperta", NEGOZIATA = "Negoziata", RISTRETTA = "Ristretta"))
  df
}

panel_label_l <- function(df, title) {
  df_plot <- df |>
    mutate(gruppo = factor(gruppo, levels = c("Non etichettato", "Assolto (N)", "Condannato (P)"))) |>
    arrange(gruppo)
  ggplot(df_plot, aes(U1, U2, color = gruppo, size = gruppo, alpha = gruppo)) +
    geom_point(shape = 16) +
    scale_color_manual(values = pal_label_l,
                       guide  = guide_legend(override.aes = list(size=3, alpha=1), nrow=1)) +
    scale_size_manual(values  = c("Non etichettato"=0.45, "Assolto (N)"=1.2, "Condannato (P)"=1.0),
                      guide   = "none") +
    scale_alpha_manual(values = c("Non etichettato"=0.25, "Assolto (N)"=0.80, "Condannato (P)"=1.0),
                       guide  = "none") +
    labs(title = title) + theme_land()
}

panel_var_l <- function(df, varname, pal, nrow_leg = 2) {
  if (!varname %in% names(df)) return(ggplot() + theme_void())
  df2     <- df |> filter(!is.na(.data[[varname]]))
  present <- unique(as.character(df2[[varname]]))
  vals    <- c(intersect(names(pal), present), setdiff(present, names(pal)))
  col_v   <- pal[vals]; col_v[is.na(col_v)] <- "#9E9E9E"; names(col_v) <- vals
  df2     <- df2 |> mutate(across(all_of(varname), ~ factor(., levels = vals)))
  ggplot(df2, aes(U1, U2, color = .data[[varname]])) +
    geom_point(alpha = 0.30, size = 0.5, shape = 16) +
    scale_color_manual(values = col_v,
                       guide  = guide_legend(override.aes = list(size=3, alpha=1), nrow=nrow_leg)) +
    labs(title = NULL) + theme_land()
}

make_row_l <- function(panels_list, row_label) {
  panels_list[[1]] <- panels_list[[1]] +
    labs(tag = row_label) +
    theme(plot.tag = element_text(size=25, face="bold", color="grey30", angle=90, hjust=0.5),
          plot.tag.position = "left")
  wrap_plots(panels_list, nrow = 1) +
    plot_layout(guides = "collect") & theme(legend.position = "bottom")
}

message("Carico coordinate da cache/..."); flush.console()
dfs        <- setNames(lapply(c("M1","M2","M3"), load_coords), c("M1","M2","M3"))
titles_col <- c("M1", "M2", "M3")

row_label <- make_row_l(lapply(1:3, function(i) panel_label_l(dfs[[i]], titles_col[i])), "Etichetta")
row_cpv   <- make_row_l(lapply(1:3, function(i) panel_var_l(dfs[[i]], "cpv_macro_categoria", pal_cpv_l, 2)), "CPV")
row_proc  <- make_row_l(lapply(1:3, function(i) panel_var_l(dfs[[i]], "tipo_scelta_4cls",    pal_proc_l, 1)), "Procedura")
row_imp   <- make_row_l(lapply(1:3, function(i) panel_var_l(dfs[[i]], "fascia_importo",      pal_imp_l,  1)), "Importo")

fig_land <- row_label / row_cpv / row_proc / row_imp +
  plot_annotation(theme = theme(plot.background = element_rect(fill = "white", color = NA)))

path_pdf <- file.path(OUTDIR, "umap_landscape.pdf")
message("Salvo landscape PDF..."); flush.console()
ggsave(path_pdf, fig_land, width = 28, height = 32,
       units = "in", device = cairo_pdf, bg = "white")
message("  → ", path_pdf)

cat("EDA completata. Grafici in:\n"); cat(OUTDIR, "\n")
for (f in sort(list.files(OUTDIR, "\\.(png|pdf)$"))) cat("  -", f, "\n")
