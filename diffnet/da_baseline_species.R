#!/usr/bin/env Rscript
# da_baseline_species.R -- SPECIES-level differential-abundance baseline with ALDEx2 (Reviewer 1.6).
#
# Count-based compositional DA (ALDEx2) requires INTEGER counts. cMD provides integer counts only
# at the MetaPhlAn species level (counts=TRUE is a no-op for HUMAnN gene families). We therefore run
# ALDEx2 on species-level integer counts as the count-based compositional DA reference, testing
# male vs female WITHIN each condition (same design as the networks). The Python side then compares
# the DA-significant species to the network-derived signal to address whether the differential
# NETWORK view adds information beyond standard differential ABUNDANCE.
#
# Usage: Rscript da_baseline_species.R <cohort> [--ibd-mode samples|subject]
# Outputs -> /mnt/shared-workspace/micro/robust/<cohort>/da_aldex2_species_<condition>.tsv
suppressMessages({ library(Matrix); library(ALDEx2) })

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript da_baseline_species.R <cohort> [--ibd-mode ...]")
cohort <- args[1]
ibd_mode <- "samples"
if ("--ibd-mode" %in% args) ibd_mode <- args[which(args == "--ibd-mode") + 1]

DATA <- "/mnt/shared-workspace/micro/data"
OUT <- file.path("/mnt/shared-workspace/micro/robust", cohort)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

conds <- list(
  acvd = list(c("control","control_female","control_male"), c("ACVD","ACVD_female","ACVD_male")),
  ibd  = list(c("control","control_female","control_male"), c("IBD","IBD_female","IBD_male")),
  t2d  = list(c("control","control_female","control_male"), c("T2D","T2D_female","T2D_male"))
)[[cohort]]

meta <- read.delim(file.path(DATA, paste0(cohort, "_metadata.tsv")), stringsAsFactors = FALSE)
meta$group <- paste0(meta$study_condition, "_", meta$gender)
if (cohort == "ibd" && ibd_mode == "subject") {
  meta$._d <- suppressWarnings(as.numeric(meta$days_from_first_collection)); meta$._d[is.na(meta$._d)] <- 0
  meta <- do.call(rbind, lapply(split(meta, meta$subject_id), function(d) d[which.min(d$._d), ]))
}

col_ids <- readLines(file.path(DATA, paste0(cohort, "_species_counts_cols.txt")))
row_ids <- readLines(file.path(DATA, paste0(cohort, "_species_counts_rows.txt")))
sample_col <- setNames(seq_along(col_ids), col_ids)

message("[", cohort, "] reading SPECIES counts mtx ...")
M <- readMM(file.path(DATA, paste0(cohort, "_species_counts.mtx")))  # species x samples (integer counts)
rownames(M) <- row_ids

for (cc in conds) {
  cond <- cc[1]; g_f <- cc[2]; g_m <- cc[3]
  sids <- meta$sample_id[meta$group %in% c(g_f, g_m)]
  sids <- sids[sids %in% names(sample_col)]
  condv <- ifelse(meta$group[match(sids, meta$sample_id)] == g_m, "male", "female")
  if (min(table(condv)) < 5) { message("  [", cond, "] a sex has <5 samples, skip"); next }

  sub <- as.matrix(M[, sample_col[sids], drop = FALSE])   # species x samples
  # drop species that are all-zero within this condition's samples (ALDEx2 needs nonzero rows)
  sub <- sub[rowSums(sub) > 0, , drop = FALSE]
  storage.mode(sub) <- "integer"
  colnames(sub) <- sids
  message("  [", cond, "] ALDEx2 on ", nrow(sub), " species x ", ncol(sub),
          " samples (", paste(names(table(condv)), table(condv), collapse=", "), ")")

  set.seed(1)
  x <- aldex(sub, condv, mc.samples = 128, test = "t", effect = TRUE, denom = "all", verbose = FALSE)
  x$feature <- rownames(x)
  x <- x[order(x$we.eBH), ]
  write.table(x, file.path(OUT, paste0("da_aldex2_species_", cond, ".tsv")),
              sep = "\t", quote = FALSE, row.names = FALSE)
  n_sig <- sum(x$we.eBH < 0.05, na.rm = TRUE)
  message("    -> ", n_sig, " species with we.eBH<0.05 (of ", nrow(x), ")")
}
message("[", cohort, "] SPECIES DA baseline DONE -> ", OUT)
