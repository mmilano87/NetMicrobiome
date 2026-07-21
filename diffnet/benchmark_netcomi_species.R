#!/usr/bin/env Rscript
# ============================================================================
# NetCoMi benchmark on species counts (ACVD cohort) vs our species_diffnet.
#
# For each condition (control, ACVD):
#   - split samples by sex (male / female)
#   - restrict to species with prevalence >= PREV in EITHER sex (shared node set)
#   - build NetCoMi differential network:
#        netConstruct(spearman, CLR) on the two groups, then
#        diffnet(permute, nPerm, FDR)  -> significant differential edges
#   - load OUR species_diffnet true_change edges for the same condition
#   - compute overlap / Jaccard / concordance between the two edge sets
#
# NetCoMi association thresholding uses the SAME threshold (thresh=THR) and the
# SAME CLR normalization as our pipeline, so the only methodological difference
# being benchmarked is the differential-edge *test* (NetCoMi's permutation z-test
# with FDR vs our |Delta rho| >= DELTA rule).
# ============================================================================
suppressMessages({
  library(NetCoMi)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
cohort <- ifelse(length(args) >= 1, args[1], "acvd")
NPERM  <- ifelse(length(args) >= 2, as.integer(args[2]), 1000L)

PREV   <- 0.20      # prevalence filter (matches species_diffnet)
THR    <- 0.30      # association threshold tau (matches species_diffnet)
DELTA  <- 0.30      # our |Delta rho| cutoff (for reference / reporting)
ALPHA  <- 0.05      # FDR level for NetCoMi diffnet
set.seed(1)

base   <- "/mnt/shared-workspace/micro"
datdir <- file.path(base, "data")
outdir <- file.path(base, "robust", cohort)
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

cat(sprintf("[%s] NetCoMi species benchmark  nPerm=%d prev=%.2f thr=%.2f\n",
            cohort, NPERM, PREV, THR))

## ---- load species counts (MatrixMarket, integer) --------------------------
mtx  <- readMM(file.path(datdir, sprintf("%s_species_counts.mtx", cohort)))
rn   <- readLines(file.path(datdir, sprintf("%s_species_counts_rows.txt", cohort)))
cn   <- readLines(file.path(datdir, sprintf("%s_species_counts_cols.txt", cohort)))
mtx  <- as.matrix(mtx)
rownames(mtx) <- rn        # species lineage strings
colnames(mtx) <- cn        # sample ids
cat(sprintf("  counts: %d species x %d samples\n", nrow(mtx), ncol(mtx)))

## ---- metadata: sample_id, study_condition, gender -------------------------
meta <- read.delim(file.path(datdir, sprintf("%s_metadata.tsv", cohort)),
                   stringsAsFactors = FALSE, check.names = FALSE)
# locate columns robustly by name
sid_col <- grep("^sample_id$", names(meta), ignore.case = TRUE)[1]
con_col <- grep("study_condition", names(meta), ignore.case = TRUE)[1]
sex_col <- grep("^gender$|^sex$", names(meta), ignore.case = TRUE)[1]
meta$.sid <- meta[[sid_col]]; meta$.con <- meta[[con_col]]; meta$.sex <- meta[[sex_col]]
rownames(meta) <- meta$.sid

# condition labels present in this cohort
cond_levels <- setdiff(unique(meta$.con), c(NA, ""))
cat("  conditions:", paste(cond_levels, collapse=", "), "\n")

prevfun <- function(M) rowMeans(M > 0)

# comparison helper: undirected edge key from two node names
ekey <- function(a, b) {
  ifelse(a < b, paste(a, b, sep = "\t"), paste(b, a, sep = "\t"))
}

results <- list()

for (cond in cond_levels) {
  # our species_diffnet uses "control" and the disease label (e.g. "ACVD")
  cond_tag <- cond
  sf <- file.path(outdir, sprintf("species_diffnet_%s.tsv", cond_tag))
  if (!file.exists(sf)) {
    cat(sprintf("  [%s] no species_diffnet file (%s) -> skip\n", cond, basename(sf)))
    next
  }

  smpl <- rownames(meta)[meta$.con == cond & meta$.sid %in% colnames(mtx)]
  male_ids   <- intersect(smpl, rownames(meta)[meta$.sex == "male"])
  female_ids <- intersect(smpl, rownames(meta)[meta$.sex == "female"])
  cat(sprintf("\n== %s ==  male=%d female=%d\n", cond, length(male_ids), length(female_ids)))
  if (length(male_ids) < 10 || length(female_ids) < 10) {
    cat("   too few samples in a group -> skip\n"); next
  }

  Xm <- mtx[, male_ids,   drop = FALSE]
  Xf <- mtx[, female_ids, drop = FALSE]

  # shared node set: prevalence >= PREV in EITHER sex
  keep <- (prevfun(Xm) >= PREV) | (prevfun(Xf) >= PREV)
  feats <- rownames(mtx)[keep]
  cat(sprintf("   shared feature set (prev>=%.2f in either sex): %d species\n",
              PREV, length(feats)))

  # NetCoMi expects samples in ROWS, taxa in COLUMNS
  Am <- t(Xm[feats, , drop = FALSE])
  Af <- t(Xf[feats, , drop = FALSE])

  net <- netConstruct(
    data = Am, data2 = Af, dataType = "counts",
    measure = "spearman", normMethod = "clr",
    zeroMethod = "none", sparsMethod = "threshold", thresh = THR,
    dissFunc = "signed", verbose = 0
  )

  dn <- diffnet(
    net, diffMethod = "permute", nPerm = NPERM,
    adjust = "fdr", alpha = ALPHA, cores = 4, seed = 1, verbose = FALSE
  )

  ## --- NetCoMi significant differential edges (FDR < ALPHA) ---------------
  padj <- dn$pAdjustMat
  nm   <- rownames(padj)
  sig_idx <- which(padj < ALPHA & upper.tri(padj), arr.ind = TRUE)
  nc_edges <- if (nrow(sig_idx) > 0)
    ekey(nm[sig_idx[,1]], nm[sig_idx[,2]]) else character(0)
  nc_edges <- unique(nc_edges)
  cat(sprintf("   NetCoMi diffnet FDR<%.2f differential edges: %d\n", ALPHA, length(nc_edges)))

  ## --- our species_diffnet true_change edges ------------------------------
  sd <- read.delim(sf, stringsAsFactors = FALSE, check.names = FALSE)
  sd_true <- sd[sd$artifact == "true_change", , drop = FALSE]
  our_true  <- unique(ekey(sd_true$Node1, sd_true$Node2))
  sd_all_diff <- unique(ekey(sd$Node1, sd$Node2))  # true_change + node_missing
  cat(sprintf("   our true_change edges: %d ; all differential (incl node_missing): %d\n",
              length(our_true), length(sd_all_diff)))

  ## --- restrict comparison to the SHARED node set both tools saw ----------
  # (our node_missing edges by definition involve a feature absent in one sex,
  #  which NetCoMi cannot flag; the fair comparison is on true_change.)
  fe <- feats
  in_shared <- function(ek) {
    parts <- strsplit(ek, "\t", fixed = TRUE)
    vapply(parts, function(p) all(p %in% fe), logical(1))
  }
  our_true_sh <- our_true[in_shared(our_true)]
  nc_sh       <- nc_edges[in_shared(nc_edges)]

  inter <- intersect(nc_sh, our_true_sh)
  uni   <- union(nc_sh, our_true_sh)
  jacc  <- if (length(uni) > 0) length(inter) / length(uni) else NA_real_
  # fraction of NetCoMi edges we also call true_change; and vice versa
  frac_nc_in_ours  <- if (length(nc_sh) > 0) length(inter) / length(nc_sh) else NA_real_
  frac_ours_in_nc  <- if (length(our_true_sh) > 0) length(inter) / length(our_true_sh) else NA_real_

  cat(sprintf("   [shared-node comparison] NetCoMi=%d  ours(true_change)=%d  overlap=%d  Jaccard=%.4f\n",
              length(nc_sh), length(our_true_sh), length(inter), jacc))
  cat(sprintf("   frac NetCoMi edges in ours=%.4f ; frac ours in NetCoMi=%.4f\n",
              frac_nc_in_ours, frac_ours_in_nc))

  results[[cond]] <- data.frame(
    cohort = cohort, condition = cond, nPerm = NPERM,
    n_male = length(male_ids), n_female = length(female_ids),
    n_shared_feat = length(feats),
    netcomi_sig_edges = length(nc_sh),
    our_true_change  = length(our_true_sh),
    overlap = length(inter),
    jaccard = jacc,
    frac_netcomi_in_ours = frac_nc_in_ours,
    frac_ours_in_netcomi = frac_ours_in_nc,
    our_node_missing_edges = sum(sd$artifact == "node_missing"),
    stringsAsFactors = FALSE
  )

  # save the NetCoMi edge list + adjusted p for this condition
  write.table(
    data.frame(edge = nc_sh),
    file.path(outdir, sprintf("netcomi_diffedges_%s.tsv", cond_tag)),
    sep = "\t", quote = FALSE, row.names = FALSE
  )
}

if (length(results) > 0) {
  summ <- do.call(rbind, results)
  outf <- file.path(outdir, "benchmark_netcomi_species_summary.csv")
  # write to /workspace first is unnecessary for CSV; write direct
  write.csv(summ, outf, row.names = FALSE)
  cat("\n=== SUMMARY ===\n")
  print(summ, row.names = FALSE)
  cat(sprintf("\nWrote %s\n", outf))
} else {
  cat("\nNo conditions produced results.\n")
}
