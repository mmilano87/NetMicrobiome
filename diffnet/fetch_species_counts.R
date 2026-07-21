#!/usr/bin/env Rscript
# fetch_species_counts.R -- fetch MetaPhlAn SPECIES-level integer counts from curatedMetagenomicData.
#
# Unlike the HUMAnN gene_families table (where counts=TRUE is a no-op and returns fractional
# relative abundance), the MetaPhlAn relative_abundance table's counts=TRUE returns proper
# INTEGER counts (relative abundance x sample read depth, rounded). Verified directly:
# all nonzero values integer-valued, colSums ~ read depths (5-6e7).
#
# These species-level integer counts are the valid input for count-based compositional methods
# (ALDEx2, SPIEC-EASI, SparCC) requested by Reviewers 2/3. This is a SPECIES-level companion
# track to the gene-family association networks in the manuscript.
#
# Usage: Rscript fetch_species_counts.R <cohort>
# Writes to /mnt/shared-workspace/micro/data/:
#   <cohort>_species_counts.mtx        (features x samples, integer counts, MatrixMarket)
#   <cohort>_species_counts_rows.txt   (species names, full lineage)
#   <cohort>_species_counts_cols.txt   (sample_ids, aligned to metadata sample_id)
suppressMessages({ library(Matrix); library(curatedMetagenomicData); library(SummarizedExperiment) })

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("usage: Rscript fetch_species_counts.R <cohort>")
cohort <- args[1]

resource <- c(
  acvd = "2021-10-14.JieZ_2017.relative_abundance",
  ibd  = "2021-10-14.HMP_2019_ibdmdb.relative_abundance",
  t2d  = "2022-10-19.MetaCardis_2020_a.relative_abundance"
)[[cohort]]
stopifnot(!is.null(resource))

DATA <- "/mnt/shared-workspace/micro/data"
dir.create(DATA, recursive = TRUE, showWarnings = FALSE)

message("[", cohort, "] fetching SPECIES counts: ", resource)
x <- curatedMetagenomicData(resource, dryrun = FALSE, counts = TRUE, rownames = "long")
se <- x[[1]]
a <- assay(se)                       # features x samples
# sanity: must be integer counts
nz <- as.numeric(a[a != 0])
n_int <- sum(nz == round(nz))
message("  fetched: ", nrow(a), " species x ", ncol(a), " samples | nnz=", length(nz),
        " | integer-valued=", n_int, " (", round(100 * n_int / length(nz), 1), "%)",
        " | max=", max(a), " | median colSum=", round(median(colSums(a))))
if (n_int < length(nz)) {
  # round defensively (should already be integers)
  a <- round(a)
  message("  WARNING: non-integer values present; rounded.")
}

# Align sample IDs to our metadata's sample_id. cMD colnames are the cMD sample names;
# our metadata sample_id column should match. Keep intersection.
meta <- read.delim(file.path(DATA, paste0(cohort, "_metadata.tsv")), stringsAsFactors = FALSE)
cn <- colnames(a)
keep <- cn %in% meta$sample_id
message("  samples matching metadata sample_id: ", sum(keep), " / ", length(cn))
if (sum(keep) < length(cn)) {
  # try matching on the cMD default (rownames of colData) as fallback
  cd <- as.data.frame(colData(se))
  message("  colData columns: ", paste(head(colnames(cd), 20), collapse = ", "))
}
a <- a[, keep, drop = FALSE]

# drop all-zero species (after sample subset)
rs <- Matrix::rowSums(a)
a <- a[rs > 0, , drop = FALSE]
message("  after sample-match + drop-empty: ", nrow(a), " species x ", ncol(a), " samples")

sp <- Matrix(as.matrix(a), sparse = TRUE)
local_mtx <- file.path("/workspace", paste0(cohort, "_species_counts.mtx"))
writeMM(sp, local_mtx)
writeLines(rownames(a), file.path(DATA, paste0(cohort, "_species_counts_rows.txt")))
writeLines(colnames(a), file.path(DATA, paste0(cohort, "_species_counts_cols.txt")))
# copy mtx from local /workspace to shared (S3 FUSE lacks random-access writes)
dest <- file.path(DATA, paste0(cohort, "_species_counts.mtx"))
st <- system2("cp", c(local_mtx, dest))
message("  copied mtx -> ", dest, " (status ", st, ")")
message("[", cohort, "] SPECIES counts fetch DONE")
