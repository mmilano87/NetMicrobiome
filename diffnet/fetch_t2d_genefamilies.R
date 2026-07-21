#!/usr/bin/env Rscript
# fetch_t2d_genefamilies.R
# Build the T2D (MetaCardis_2020_a) inputs for the differential-network pipeline,
# matching exactly how the ACVD gene-family relative-abundance table was produced:
#   - HUMAnN gene_families table, counts = FALSE  (relative abundance; counts=TRUE is a
#     documented no-op for gene_families in cMD -> we use relative abundance and CLR downstream)
#   - rownames = "long"
#   - columns aligned to a filtered metadata (T2D + control, gender known)
#
# Writes to /mnt/shared-workspace/micro/data/:
#   t2d_metadata.tsv               (filtered: study_condition in {T2D,control}, gender in {male,female})
#   t2d_gene_families.mtx          (features x samples, relative abundance, MatrixMarket)
#   t2d_gene_families_rows.txt
#   t2d_gene_families_cols.txt
suppressMessages({
  library(Matrix); library(curatedMetagenomicData); library(SummarizedExperiment)
})

DATA <- "/mnt/shared-workspace/micro/data"
resource <- "2022-10-19.MetaCardis_2020_a.gene_families"

## ---- 1. filtered metadata ------------------------------------------------
full <- read.delim(file.path(DATA, "t2d_metadata_full.tsv"),
                   stringsAsFactors = FALSE, check.names = FALSE)
keep <- full$study_condition %in% c("T2D", "control") &
        full$gender %in% c("male", "female")
meta <- full[keep, , drop = FALSE]
message("[t2d] filtered metadata: ", nrow(meta), " samples (T2D+control, gender known)")
tab <- table(meta$study_condition, meta$gender)
print(tab)
# write with sample_id as first column (pipeline load_meta sets index=sample_id)
write.table(meta, file.path(DATA, "t2d_metadata.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
message("  wrote t2d_metadata.tsv")

## ---- 2. fetch gene_families relative abundance ---------------------------
message("[t2d] fetching gene_families (counts=FALSE): ", resource)
tse <- curatedMetagenomicData(resource, dryrun = FALSE, counts = FALSE,
                              rownames = "long")[[1]]
a <- assay(tse)                       # features x samples (relative abundance)
message("  fetched: ", nrow(a), " gene families x ", ncol(a), " samples")

# align to filtered metadata
cn <- colnames(a)
sel <- cn %in% meta$sample_id
message("  samples matching filtered metadata: ", sum(sel), " / ", length(cn))
a <- a[, sel, drop = FALSE]

# drop all-zero features after subsetting (keeps file smaller; pipeline re-filters anyway)
rs <- Matrix::rowSums(a)
a <- a[rs > 0, , drop = FALSE]
message("  after sample-subset + drop-empty features: ", nrow(a), " x ", ncol(a))

# sanity: relative abundance -> fractional, colSums ~ per-sample stratified total
message("  value check: max=", signif(max(a), 4),
        " median colSum=", signif(median(colSums(a)), 4))

sp <- as(a, "CsparseMatrix")
local_mtx <- file.path("/workspace", "t2d_gene_families.mtx")
writeMM(sp, local_mtx)
writeLines(rownames(a), file.path(DATA, "t2d_gene_families_rows.txt"))
writeLines(colnames(a), file.path(DATA, "t2d_gene_families_cols.txt"))
dest <- file.path(DATA, "t2d_gene_families.mtx")
st <- system2("cp", c(local_mtx, dest))
message("  copied mtx -> ", dest, " (status ", st, ")")
message("[t2d] gene_families fetch DONE")
