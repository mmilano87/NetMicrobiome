#!/usr/bin/env Rscript
# fetch_t2d_pathways.R
# Fetch the T2D (MetaCardis_2020_a) pathway_abundance table (MetaCyc pathways x samples),
# matching how ACVD/IBD pathway_abundance were produced:
#   - HUMAnN pathway_abundance table, counts = FALSE (relative abundance; CLR downstream)
#   - rownames = "long"
#   - columns aligned to the already-filtered t2d_metadata.tsv (T2D+control, gender known)
# Writes to /mnt/shared-workspace/micro/data/:
#   t2d_pathway_abundance.mtx / _rows.txt / _cols.txt
suppressMessages({
  library(Matrix); library(curatedMetagenomicData); library(SummarizedExperiment)
})

DATA <- "/mnt/shared-workspace/micro/data"
resource <- "2022-10-19.MetaCardis_2020_a.pathway_abundance"

## ---- 1. reuse the SAME filtered metadata used for the gene-family fetch ----
meta <- read.delim(file.path(DATA, "t2d_metadata.tsv"),
                   stringsAsFactors = FALSE, check.names = FALSE)
message("[t2d] metadata: ", nrow(meta), " samples")
print(table(meta$study_condition, meta$gender))

## ---- 2. fetch pathway_abundance relative abundance ----
message("[t2d] fetching pathway_abundance (counts=FALSE): ", resource)
tse <- curatedMetagenomicData(resource, dryrun = FALSE, counts = FALSE,
                              rownames = "long")[[1]]
a <- assay(tse)
message("  fetched: ", nrow(a), " pathway rows x ", ncol(a), " samples")

cn <- colnames(a)
sel <- cn %in% meta$sample_id
message("  samples matching filtered metadata: ", sum(sel), " / ", length(cn))
a <- a[, sel, drop = FALSE]

rs <- Matrix::rowSums(a)
a <- a[rs > 0, , drop = FALSE]
message("  after sample-subset + drop-empty rows: ", nrow(a), " x ", ncol(a))
message("  value check: max=", signif(max(a), 4),
        " median colSum=", signif(median(colSums(a)), 4))

sp <- as(a, "CsparseMatrix")
local_mtx <- file.path("/workspace", "t2d_pathway_abundance.mtx")
writeMM(sp, local_mtx)
writeLines(rownames(a), file.path(DATA, "t2d_pathway_abundance_rows.txt"))
writeLines(colnames(a), file.path(DATA, "t2d_pathway_abundance_cols.txt"))
dest <- file.path(DATA, "t2d_pathway_abundance.mtx")
st <- system2("cp", c(local_mtx, dest))
message("  copied mtx -> ", dest, " (status ", st, ")")
message("[t2d] pathway_abundance fetch DONE")
