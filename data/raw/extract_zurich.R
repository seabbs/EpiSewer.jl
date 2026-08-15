# Recreation script for the Zurich SARS-CoV-2 example data packaged with
# EpiSewer.jl.
#
# The original EpiSewer R package source (gitignored) ships these as .rda
# files. This script reads them and writes clean CSVs under
# `data/example_zurich/` so the data can be loaded from Julia and used by the
# model components.
#
# Source .rda files (from .resources/EpiSewer/data/):
#   - SARS_CoV_2_Zurich.rda            -> measurements, flows, cases
#   - ww_assumptions_SARS_CoV_2_Zurich.rda -> discretised distribution PMFs
#
# Usage (from the package root):
#   Rscript data/raw/extract_zurich.R

options(scipen = 999)

src <- ".resources/EpiSewer/data"
out <- "data/example_zurich"
dir.create(out, showWarnings = FALSE, recursive = TRUE)

# --- measurements / flows / cases -------------------------------------------
load(file.path(src, "SARS_CoV_2_Zurich.rda"))
zurich <- SARS_CoV_2_Zurich

write_zurich_table <- function(df, fname) {
  df <- df
  df$date <- format(as.Date(df$date), "%Y-%m-%d")
  write.csv(df, file.path(out, fname), row.names = FALSE, quote = FALSE)
}

write_zurich_table(zurich$measurements, "measurements.csv")
write_zurich_table(zurich$flows, "flows.csv")
write_zurich_table(zurich$cases, "cases.csv")

# --- assumptions (discretised distributions) --------------------------------
# The assumption PMFs (generation / incubation / shedding load) are no longer
# stored: they are generated on demand by `EpiSewer.example_distributions()`
# from the continuous distributions via CensoredDistributions.jl
# (`double_interval_censored` + `Distributions.truncated`).
# The R parameter values used there match the original ww_assumptions object:
#   generation: shifted Gamma mean = 3, sd = 2.4
#   shedding load: Gamma shape = 0.929639, scale = 7.241397
#   incubation: Gamma shape = 8.5, scale = 0.4
load(file.path(src, "ww_assumptions_SARS_CoV_2_Zurich.rda"))
assum <- ww_assumptions_SARS_CoV_2_Zurich
cat("shedding_reference:", assum$shedding_reference, "\n")
cat("Wrote raw example data to", out, "\n")
