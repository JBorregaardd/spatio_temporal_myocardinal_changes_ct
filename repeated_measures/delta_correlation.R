#################################################################################################################################################
# This script computes the correlation matrix of the changes (delta = CGPS10 - CGPS1) of the patient summary features.
# It reads <PROJECT_ROOT>/df/repeated_measures/deltas.csv, made by repeated_measures/deltas.py, and uses every delta_<feature> column.
# Each correlation uses the subjects with both deltas (pairwise complete observations).
# The matrix is saved in <PROJECT_ROOT>/df/repeated_measures/delta_correlation_<method>.csv and drawn with the coefficients in
# <PROJECT_ROOT>/reports/figures/repeated_measures/delta_correlation_<method>.png. Needs the R package corrplot.
# e.g. Rscript repeated_measures/delta_correlation.R pearson   (or spearman, default: pearson)
#################################################################################################################################################

if (!requireNamespace("corrplot", quietly = TRUE)) {
  stop("The R package corrplot is needed: install.packages(\"corrplot\")")
}

#' Project root: the parent of the folder this script is in.
#'
#' @return Absolute path of the project root.
project_root <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg))) else getwd()
  normalizePath(file.path(script_dir, ".."))
}

#' Correlation matrix of the delta columns.
#'
#' @param deltas Data frame with one row per subject and the columns delta_<feature>.
#' @param method Correlation method, "pearson" or "spearman".
#' @return Square matrix with one row and column per feature, named without the delta_ prefix.
delta_correlation <- function(deltas, method = "pearson") {
  values <- deltas[, grep("^delta_", names(deltas)), drop = FALSE]
  names(values) <- sub("^delta_", "", names(values))
  cor(values, use = "pairwise.complete.obs", method = method)
}

args <- commandArgs(trailingOnly = TRUE)
method <- if (length(args)) args[1] else "pearson"
if (!method %in% c("pearson", "spearman")) {
  stop(sprintf("Unknown method '%s', use pearson or spearman.", method))
}

root <- project_root()
deltas_path <- file.path(root, "df", "repeated_measures", "deltas.csv")
if (!file.exists(deltas_path)) {
  stop(sprintf("Deltas not found: %s\nRun repeated_measures/deltas.py first.", deltas_path))
}
deltas <- read.csv(deltas_path, colClasses = c(CGPS1 = "character", CGPS10 = "character"))
cat(sprintf("Read %d subjects from %s\n", nrow(deltas), deltas_path))

correlation <- delta_correlation(deltas, method)
print(round(correlation, 2))

table_path <- file.path(root, "df", "repeated_measures", sprintf("delta_correlation_%s.csv", method))
write.csv(round(correlation, 4), table_path)
cat(sprintf("Table saved to: %s\n", table_path))

figure_folder <- file.path(root, "reports", "figures", "repeated_measures")
dir.create(figure_folder, recursive = TRUE, showWarnings = FALSE)
figure_path <- file.path(figure_folder, sprintf("delta_correlation_%s.png", method))
png(figure_path, width = 2400, height = 2400, res = 300)
corrplot::corrplot(
  correlation,
  method = "color", type = "lower", addCoef.col = "black", number.cex = 0.6, tl.col = "black", tl.cex = 0.7,
  tl.srt = 45, col = corrplot::COL2("RdBu", 200), diag = FALSE, na.label = " ",
  title = sprintf("Correlation of the deltas (CGPS10 - CGPS1), %s, n = %d", method, nrow(deltas)),
  mar = c(0, 0, 2, 0)
)
invisible(dev.off())
cat(sprintf("Plot saved to: %s\n", figure_path))
