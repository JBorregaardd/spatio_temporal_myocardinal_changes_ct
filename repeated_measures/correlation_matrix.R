#################################################################################################################################################
# This script computes three correlation matrices of the patient summary features of the subjects linked between CGPS1 and CGPS10:
# the values at CGPS1 (<feature>_CGPS1), the values at CGPS10 (<feature>_CGPS10) and the changes (delta_<feature> = CGPS10 - CGPS1).
# It reads <PROJECT_ROOT>/df/repeated_measures/deltas.csv, made by repeated_measures/deltas.py, so all three use the same linked subjects.
# Each correlation uses the subjects with both values (pairwise complete observations).
# Each matrix is saved in <PROJECT_ROOT>/df/repeated_measures/correlation_<set>_<method>.csv and drawn with the coefficients in
# <PROJECT_ROOT>/reports/figures/repeated_measures/correlation_<set>_<method>.png, with <set> CGPS1, CGPS10 or delta. Needs the R package
# corrplot.
# e.g. Rscript repeated_measures/correlation_matrix.R pearson   (or spearman, default: pearson)
#################################################################################################################################################

if (!requireNamespace("corrplot", quietly = TRUE)) {
  stop("The R package corrplot is needed: install.packages(\"corrplot\")")
}

SETS <- list(
  CGPS1 = list(pattern = "_CGPS1$", title = "Correlation at CGPS1"),
  CGPS10 = list(pattern = "_CGPS10$", title = "Correlation at CGPS10"),
  delta = list(pattern = "^delta_", title = "Correlation of the deltas (CGPS10 - CGPS1)")
)

#' Project root: the parent of the folder this script is in.
#'
#' @return Absolute path of the project root.
project_root <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg))) else getwd()
  normalizePath(file.path(script_dir, ".."))
}

#' Correlation matrix of the feature columns matching a pattern.
#'
#' @param table Data frame with one row per subject and the feature columns.
#' @param pattern Regular expression selecting the columns; the matched part is removed from the names.
#' @param method Correlation method, "pearson" or "spearman".
#' @return Square matrix with one row and column per feature, named without the matched part.
feature_correlation <- function(table, pattern, method = "pearson") {
  values <- table[, grep(pattern, names(table)), drop = FALSE]
  if (!ncol(values)) {
    stop(sprintf("No columns match '%s'. Rerun repeated_measures/deltas.py.", pattern))
  }
  names(values) <- sub(pattern, "", names(values))
  cor(values, use = "pairwise.complete.obs", method = method)
}

#' Draw a correlation matrix with the coefficients in the lower triangle and save it as a PNG.
#'
#' @param correlation Square correlation matrix.
#' @param title Plot title.
#' @param path Output PNG path.
save_correlation_plot <- function(correlation, title, path) {
  png(path, width = 2400, height = 2400, res = 300)
  corrplot::corrplot(
    correlation,
    method = "color", type = "lower", addCoef.col = "black", number.cex = 0.6, tl.col = "black", tl.cex = 0.7,
    tl.srt = 45, col = corrplot::COL2("RdBu", 200), diag = FALSE, na.label = " ", title = title,
    mar = c(0, 0, 2, 0)
  )
  invisible(dev.off())
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

figure_folder <- file.path(root, "reports", "figures", "repeated_measures")
dir.create(figure_folder, recursive = TRUE, showWarnings = FALSE)

for (set in names(SETS)) {
  correlation <- feature_correlation(deltas, SETS[[set]]$pattern, method)
  cat(sprintf("\n%s (%s):\n", set, method))
  print(round(correlation, 2))

  table_path <- file.path(root, "df", "repeated_measures", sprintf("correlation_%s_%s.csv", set, method))
  write.csv(round(correlation, 4), table_path)
  cat(sprintf("Table saved to: %s\n", table_path))

  figure_path <- file.path(figure_folder, sprintf("correlation_%s_%s.png", set, method))
  title <- sprintf("%s, %s, n = %d", SETS[[set]]$title, method, nrow(deltas))
  save_correlation_plot(correlation, title, figure_path)
  cat(sprintf("Plot saved to: %s\n", figure_path))
}
