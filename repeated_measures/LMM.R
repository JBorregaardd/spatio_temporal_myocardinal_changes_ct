#######################################################################################################################
# This script fits a linear mixed model (LMM) of one patient summary measurement, by default lvm_average_thickness_mm,
# for the subjects scanned in CGPS1 (baseline) and CGPS10 (10-year follow-up):
#     value ~ time * HT * sex + age_c + (1 | subject)
# time is 0 at CGPS1 and 1 at CGPS10, HT is the hypertrophy class of the subject's CGPS1 scan
# (df/repeated_measures/CGPS1_HT.csv, made by ht_classifier.py), sex is from CGPS1 (else CGPS10) and age_c is the CGPS1
# age centred on its mean. The random intercept models that the two scans of a subject are correlated. The time terms
# answer: does the measurement change (time), does the change differ between HT and no HT (time:HT) and does that
# difference differ between women and men (time:HT:sex).
# Linked subjects with a value at only one examination are kept, as the LMM can use them (missing at random). With
# --dropouts the CGPS1 subjects not in the links table are added with their baseline value only.
# As a sensitivity analysis the subjects with both values are also fitted with an ANCOVA:
# CGPS10 ~ CGPS1 + HT * sex + age_c. As HT is defined from the CGPS1 thickness, the change in thickness is affected by
# regression to the mean, and the LMM (change scores) and the ANCOVA can disagree (Lord's paradox).
# The values are read from output/<dataset>_<mm>mm/statistics/statistics_patient_summary.csv of each dataset and the
# subjects are linked by df/CGPS1_CGPS10_links.csv. The tables are saved in df/repeated_measures/LMM_<measurement>_*.csv
# and the figures in reports/figures/repeated_measures/LMM_<measurement>_*.png. Uses the R package nlme.
# e.g. Rscript repeated_measures/LMM.R --measurement lvm_g --aorta-exclusion-mm 0 --dropouts
#######################################################################################################################
suppressPackageStartupMessages(library(nlme))

DATASETS <- c("CGPS1", "CGPS10")
HT_LEVELS <- c("No HT", "HT")
SEX_LEVELS <- c("Men", "Women")
FIXED <- value ~ time * HT * sex + age_c

#' Project root: the parent of the folder this script is in.
#'
#' @return Absolute path of the project root.
project_root <- function() {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg))) else getwd()
  normalizePath(file.path(script_dir, ".."))
}

#' Parse the command line arguments.
#'
#' @param args Arguments after the script name.
#' @return List with measurement, aorta_exclusion_mm and dropouts.
parse_args <- function(args) {
  options <- list(measurement = "lvm_average_thickness_mm", aorta_exclusion_mm = 0, dropouts = FALSE)
  i <- 1
  while (i <= length(args)) {
    if (args[i] == "--measurement") {
      options$measurement <- args[i + 1]
      i <- i + 2
    } else if (args[i] == "--aorta-exclusion-mm") {
      options$aorta_exclusion_mm <- as.numeric(args[i + 1])
      i <- i + 2
    } else if (args[i] == "--dropouts") {
      options$dropouts <- TRUE
      i <- i + 1
    } else {
      stop(sprintf("Unknown argument '%s'. Use --measurement, --aorta-exclusion-mm and --dropouts.", args[i]))
    }
  }
  options
}

#' Patient summary of one dataset, written by main_statistics_df.py.
#'
#' @param root Project root.
#' @param dataset Dataset name, e.g. "CGPS1".
#' @param aorta_exclusion_mm Aortic exclusion distance of the run.
#' @return Data frame with patient_id as character.
read_summary <- function(root, dataset, aorta_exclusion_mm) {
  path <- file.path(root, "output", sprintf("%s_%gmm", dataset, aorta_exclusion_mm), "statistics",
                    "statistics_patient_summary.csv")
  if (!file.exists(path)) {
    stop(sprintf("Patient summary not found: %s\nRun main_statistics_df.py for %s first.", path, dataset))
  }
  cat(sprintf("%s: %s\n", dataset, path))
  summary <- read.csv(path, colClasses = c(patient_id = "character"))
  summary[!duplicated(summary$patient_id), ]
}

#' Long table with one row per subject and examination with a value.
#'
#' @param summaries Named list (CGPS1, CGPS10) of patient summaries.
#' @param links Data frame with the patient ids in the columns CGPS1 and CGPS10.
#' @param ht Data frame with patient_id and HT of the CGPS1 patients.
#' @param measurement Column of the measurement.
#' @param dropouts Add the CGPS1 subjects not in the links table.
#' @return Data frame with subject, time, value, HT, sex, age and linked (the subject is in the links table).
long_table <- function(summaries, links, ht, measurement, dropouts) {
  subjects <- unique(links[, DATASETS])
  subjects$linked <- TRUE
  if (dropouts) {
    unlinked <- setdiff(summaries$CGPS1$patient_id, subjects$CGPS1)
    subjects <- rbind(subjects, data.frame(CGPS1 = unlinked, CGPS10 = NA_character_, linked = FALSE))
  }
  subjects$subject <- ifelse(is.na(subjects$CGPS10), subjects$CGPS1, paste(subjects$CGPS1, subjects$CGPS10, sep = "/"))

  baseline <- summaries$CGPS1[match(subjects$CGPS1, summaries$CGPS1$patient_id), ]
  follow_up <- summaries$CGPS10[match(subjects$CGPS10, summaries$CGPS10$patient_id), ]
  sex <- ifelse(is.na(baseline$sex), follow_up$sex, baseline$sex)
  ht_class <- as.character(ht$HT[match(subjects$CGPS1, ht$patient_id)])
  ht_class <- ifelse(toupper(ht_class) == "TRUE", "HT", ifelse(toupper(ht_class) == "FALSE", "No HT", NA))

  rows <- lapply(seq_along(DATASETS), function(i) {
    values <- if (i == 1) baseline[[measurement]] else follow_up[[measurement]]
    data.frame(subject = subjects$subject, time = i - 1, value = values, HT = ht_class,
               sex = ifelse(sex == 1, "Women", ifelse(sex == 0, "Men", NA)), age = baseline$age,
               linked = subjects$linked)
  })
  data <- do.call(rbind, rows)
  data <- data[!is.na(data$value), ]
  data$HT <- factor(data$HT, levels = HT_LEVELS)
  data$sex <- factor(data$sex, levels = SEX_LEVELS)
  data
}

#' Estimates, standard errors, t-tests and 95% confidence intervals of linear combinations of the fixed effects.
#'
#' The degrees of freedom of a contrast are the smallest of those of the fixed effects it uses, so contrasts of the
#' change use the within-subject df and contrasts at one examination the between-subject df.
#'
#' @param fit Fitted lme model.
#' @param contrasts Matrix with one named row per contrast and one column per fixed effect.
#' @return Data frame with contrast, estimate, se, df, t, p, ci_low and ci_high.
linear_contrasts <- function(fit, contrasts) {
  df <- apply(contrasts, 1, function(weights) min(fit$fixDF$X[weights != 0]))
  estimate <- as.vector(contrasts %*% fixef(fit))
  se <- sqrt(diag(contrasts %*% vcov(fit) %*% t(contrasts)))
  t_value <- estimate / se
  data.frame(contrast = rownames(contrasts), estimate = estimate, se = se, df = df, t = t_value,
             p = 2 * pt(-abs(t_value), df), ci_low = estimate - qt(0.975, df) * se,
             ci_high = estimate + qt(0.975, df) * se, row.names = NULL)
}

#' Model matrix rows of the fixed effects for each HT class, sex and time, at the mean age.
#'
#' @return Data frame of the grid and the matching model matrix in the attribute "X".
design_grid <- function() {
  grid <- expand.grid(time = 0:1, HT = factor(HT_LEVELS, levels = HT_LEVELS),
                      sex = factor(SEX_LEVELS, levels = SEX_LEVELS))
  grid$age_c <- 0
  attr(grid, "X") <- model.matrix(delete.response(terms(FIXED)), grid)
  grid
}

#' Contrasts of the change from CGPS1 to CGPS10 per group and of the differences in change between the groups.
#'
#' @param grid Grid from design_grid.
#' @return Matrix with one named row per contrast.
change_contrasts <- function(grid) {
  X <- attr(grid, "X")
  row_of <- function(time, ht, sex) X[grid$time == time & grid$HT == ht & grid$sex == sex, ]
  change <- function(ht, sex) row_of(1, ht, sex) - row_of(0, ht, sex)
  rows <- list()
  for (sex in SEX_LEVELS) {
    for (ht in HT_LEVELS) {
      rows[[sprintf("Change, %s, %s", sex, ht)]] <- change(ht, sex)
    }
  }
  for (sex in SEX_LEVELS) {
    rows[[sprintf("Change HT - change no HT, %s", sex)]] <- change("HT", sex) - change("No HT", sex)
    rows[[sprintf("Baseline HT - baseline no HT, %s", sex)]] <- row_of(0, "HT", sex) - row_of(0, "No HT", sex)
  }
  rows[["(Change HT - change no HT), women - men"]] <-
    (change("HT", "Women") - change("No HT", "Women")) - (change("HT", "Men") - change("No HT", "Men"))
  do.call(rbind, rows)
}

#' ANCOVA of the follow-up value on the baseline value for the subjects with both values.
#'
#' @param data Long table from long_table.
#' @return Data frame with the coefficients, standard errors, t-tests, p-values and 95% confidence intervals.
ancova <- function(data) {
  wide <- reshape(data[, c("subject", "time", "value", "HT", "sex", "age_c")],
                  idvar = c("subject", "HT", "sex", "age_c"), timevar = "time", direction = "wide")
  wide <- wide[complete.cases(wide), ]
  fit <- lm(value.1 ~ value.0 + HT * sex + age_c, data = wide)
  coefficients <- summary(fit)$coefficients
  intervals <- confint(fit)
  data.frame(term = rownames(coefficients), estimate = coefficients[, 1], se = coefficients[, 2],
             t = coefficients[, 3], p = coefficients[, 4], ci_low = intervals[, 1], ci_high = intervals[, 2],
             n = nrow(wide), row.names = NULL)
}

#' Plot of the model estimated means with 95% confidence intervals per examination, HT class and sex.
#'
#' One panel per sex with a line per HT class from CGPS1 to CGPS10, drawn with base R graphics on the open device.
#'
#' @param means Data frame with time, HT, sex, estimate, ci_low and ci_high.
#' @param measurement Name of the measurement for the axis label.
means_plot <- function(means, measurement) {
  colours <- c("No HT" = "#1f77b4", "HT" = "#d62728")
  offsets <- c("No HT" = -0.04, "HT" = 0.04)
  ylim <- range(means$ci_low, means$ci_high)
  ylim[2] <- ylim[2] + 0.25 * diff(ylim)
  par(mfrow = c(1, length(SEX_LEVELS)), mar = c(3, 4.5, 2, 1), oma = c(0, 0, 3.5, 0))
  for (sex in SEX_LEVELS) {
    plot(NA, xlim = c(-0.3, 1.3), ylim = ylim, xaxt = "n", xlab = "", ylab = measurement, main = sex)
    axis(1, at = 0:1, labels = DATASETS)
    for (ht in HT_LEVELS) {
      group <- means[means$sex == sex & means$HT == ht, ]
      group <- group[order(group$time), ]
      x <- group$time + offsets[[ht]]
      lines(x, group$estimate, col = colours[[ht]], lwd = 2)
      points(x, group$estimate, col = colours[[ht]], pch = 16)
      arrows(x, group$ci_low, x, group$ci_high, angle = 90, code = 3, length = 0.04, col = colours[[ht]])
    }
    legend("topleft", legend = HT_LEVELS, col = colours[HT_LEVELS], lwd = 2, pch = 16, bty = "n")
  }
  mtext(sprintf("LMM estimated means of %s (95%% CI)", measurement), outer = TRUE, line = 2, cex = 1.1, font = 2)
  mtext("At the mean baseline age; HT class from the CGPS1 scan", outer = TRUE, line = 0.7, cex = 0.85)
}

if (sys.nframe() == 0) {
  options <- parse_args(commandArgs(trailingOnly = TRUE))
  root <- project_root()
  measurement <- options$measurement

  links_path <- file.path(root, "df", "CGPS1_CGPS10_links.csv")
  ht_path <- file.path(root, "df", "repeated_measures", "CGPS1_HT.csv")
  if (!file.exists(links_path)) {
    stop(sprintf("Links table not found: %s\nRun comparative_analysis/compare.py first.", links_path))
  }
  if (!file.exists(ht_path)) {
    stop(sprintf("HT classes not found: %s\nRun repeated_measures/ht_classifier.py first.", ht_path))
  }
  links <- read.csv(links_path, colClasses = c(CGPS1 = "character", CGPS10 = "character"))
  ht <- read.csv(ht_path, colClasses = c(patient_id = "character", HT = "character"))
  summaries <- setNames(lapply(DATASETS, function(dataset) read_summary(root, dataset, options$aorta_exclusion_mm)),
                        DATASETS)
  for (dataset in DATASETS) {
    if (!measurement %in% names(summaries[[dataset]])) {
      stop(sprintf("Column '%s' not in the %s summary.", measurement, dataset))
    }
  }

  data <- long_table(summaries, links, ht, measurement, options$dropouts)
  incomplete <- !complete.cases(data[, c("HT", "sex", "age")])
  if (any(incomplete)) {
    cat(sprintf("[WARN] %d subjects without HT class, sex or age are left out.\n",
                length(unique(data$subject[incomplete]))))
  }
  data <- data[!incomplete, ]
  data$age_c <- data$age - mean(data$age[data$time == 0])
  scans <- table(data$subject)
  cat(sprintf("%d subjects: %d with both examinations, %d with one (%d not linked).\n", length(scans), sum(scans == 2),
              sum(scans == 1), length(unique(data$subject[!data$linked]))))
  print(with(unique(data[, c("subject", "HT", "sex")]), table(HT, sex)))

  fit <- lme(FIXED, random = ~ 1 | subject, data = data, method = "REML")
  cat("\nFixed effects:\n")
  print(summary(fit)$tTable)
  variances <- as.numeric(VarCorr(fit)[, "Variance"])
  cat(sprintf("\nIntra-class correlation (between-subject share of the variance): %.2f\n",
              variances[1] / sum(variances)))

  type3 <- anova(fit, type = "marginal")
  cat("\nMarginal F-tests:\n")
  print(type3)

  grid <- design_grid()
  contrasts <- linear_contrasts(fit, change_contrasts(grid))
  cat("\nChange from CGPS1 to CGPS10 at the mean baseline age:\n")
  print(contrasts, digits = 3)

  sensitivity <- ancova(data)
  cat(sprintf("\nANCOVA, CGPS10 ~ CGPS1 + HT * sex + age_c (n = %d):\n", sensitivity$n[1]))
  print(sensitivity[, -ncol(sensitivity)], digits = 3)

  table_folder <- file.path(root, "df", "repeated_measures")
  figure_folder <- file.path(root, "reports", "figures", "repeated_measures")
  dir.create(table_folder, recursive = TRUE, showWarnings = FALSE)
  dir.create(figure_folder, recursive = TRUE, showWarnings = FALSE)
  prefix <- sprintf("LMM_%s", measurement)
  fixed <- as.data.frame(summary(fit)$tTable)
  write.csv(cbind(term = rownames(fixed), fixed), file.path(table_folder, sprintf("%s_fixed_effects.csv", prefix)),
            row.names = FALSE)
  write.csv(cbind(term = rownames(type3), as.data.frame(type3)),
            file.path(table_folder, sprintf("%s_anova.csv", prefix)), row.names = FALSE)
  write.csv(contrasts, file.path(table_folder, sprintf("%s_contrasts.csv", prefix)), row.names = FALSE)
  write.csv(sensitivity, file.path(table_folder, sprintf("%s_ancova.csv", prefix)), row.names = FALSE)
  cat(sprintf("\nTables saved to: %s/%s_*.csv\n", table_folder, prefix))

  X <- attr(grid, "X")
  rownames(X) <- seq_len(nrow(X))
  means <- cbind(grid, linear_contrasts(fit, X))
  png(file.path(figure_folder, sprintf("%s_means.png", prefix)), width = 2400, height = 1500, res = 300)
  means_plot(means, measurement)
  invisible(dev.off())

  png(file.path(figure_folder, sprintf("%s_diagnostics.png", prefix)), width = 2400, height = 1200, res = 300)
  par(mfrow = c(1, 2))
  residuals_fit <- residuals(fit, type = "normalized")
  plot(fitted(fit), residuals_fit, pch = 16, cex = 0.4, xlab = "Fitted", ylab = "Normalised residual",
       main = "Residuals vs fitted")
  abline(h = 0, lty = 2)
  qqnorm(residuals_fit, pch = 16, cex = 0.4, main = "Normal Q-Q of the residuals")
  qqline(residuals_fit)
  invisible(dev.off())
  cat(sprintf("Figures saved to: %s/%s_{means,diagnostics}.png\n", figure_folder, prefix))
}
