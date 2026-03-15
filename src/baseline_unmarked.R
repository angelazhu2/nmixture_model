library(unmarked)

dat <- read.csv("simulated_data.csv")

# Your count columns are C_1 through C_6
count_cols <- grep("^C_", names(dat), value = TRUE)
Y <- as.matrix(dat[, count_cols])

true_N <- dat$N

cat("Loaded", nrow(Y), "sites x", ncol(Y), "visits\n")
cat("True total N =", sum(true_N), "\n")

umf <- unmarkedFramePCount(y = Y)

# --- Fit N-mixture model ---
fit <- pcount(~1 ~1, data = umf, K = 50)

# --- Results ---
summary(fit)

# Back-transform lambda
bt_lambda <- backTransform(fit, type = "state")
bt_p <- backTransform(fit, type = "det")

# --- Site-specific abundance (empirical Bayes) ---
re <- ranef(fit)
eb_N <- bup(re)

comparison <- data.frame(
  true_N = true_N,
  estimated_N = round(eb_N, 2)
)
print(comparison)

cat("\nTotal true N      =", sum(true_N), "\n")
cat("Total estimated N =", round(sum(eb_N), 2), "\n")
cat("Mean true N       =", round(mean(true_N), 2), "\n")
cat("Mean estimated N  =", round(mean(eb_N), 2), "\n")

# --- Save results ---
cat("True Lambda:", 2, "  est. lam:", bt_lambda@estimate, "\n")
cat("True p:", 0.25, "        est. p:", bt_p@estimate, "\n")
cat("True Total Abundance:", sum(true_N), "\n")
cat("Estimated Total Abundance:", bt_lambda@estimate * nrow(Y), "\n")

summary_lines <- data.frame(
  key = c("true_lam", "est_lam", "true_p", "est_p",
          "true_avg_N", "est_avg_N", "true_total_N", "est_total_N", "N_mae"),
  value = c(2.0, bt_lambda@estimate, 0.25, bt_p@estimate,
            mean(true_N), mean(eb_N), sum(true_N), sum(eb_N),
            mean(abs(true_N - eb_N)))
)
write.table(summary_lines, "unmarked_summary.csv",
            sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)


# --- Save runs CSV (one row per site, easier to read) ---
runs_df <- data.frame(
  site = 1:nSites,
  true_N = true_N,
  est_N = round(eb_N, 2),
  lambda = bt_lambda@estimate,
  p = bt_p@estimate
)
write.csv(runs_df, "unmarked_lam2_p0.25_runs.csv", row.names = FALSE)