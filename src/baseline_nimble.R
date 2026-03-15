library(nimble)

dat <- read.csv("simulated_data_N_20_C_6.csv")

count_cols <- grep("^C_", names(dat), value = TRUE)
Y <- as.matrix(dat[, count_cols])
true_N <- dat$N

nSites <- nrow(Y)
nVisits <- ncol(Y)

nmixture <- nimbleCode({
  # Priors
  log_lambda ~ dnorm(0, sd = 2.5)
  logit_p ~ dnorm(0, sd = 2.5)
  
  # Transform to natural scale
  lambda <- exp(log_lambda)
  p <- expit(logit_p)
  
  # Ecological process: true abundance at each site
  for (i in 1:nSites) {
    N[i] ~ dpois(lambda)
    
    # Observation process: counts given true abundance
    for (j in 1:nVisits) {
      C[i, j] ~ dbin(p, N[i])
    }
  }
  
  # Derived quantities
  totalN <- sum(N[1:nSites])
})

nimble_constants <- list(
  nSites = nSites,
  nVisits = nVisits
)

nimble_data <- list(
  C = Y
)

# Initialize N at max observed count per site (must be >= max count)
N_init <- apply(Y, 1, max)
N_init[N_init == 0] <- 1 # avoid zero initialization

nimble_inits <- list(
  lambda = mean(N_init),
  p = 0.3,
  N = N_init
)

model <- nimbleModel(
  code = nmixture,
  constants = nimble_constants,
  data = nimble_data,
  inits = nimble_inits
)

# Check the model is valid
model$calculate()
cat("Initial log probability:", model$calculate(), "\n\n")

mcmc_conf <- configureMCMC(model, monitors = c("lambda", "p", "totalN", "N"))

mcmc_conf$printSamplers()

mcmc <- buildMCMC(mcmc_conf)
compiled_model <- compileNimble(model)
compiled_mcmc <- compileNimble(mcmc, project = model)

# run the mcmc model
n_iter <- 50000
n_burn <- 10000
n_thin <- 1
n_chains <- 3

cat("Running MCMC:", n_iter, "iterations,", n_burn, "burn-in,", n_chains, "chains\n")

samples <- runMCMC(
  compiled_mcmc,
  niter = n_iter,
  nburnin = n_burn,
  thin = n_thin,
  nchains = n_chains,
  summary = TRUE,
  samplesAsCodaMCMC = TRUE
)

print(samples$summary$all.chains[c("lambda", "p", "totalN"), ])

N_cols <- paste0("N[", 1:nSites, "]")
N_summary <- samples$summary$all.chains[N_cols, ]

cat("\n====================================\n")
cat("SITE-LEVEL ABUNDANCE ESTIMATES\n")
cat("====================================\n")
comparison <- data.frame(
  site = 1:nSites,
  true_N = true_N,
  est_N_mean = round(N_summary[, "Mean"], 2),
  est_N_median = round(N_summary[, "Median"], 2),
  ci_lower = round(N_summary[, "95%CI_low"], 2),
  ci_upper = round(N_summary[, "95%CI_upp"], 2)
)
print(comparison)

est_lambda <- samples$summary$all.chains["lambda", "Mean"]
est_p <- samples$summary$all.chains["p", "Mean"]
est_totalN <- samples$summary$all.chains["totalN", "Mean"]
est_N_means <- N_summary[, "Mean"]