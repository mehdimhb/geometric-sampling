library(BalancedSampling)
library(WaveSampling)

M <- 1000
input_folder <- "/home/divar/projects/geometric-sampling/data_samples/coords_probs"
output_folder <- "/home/divar/projects/geometric-sampling/data_samples/samples"

# Specify only the files you want to process:
selected_files <- c("swiss_uneq.csv")
شسبی
data_files <- file.path(input_folder, selected_files)

for (file_path in data_files) {
  dataname <- tools::file_path_sans_ext(basename(file_path))
  df <- read.csv(file_path)
  N <- nrow(df)
  n <- round(sum(df$prob))
  U <- seq_len(N)
  
  samples_cube <- matrix(NA_integer_, nrow = M, ncol = n)
  samples_wave <- matrix(NA_integer_, nrow = M, ncol = n)
  coords <- as.matrix(df[, c("x", "y")])
  
  for (i in 1:M) {
    cat("Dataset:", dataname, "- Iteration:", i, "\n")
    coords_jit <- jitter(coords, 1)
    
    # CUBE
    cube_ind <- cube(df$prob, coords_jit)
    cube_sample <- U[cube_ind == 1]
    len_cube <- length(cube_sample)
    if (len_cube < n) {
      add_sample <- sample(setdiff(U, cube_sample), n - len_cube, replace = FALSE)
      cube_sample <- c(cube_sample, add_sample)
    } else if (len_cube > n) {
      cube_sample <- sample(cube_sample, n, replace = FALSE)
    }
    samples_cube[i, ] <- cube_sample
    
    # WAVE
    wave_ind <- wave(coords_jit, df$prob)
    wave_sample <- U[wave_ind == 1]
    len_wave <- length(wave_sample)
    if (len_wave < n) {
      add_sample <- sample(setdiff(U, wave_sample), n - len_wave, replace = FALSE)
      wave_sample <- c(wave_sample, add_sample)
    } else if (len_wave > n) {
      wave_sample <- sample(wave_sample, n, replace = FALSE)
    }
    samples_wave[i, ] <- wave_sample
  }
  
  write.csv(
    samples_cube,
    file = file.path(output_folder, paste0(dataname, "_cube_samples.csv")),
    row.names = FALSE
  )
  write.csv(
    samples_wave,
    file = file.path(output_folder, paste0(dataname, "_wave_samples.csv")),
    row.names = FALSE
  )
  
  cat("Finished sampling and saving for dataset:", dataname, "\n")
}