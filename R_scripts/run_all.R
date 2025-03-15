# Define the directory containing the R files
directory_path <- "path/to/your/Rscripts"

# List all files in the directory ending with .R
r_files <- list.files(path = directory_path, pattern = "\\.R$", full.names = TRUE)

# Remove the current script from the list if its path was determined
r_files <- r_files[basename(r_files) != "run_all.R"]

# Loop over each file and run it
for (file in r_files) {
  cat("Sourcing:", file, "\n")
  source(file)
}