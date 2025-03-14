# Define the directory containing the R files
directory_path <- "path/to/your/Rscripts"

# List all files in the directory ending with .R
r_files <- list.files(path = directory_path, pattern = "\\.R$", full.names = TRUE)

# Loop over each file and run it
for (file in r_files) {
  cat("Sourcing:", file, "\n")
  source(file)
}