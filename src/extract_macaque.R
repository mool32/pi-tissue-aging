#!/usr/bin/env Rscript
# Extract macaque aging multi-tissue count matrix from RData
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript extract_macaque.R <input.RData> <output_dir>\n")
  quit(status = 1)
}
input_file <- args[1]
output_dir <- args[2]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
cat("Loading RData file:", input_file, "\n")
env <- new.env()
load(input_file, envir = env)
cat("Objects found:\n")
for (name in ls(env)) {
  obj <- get(name, envir = env)
  cat(sprintf("  %s: %s", name, class(obj)[1]))
  if (is.matrix(obj) || is.data.frame(obj)) {
    cat(sprintf(" [%d x %d]", nrow(obj), ncol(obj)))
  } else if (is.list(obj)) {
    cat(sprintf(" [length %d]", length(obj)))
  }
  cat("\n")
}
for (name in ls(env)) {
  obj <- get(name, envir = env)
  if (is.matrix(obj) || is.data.frame(obj)) {
    outfile <- file.path(output_dir, paste0(name, ".csv"))
    if (nrow(obj) > 1000 || ncol(obj) > 1000) {
      outfile <- paste0(outfile, ".gz")
      gz <- gzfile(outfile, "w")
      write.csv(obj, gz, row.names = TRUE)
      close(gz)
    } else {
      write.csv(obj, outfile, row.names = TRUE)
    }
    cat(sprintf("  Saved %s [%d x %d]\n", name, nrow(obj), ncol(obj)))
  }
  if (is.list(obj) && !is.data.frame(obj)) {
    cat(sprintf("  List '%s' elements: %s\n", name, paste(names(obj), collapse = ", ")))
    for (subname in names(obj)) {
      subobj <- obj[[subname]]
      if (is.matrix(subobj) || is.data.frame(subobj)) {
        outfile <- file.path(output_dir, paste0(name, "_", subname, ".csv"))
        if (nrow(subobj) > 1000 || ncol(subobj) > 1000) {
          outfile <- paste0(outfile, ".gz")
          gz <- gzfile(outfile, "w")
          write.csv(subobj, gz, row.names = TRUE)
          close(gz)
        } else {
          write.csv(subobj, outfile, row.names = TRUE)
        }
        cat(sprintf("  Saved %s/%s [%d x %d]\n", name, subname, nrow(subobj), ncol(subobj)))
      }
    }
  }
}
cat("\nDone. Output in:", output_dir, "\n")
