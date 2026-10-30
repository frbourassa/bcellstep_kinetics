#!/usr/bin/env Rscript

# ==============================================================================
# Calcium Imaging Data Processing
#
# Usage:
#   Rscript process_calcium_data.R <input_directory>
#
# Outputs:
#   <input_directory>/processed/
#       Individual processed CSV files
#       merged_ecdf.csv
#       merged_auc_peakMFI.csv
#       merged_timeData.csv
# ==============================================================================

# ------------------------------------------------------------------------------
# Packages
# ------------------------------------------------------------------------------

packages <- c("dplyr", "tidyr")

missing_packages <- packages[
  !sapply(packages, requireNamespace, quietly = TRUE)
]

if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

invisible(lapply(packages, library, character.only = TRUE))

# ------------------------------------------------------------------------------
# Command-line arguments
# ------------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop(
    "Usage: Rscript process_calcium_data.R <input_directory>",
    call. = FALSE
  )
}

dataDir <- normalizePath(args[1], mustWork = TRUE)
processedDir <- file.path(dataDir, "processed")

dir.create(
  processedDir,
  recursive = TRUE,
  showWarnings = FALSE
)

# ------------------------------------------------------------------------------
# Input files
# ------------------------------------------------------------------------------

dataFiles <- list.files(
  dataDir,
  pattern = "\\.csv$",
  full.names = TRUE
)

if (length(dataFiles) == 0) {
  stop(
    paste("No CSV files found in:", dataDir),
    call. = FALSE
  )
}

message("Found ", length(dataFiles), " CSV files.")
message("Output directory: ", processedDir)

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

# Extract condition and technical replicate from filename.
#
# Examples:
#   pHELD-AF594_1pM.csv   -> condition = pHELD-AF594_1pM, replicate = 1
#   pHELD-AF594_1pM_2.csv -> condition = pHELD-AF594_1pM, replicate = 2
#   Ionomycin.csv          -> condition = Ionomycin, replicate = 1

getFileInfo <- function(file) {
  filename <- tools::file_path_sans_ext(basename(file))

  tibble(
    condition = sub("_2$", "", filename),
    replicate = ifelse(grepl("_2$", filename), 2L, 1L)
  )
}

# Read TrackMate-style data and select relevant columns.
readRawData <- function(file) {
  read.csv(file)[-c(1:3), c(3, 9, 19)]
}

# Apply a function to every input file, add condition/replicate metadata,
# and combine results into one data frame.
mergeFiles <- function(files, processingFunction) {
  bind_rows(lapply(files, function(file) {
    info <- getFileInfo(file)

    processingFunction(file) %>%
      mutate(
        condition = info$condition,
        replicate = info$replicate
      )
  }))
}

# ------------------------------------------------------------------------------
# Individual Ca2+ traces
#
# Keep tracks with >=30 time points and select the 14 tracks with the largest
# CH2 intensity range.
# ------------------------------------------------------------------------------

processData <- function(file, n = 14) {
  df <- readRawData(file) %>%
    group_by(TRACK_ID) %>%
    filter(n() >= 30) %>%
    mutate(
      mean_CH2 = as.numeric(MEAN_INTENSITY_CH2),
      timepoint = 2 * as.numeric(FRAME),
      peak_CH2 = max(mean_CH2, na.rm = TRUE),
      range = max(mean_CH2, na.rm = TRUE) - min(mean_CH2, na.rm = TRUE)
    ) %>%
    rename(track_id = TRACK_ID) %>%
    dplyr::select(track_id, timepoint, mean_CH2, peak_CH2, range) %>%
    ungroup()

  top_tracks <- df %>%
    group_by(track_id) %>%
    summarise(
      range = max(range, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(range)) %>%
    slice_head(n = n)

  df %>%
    filter(track_id %in% top_tracks$track_id) %>%
    arrange(desc(range))
}

message("Processing individual Ca2+ traces...")

for (file in dataFiles) {
  output_file <- file.path(
    processedDir,
    basename(file)
  )

  write.csv(
    processData(file, n = 14),
    output_file,
    row.names = FALSE
  )
}

# ------------------------------------------------------------------------------
# eCDF data
#
# Keep tracks with >=30 time points and classify responders using CH2
# intensity range >300.
# ------------------------------------------------------------------------------

filterData <- function(file) {
  readRawData(file) %>%
    group_by(TRACK_ID) %>%
    filter(n() >= 30) %>%
    mutate(
      mean_CH2 = as.numeric(MEAN_INTENSITY_CH2),
      timepoint = 2 * as.numeric(FRAME),
      peak_CH2 = max(mean_CH2, na.rm = TRUE),
      range = max(mean_CH2, na.rm = TRUE) - min(mean_CH2, na.rm = TRUE)
    ) %>%
    rename(track_id = TRACK_ID) %>%
    dplyr::select(track_id, timepoint, mean_CH2, peak_CH2, range) %>%
    filter(range > 300) %>%
    ungroup()
}

message("Generating merged eCDF data...")

merged_ecdf <- mergeFiles(
  dataFiles,
  filterData
)

write.csv(
  merged_ecdf,
  file.path(processedDir, "merged_ecdf.csv"),
  row.names = FALSE
)

# ------------------------------------------------------------------------------
# Area Under Curve and Peak MFI
#
# Keep tracks with >=30 time points.
# ------------------------------------------------------------------------------

readData <- function(file) {
  readRawData(file) %>%
    group_by(TRACK_ID) %>%
    filter(n() >= 30) %>%
    mutate(
      mean_CH2 = as.numeric(MEAN_INTENSITY_CH2),
      timepoint = 2 * as.numeric(FRAME),
      area = sum(mean_CH2, na.rm = TRUE) * 2,
      max_MFI = max(mean_CH2, na.rm = TRUE)
    ) %>%
    rename(track_id = TRACK_ID) %>%
    summarise(
      area = first(area),
      max_MFI = first(max_MFI),
      .groups = "drop"
    )
}

message("Calculating AUC and peak MFI...")

merged_auc_peakMFI <- mergeFiles(
  dataFiles,
  readData
)

write.csv(
  merged_auc_peakMFI,
  file.path(processedDir, "merged_auc_peakMFI.csv"),
  row.names = FALSE
)

# ------------------------------------------------------------------------------
# Time to Ca2+ flux
#
# Define flux as CH2 intensity reaching >=1.5x the initial CH2 intensity.
# Tracks that never cross this threshold are classified as nonresponders.
# ------------------------------------------------------------------------------

timeData <- function(file) {
  df <- readRawData(file) %>%
    group_by(TRACK_ID) %>%
    filter(n() >= 30) %>%
    mutate(
      mean_CH2 = as.numeric(MEAN_INTENSITY_CH2),
      timepoint = 2 * as.numeric(FRAME),
      peak_CH2 = max(mean_CH2, na.rm = TRUE),
      range = max(mean_CH2, na.rm = TRUE) - min(mean_CH2, na.rm = TRUE),
      initial_CH2 = first(mean_CH2),
      flux = ifelse(mean_CH2 >= 1.5 * initial_CH2, 1, 0)
    ) %>%
    rename(track_id = TRACK_ID) %>%
    dplyr::select(
      track_id,
      timepoint,
      mean_CH2,
      peak_CH2,
      range,
      initial_CH2,
      flux
    ) %>%
    ungroup()

  time <- df %>%
    filter(flux == 1) %>%
    group_by(track_id) %>%
    summarise(
      time_to_flux = min(timepoint),
      .groups = "drop"
    )

  nonresponders <- df %>%
    filter(!(track_id %in% time$track_id)) %>%
    distinct(track_id) %>%
    mutate(time_to_flux = NA_real_)

  alltime <- bind_rows(
    time,
    nonresponders
  ) %>%
    arrange(time_to_flux)

  alltime$fraction_fluxed <- vapply(
    alltime$time_to_flux,
    function(t) {
      if (is.na(t)) {
        NA_real_
      } else {
        sum(alltime$time_to_flux <= t, na.rm = TRUE) / nrow(alltime)
      }
    },
    numeric(1)
  )

  alltime
}

message("Calculating time-to-flux data...")

merged_timeData <- mergeFiles(
  dataFiles,
  timeData
) %>%
  filter(!is.na(time_to_flux))

write.csv(
  merged_timeData,
  file.path(processedDir, "merged_timeData.csv"),
  row.names = FALSE
)

message("Processing complete.")
message("Results written to: ", processedDir)