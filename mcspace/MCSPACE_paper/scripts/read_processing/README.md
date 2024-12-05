# Processing SAMPL-seq sequencing data
These scripts can be used to obtain the processed sequence count data for concensus OTUs from raw sequencing data. The sequencing data for the mouse study can be obtained from the SRA: PRJNA1182308. Preprocessed data is also available in the `MCSPACE_paper/datasets` repo.

## Step 1: Install required software
To install the required bioinformatics packages for processing SAMPL-seq data, execute the script,
```bash
bash install_packages.sh
```

## Step 2: Filter reads, extract barcodes, and map reads to OTUs:
To perform the steps to filter reads, extract particle barcodes, and map reads to OTUs, execute the script
```bash
bash BarcodeingProcessing_nthreads_final.sh
```

## Step 3: Barcode error correction
To perform the barcode error correction step and obtain the final processed count table, first run the R script `Biom File Melting.R`, then run the R script `Barcode Correction.R`
