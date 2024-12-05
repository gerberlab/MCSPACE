THREADS=30
# housekeeping

# this script requires a bin of software/resources, including usearch/vsearch and the primer sequences in the local directory
# the primer sequences are included on github, but I cannot distribute usearch without a license

# we put all fastqs to process in the fastq folder, everything else should work from there

#### setup your new machine with all the requisite software ####
# I use an ubuntu 18.04 machine


# USEARCH is not included here but can be found at https://www.drive5.com/usearch/. Is in TUMs AMI too

#Remove R2 if present
#rm fastq/*R2.fastq.pigz

#### Processing Samples ####



#First, replace '_' in samplenames, but keep it at the '_R'. Otherwise, usearch will remove from files

for file in fastq/*; do
    if [[ "$file" == *_* ]]; then
        mv "$file" "${file//_/-}"
    fi
done

#but return _R
for file in fastq/*; do
    if [[ "$file" == *-R* ]]; then
        mv "$file" "${file//-R/_R}"
    fi
done

#unzip fastqs for usearch
unpigz -p $THREADS ./fastq/*

# filter reads from individual samples
foo () {
    local FILENAME=$1
    OUTFILE="$(echo ${FILENAME} | sed 's/_.*$//')_filtered.fastq";
   ./usearch10 -fastq_filter "${FILENAME}" -fastq_maxee 1.0 -fastq_minlen 140 -relabel @ -fastqout "${OUTFILE}" -threads $THREADS
}

for run in $(find ./fastq/*R1* -type f); do foo "$run" & done


#remove the original fastqs
rm ./fastq/*_R1.fastq

# zip filtered fastqs for ultraplex
pigz -p $THREADS ./fastq/*_filtered.fastq

#### extracting unique particles from barcodes MASSIVELY FASTER NOWWWWW ####
#unpacking sequencing files
find ./fastq -name "*_filtered.fastq.gz" -type f | while read FILENAME ; do
    OUTFILE="$(echo ${FILENAME} | sed 's/_.*$//')_filtered_particleid.fastq.gz";
	
	#ultraplex for 7,8,9 bp leading barcodes (it can't do all at once...)
	# this could be solved by padding those with 7 and 8 bp with pads, but that is probably slower
	# minimum length 86bp, 32 threads, and minimum qscore 20 
	ultraplex -i "${FILENAME}" -b ./bin/BarcodeCSV7leadingfull.csv -l 82 -inm -t $THREADS -dbr -d fastq/first7bp -q 20
	ultraplex -i "${FILENAME}" -b ./bin/BarcodeCSV8leadingfull.csv -l 82 -inm -t $THREADS -dbr -d fastq/first8bp -q 20
	ultraplex -i "${FILENAME}" -b ./bin/BarcodeCSV9leadingfull.csv -l 82 -inm -t $THREADS -dbr -d fastq/first9bp -q 20

	#combine them
	echo "Combining Files"
	zcat ./fastq/*first*bp/ultraplex*.fastq.gz | pigz -p $THREADS > ./fastq/AllLengths.fastq.gz
	
	# strip 16S primer (the degenerate bases don't play well with seqkit)
	echo "Removing Primers"
	seqkit amplicon  ./fastq/AllLengths.fastq.gz -F YCAGCMGCCGCGGTAA -r 1:140 -f  -j $THREADS -o "${OUTFILE}";
	
	echo "Cleaning Up"
	# remove intermediates file (could be removed by piping this into the earlier command pipe?)
	rm ./fastq/AllLengths.fastq.gz
	
	# get rid of directory with the intermediate files
	rm ./fastq/first* -rf
	
	echo "Done"
done


#combine fastqs into one sample in new directory (fast)
mkdir otu
zcat fastq/*_filtered_particleid.fastq.gz | pigz -p $THREADS > otu/filt_all_out.fastq.gz

# fix the read names so usearch/vsearch will understand the samples (slow)
seqkit replace otu/filt_all_out.fastq.gz -p "\.[0-9]+rbc:" -r 'rbc' -o otu/filt_all_particleid.fastq.gz

# check the length
seqkit stats otu/filt_all_particleid.fastq.gz

# remove short reads (slow) mostly fixed by increasing ultraplex length (still got some weird short reads)
seqkit seq otu/filt_all_particleid.fastq.gz -m 69 -o otu/69bpfilt_all_particleid.fastq.gz -j $THREADS -g

# trim to 69bp (slow but not too bad)
seqkit subseq  otu/69bpfilt_all_particleid.fastq.gz  -r 1:69 -j $THREADS -o otu/final_filt_all_particleid.fastq.gz

#convert to fasta for VSEARCH clustering (fast)
seqkit fq2fa otu/final_filt_all_particleid.fastq.gz -o otu/filt_all.fa -j $THREADS

# unzip your fastqs to make them useable with usearch
unpigz otu/final_filt_all_particleid.fastq.gz

#### OTU Clustering ####
# remove doubles (old way reduces otus by a lot ~70%)
./usearch10 -fastq_filter otu/final_filt_all_particleid.fastq -fastq_maxee 0.1 -relabel @ -fastqout otu/filt_all_double.fq -threads $THREADS

# get unique sequences
./usearch10 -fastx_uniques otu/filt_all_double.fq -sizeout -relabel Uniq -fastaout otu/uniques.fa -threads $THREADS

# cluster otus
./usearch10 -unoise3 otu/uniques.fa -zotus otu/zotus.fa

# change Zotu to Otu
sed -i 's/Zotu/Otu/g' otu/zotus.fa

# make final files
mkdir ProcessedResults

# make udb (supposed to speed up search)
vsearch --makeudb_usearch ./otu/zotus.fa --output ./otu/zotus.udb

# need to replace '-' characters with 'x' or it messes up filenames
sed 's/-/x/g' ./otu/filt_all.fa  > ./otu/filt_all_x.fa 

#do assignment with vsearch (much faster than usearch)
vsearch --usearch_global ./otu/filt_all_x.fa --db ./otu/zotus.udb --id 0.9 --otutabout ./ProcessedResults/otu_frequency_table.tsv --biomout ./ProcessedResults/otu_frequency_table.biom --threads $THREADS

# Now return 'x' characters back to '-' 
sed 's/x/-/g' ./ProcessedResults/otu_frequency_table.tsv  > ./ProcessedResults/otu_frequency_table.tsv

# add otus to ProcessedResults
cp otu/zotus.fa ProcessedResults/

#compress the frequency table, as it is often large
pigz ./ProcessedResults/otu_frequency_table.tsv

#process taxonomy using rdp
./usearch10 -sintax ./otu/zotus.fa -db tax_rdp/rdp_16s_v18.udb -tabbedout tax_rdp/reads.sintax -strand both -sintax_cutoff 0.8

#format table nicely
#may need to install pandas with 'pip install pandas'
python tax_rdp/process_tax.py --sintax tax_rdp/reads.sintax --taxtable ProcessedResults/tax.csv
