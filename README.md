# DeepClone: A New Approach for Inferring Cancer Subclone Populations Using Neural Networks
Yang Guo, Peng Nie*
***
## 
## 1. Data Preprocessing
The [GATK4](https://github.com/broadinstitute/gatk) is used as a method for detecting base mutations during the preprocessing stage. 
### 1.1 Calling Somatic SNP whit Tumor and Normal samples
If the samples include matched normal and tumor sequencing data, it is recommended to use Mutect2 for somatic mutation detection.
```bash
    # Run Mutect2 for somatic mutation detection
    gatk Mutect2 
    -R reference.fasta  # Reference genome file
    -I tumor.bam  # BAM file of the tumor sample
    -I normal.bam  # BAM file of the normal sample
    -tumor tumor_sample_name  # Name of the tumor sample
    -normal normal_sample_name  # Name of the normal sample
    -O output.vcf  # Output VCF file
    --germline-resource Mills_and_1000G_gold_standard.indels.vcf # Germline variant resource file for filtering
    --af-of-alleles-not-in-resource 0.0000025 # Prior mutation frequency for alleles not in the resource file
    
    # Optional step: Filter the results using FilterMutectCalls
    gatk FilterMutectCalls 
    -V output.vcf  # VCF file output by Mutect2
    -O filtered_output.vcf # Filtered output VCF file
```
After completing the above steps, the ```.vcf``` file can be converted to a ```.npy``` file, skipping the STEP 2 (Simualte Normal Samples), and proceeding directly to the inference task in the STEP 3.
### 1.2 Calling Somatic SNP whit Tumor samples
If the sample does not include normal sample data, we recommend using HaplotypeCaller to perform SNP detection on the sequencing data. And use our simulate_normal_samples.py method to generate normal sample data.
```bash
    gatk HaplotypeCaller 
    -R reference.fasta  # Reference genome file
    -I input.bam  # Input BAM file containing the sequencing data
    -O output.vcf  # Output VCF file with SNP calls
    -ERC GVCF  # Emit reference confidence scores, producing a gVCF file
    --output-mode EMIT_ALL_SITES  # Output all sites including those without mutations
```

### 1.3 change .vcf to .npy
The vcfTonpy.py file can be used to convert ```.vcf``` data into ```.npy``` format data.
```bash
    python vcfTonpy.py -p /path/to/dict/of/vcf
```
## 2. Simualte Normal Samples

### 2.1 Quick Start

### 2.2 Train Your Own Data

### 2.3 Prediction

## 3. Inferring Subclonal Populations

### 3.1 Quick Start

### 3.2 Train Your Own Data

### 3.3 Prediction
