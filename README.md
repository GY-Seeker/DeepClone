# DeepClone: A New Approach for Inferring Cancer Subclone Populations Using Neural Networks
Yang Guo, Peng Nie*
***
## Operating Ambient
  - python > 3.7
```bash 
    pip install -r requirements.txt
```
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
Simulate the SNP or CN information of normal samples using ```simulate_normal_samples.py``` .

### 2.1 Parameter meaning
```simulate_normal_samples.py``` has 8 parameters
  - epoch. The number of iterations during neural network training ,default=200)
  - batch. The number of a batch in training data, default=64
  - learn_rate.
  - genomatic_length. The length of gene fragments,Different based on data from WGS and WES. WGS:1000 ~ 3000, WES:300 ~ 1000. Default:2048
  - latent_dim. The value between half of genomatic_length and genomatic_length. The smaller the latent_dim, the less memory is occupied and the poorer the simulation ability. The lager the latent_dim, the more memory is occupied, and the better the simulation ability. default:1024
  - data_path. The folder path for ```.npy``` files
  - data_type. 'CN' (Copy Number) or 'SNV' (Single Nucleotide Variate)
  - mode. Running mode, 'train' or 'predict'.

### 2.2 Train Your Own Data
Using ```simulate_normal_samples.py```, you can train simulated samples based on your mutation detection data. Refer to the following script.
```bash
# Example with all parameters
    python simulate_normal_samples.py 
    -e 200 
    -b 64 
    -lr 0.02 
    -g 2048 
    -ld 1024 
    -p /path/to/.npy/dict/ 
    -dt SNV
    -m train 
# Example with required parameters.
    python simulate_normal_samples.py 
    -p /path/to/.npy/dict/ 
    -dt SNV
    -m train 
```
### 2.3 Prediction
Using ```simulate_normal_samples.py```, you can predict simulated samples based on your mutation detection data. 
It is worth noting that during the prediction process, the ```batch_size```, ```genomatic_length```, and ```latent_dim``` need to remain consistent with those used during training.
Refer to the following script.
```bash
# Example with all parameters
    python simulate_normal_samples.py 
    -b 64 
    -g 2048 
    -ld 1024 
    -p /path/to/.npy/dict/ 
    -dt SNV
    -m predict 
# Example with required parameters.
    python simulate_normal_samples.py 
    -p /path/to/.npy/dict/ 
    -dt SNV
    -m predict 
```
## 3. Inferring Subclonal Populations

### 3.1 Parameter meaning
```inferring_subclone_populations.py``` has 5 parameters:
  - data_path, The path for storing mutation SNV information, which requires the use of ```.npy``` format files. 
  - groundTruth, The folder path for ground truth.
  - encode_path, The path to load the weights of the encoding model.
  - weight_path, The path to save trained weights.
  - mode, Running mode, optional 'train' or 'predict'.

### 3.2 Train Your Own Data
Using ```inferring_subclone_populations.py```, you can classify subclone populations.
```bash
# Example with all parameters
    python inferring_subclone_populations.py 
    -dp /path/to/.npy/dict/ 
    -gt /path/to/.npy/(goundTruth)
    -ed /path/to/weight/of/simulate_mode/
    -wp /path/to/save_weight/
    -m train
```
### 3.3 Prediction
```bash
# Example with all parameters
    python inferring_subclone_populations.py 
    -dp /path/to/.npy/dict/ 
    -gt /path/to/.npy/(goundTruth)
    -ed /path/to/weight/of/simulate_mode/
    -wp /path/to/save_weight/
    -m predict
```