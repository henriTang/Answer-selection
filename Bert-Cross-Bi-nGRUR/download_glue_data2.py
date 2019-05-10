''' Script for downloading all GLUE data.

Note: for legal reasons, we are unable to host MRPC.
You can either use the version hosted by the SentEval team, which is already tokenized, 
or you can download the original data from (https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi) and extract the data from it manually.
For Windows users, you can run the .msi file. For Mac and Linux users, consider an external library such as 'cabextract' (see below for an example).
You should then rename and place specific files in a folder (see below for an example).

mkdir MRPC
cabextract MSRParaphraseCorpus.msi -d MRPC
cat MRPC/_2DEC3DBE877E4DB192D17C0256E90F1D | tr -d $'\r' > MRPC/msr_paraphrase_train.txt
cat MRPC/_D7B391F9EAFF4B1B8BCE8F21B20B1B61 | tr -d $'\r' > MRPC/msr_paraphrase_test.txt
rm MRPC/_*
rm MSRParaphraseCorpus.msi

1/30/19: It looks like SentEval is no longer hosting their extracted and tokenized MRPC data, so you'll need to download the data from the original source for now.
2/11/19: It looks like SentEval actually *is* hosting the extracted data. Hooray!
'''

import os
import sys
import shutil
import argparse
import tempfile
import zipfile


print("Processing MRPC...")
data_dir = "./glue_data"
mrpc_dir = os.path.join(data_dir, "MRPC")
mrpc_train_file = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
mrpc_test_file = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")


dev_ids = []
with open(os.path.join(mrpc_dir, "mrpc_dev_ids.tsv"), encoding="utf8") as ids_fh:
    for row in ids_fh:
        dev_ids.append(row.strip().split('\t'))

with open(mrpc_train_file, encoding="utf8") as data_fh, \
     open(os.path.join(mrpc_dir, "train.tsv"), 'w', encoding="utf8") as train_fh, \
     open(os.path.join(mrpc_dir, "dev.tsv"), 'w', encoding="utf8") as dev_fh:
    header = data_fh.readline()
    train_fh.write(header)
    dev_fh.write(header)
    for row in data_fh:
        label, id1, id2, s1, s2 = row.strip().split('\t')
        if [id1, id2] in dev_ids:
            dev_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))
        else:
            train_fh.write("%s\t%s\t%s\t%s\t%s\n" % (label, id1, id2, s1, s2))

with open(mrpc_test_file, encoding="utf8") as data_fh, \
        open(os.path.join(mrpc_dir, "test.tsv"), 'w', encoding="utf8") as test_fh:
    header = data_fh.readline()
    test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
    for idx, row in enumerate(data_fh):
        label, id1, id2, s1, s2 = row.strip().split('\t')
        test_fh.write("%d\t%s\t%s\t%s\t%s\n" % (idx, id1, id2, s1, s2))
print("\tCompleted!")



