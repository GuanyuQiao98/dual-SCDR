#%%
import os
import loompy as lp
import numpy as np
from datasets import Dataset
import anndata
import pandas as pd
import pickle

#%%
import scanpy as sc
def process(data_path,save_path = None):
    if not save_path:
        sava_path = data_path.replace(".h5ad","")
    print(sava_path)
    if not os.path.exists(sava_path):
        os.mkdir(sava_path)

    loom_file_path = sava_path +"/"+ data_path.replace("h5ad", "loom").split("/")[1]
    add_attr_nameToDict = {"cell_name":"cell_name","condition":"condition","timepoint":"timepoint","nCount_RNA":"nCount_RNA","tissue":"tissue","patient":"patient",'S.Score':'S.Score', 'G2M.Score':'G2M.Score', 'Phase':'Phase', 'CC.Difference':'CC.Difference',"celltype":"celltype"}
    loom_cell_attr = [attr_key for attr_key in add_attr_nameToDict.keys()]
    cell_metadata = {attr_key: [] for attr_key in add_attr_nameToDict.values()}
    if not os.path.exists(loom_file_path):
        ad = anndata.read_h5ad(data_path)
        print(ad.X.shape)
        sc.pp.filter_genes(ad, min_cells=3)
        print(ad.X.shape)
        sc.pp.filter_cells(ad, min_genes=200)  # 至少表达 200 个基因
        print(ad.X.shape)

        name_to_dict = {d:i for i,d in enumerate(ad.var.index)}
        with open(sava_path+"/name_to_dict.pkl", "wb") as f:
            pickle.dump(name_to_dict, f)
            f.close()

        adata = ad
        col_attrs = {'cell_name': list(adata.obs_names)}
        for k,v in add_attr_nameToDict.items():
            if k in ad.obs:
                col_attrs[v] = list(adata.obs[k])
        lp.create(loom_file_path, adata.X.T, row_attrs={'ensembl_id': list(adata.var_names)},
                      col_attrs=col_attrs)

    #%%
    gene_token_dict = pd.read_pickle(sava_path+"/name_to_dict.pkl")
    genelist_dict = dict(zip(gene_token_dict.keys(), [True] * len(gene_token_dict.keys())))



    def tokenize_cell(gene_vector, gene_tokens):
        """
        Convert normalized gene expression vector to tokenized rank value encoding.
        """
        nonzero_mask = np.nonzero(gene_vector)[0]
        # sort by median-scaled gene values
        sorted_indices = np.argsort(-gene_vector[nonzero_mask])
        # tokenize
        sentence_tokens = gene_tokens[nonzero_mask][sorted_indices]
        sentence_exp = gene_vector[nonzero_mask][sorted_indices]

        return {"sentence_tokens":sentence_tokens,"sentence_exp":sentence_exp}

    def tokenize_file(loom_file_path):
        file_cell_metadata = {
            attr_key: [] for attr_key in add_attr_nameToDict.keys()
        }
        with lp.connect(str(loom_file_path)) as data:
            coding_RNA_loc = np.where(
                [genelist_dict.get(i, False) for i in data.ra["ensembl_id"]]
            )[0]


            coding_RNA_ids = data.ra["ensembl_id"][coding_RNA_loc]
            coding_RNA_tokens = np.array(
                [gene_token_dict[i] for i in coding_RNA_ids]
            )

            filter_pass_loc = np.array([i for i in range(data.shape[1])])

            # scan through .loom files and tokenize cells
            tokenized_cells = []
            for (_ix, _selection, view) in data.scan(items=filter_pass_loc, axis=1):
                subview = view.view[coding_RNA_loc, :]
                subview_norm_array = (
                        subview[:, :]
                )
                tokenized_cells += [
                    tokenize_cell(subview_norm_array[:, i], coding_RNA_tokens)
                    for i in range(subview_norm_array.shape[1])
                ]
                for k in file_cell_metadata.keys():
                    if k in subview.ca:
                        file_cell_metadata[k] += subview.ca[k].tolist()

            numerics_cells=[isd["sentence_exp"] for isd in tokenized_cells]
            tokenize_cells=[isd["sentence_tokens"] for isd in tokenized_cells]

        return tokenize_cells,numerics_cells, file_cell_metadata

    file_tokenized_cells,file_numerics_cells, file_cell_metadata = tokenize_file(
            loom_file_path
        )
    numerics_cells = []
    tokenized_cells = []

    numerics_cells+= file_numerics_cells
    tokenized_cells += file_tokenized_cells

    for k in loom_cell_attr:
        cell_metadata[add_attr_nameToDict[k]] += file_cell_metadata[k]

    def create_dataset(tokenized_cells,numerics_cells, cell_metadata):
            dataset_dict = {"input_ids": tokenized_cells}
            dataset_dict["numeric_ids"] = numerics_cells
            cell_metadata = {i:d for i,d in cell_metadata.items() if d}
            dataset_dict.update(cell_metadata)

            output_dataset = Dataset.from_dict(dataset_dict)


            output_dataset_truncated = output_dataset
            def measure_length(example):
                example["length"] = len(example["input_ids"])
                return example

            output_dataset_truncated_w_length = output_dataset_truncated.map(
                measure_length, num_proc=8
            )
            return output_dataset_truncated_w_length


    tokenized_dataset = create_dataset(tokenized_cells, numerics_cells, cell_metadata)
    output_path = "./" +  sava_path + "/"+ sava_path.split("/")[-1].split(".")[0]+ ".dataset"
    print(tokenized_dataset)
    tokenized_dataset.save_to_disk(output_path,num_shards=1)


if __name__ == '__main__':


    data_list = ["data/GSE161801_PI_seurat_afterAnno.h5ad",
                 "data/GSE162117_seurat_afterAnno .h5ad",]

    for oi in data_list:
        process(oi)