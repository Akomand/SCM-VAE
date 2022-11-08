# SCM-VAE: Learning Identifiable Causal Representations via Structural Knowledge
This is the source code for the implementation of "SCM-VAE: Learning Identifiable Causal Representations via Structural Knowledge" 

The goal of causal representation learning is to map low-level observations to high-level causal concepts to learn interpretable and robust representations for various downstream tasks. Latent variable models such as the variational autoencoder (VAE) are frequently leveraged to learn disentangled representations. However, there are often complex non-linear causal relationships underlying the observed data that cannot be captured through disentangled representations or linear dependence assumptions. Further, an independent conditional prior assumption can make learning causal dependencies in the latent space more challenging. We propose a framework, coined SCM-VAE, which uses apriori causal knowledge, a structural causal prior, and a non-linear additive noise structural causal model (SCM) to learn independent causal mechanisms and identifiable causal representations. We conduct theoretical analysis and perform experiments on synthetic and real-world datasets to show the improved quality of learned causal representations and robustness under interventions.



## Usage

### Training and evaluating 

1. Clone the repository

     ```
     git clone https://github.com/Akomand/SCM-VAE.git
     cd SCM-VAE
     ```

2. Create New Environment

    ```
    conda env create -f environment.yml
    ```

3. Activate environment

    ```
    conda activate scm_vae
    ```

4. Generate data

    ```
    python data/data_generators/pendulum.py
    python data/data_generators/flow.py
    ```

5. Navigate to `experiments` folder

    ```
    cd experiments
    ```

6. Run training script

    ```
    python train_scm_vae.py --dataset-dir='../data/pendulum'
    ```


### Data acknowledgements
Experiments are run on the following datasets to evaluate our model:

#### Datasets
<details closed>
<summary>Pendulum Dataset</summary>

[Link to dataset](https://github.com/huawei-noah/trustworthyAI/tree/master/research/CausalVAE/causal_data)
</details>

<details closed>
<summary>Flow Dataset</summary>

[Link to dataset](https://github.com/huawei-noah/trustworthyAI/tree/master/research/CausalVAE/causal_data)
</details>

<details closed>
<summary>CelebA Dataset</summary>

[Link to dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
</details>

## Citation

If you use our code or think our work is relevant to yours, we encourage you to cite this paper:

```bibtex
@INPROCEEDINGS{komandbayesian,
  author={Komanduri, Aneesh },
  booktitle={2022 IEEE International Conference on Big Data}, 
  title={SCM-VAE: Learning Identifiable Causal Representations via Structural Knowledge}, 
  year={2022}
  }
```


## Acknowledgement
We acknowledge the following open-source repository that we built our work on
- [CausalVAE](https://github.com/huawei-noah/trustworthyAI/tree/master/research/CausalVAE)
