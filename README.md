# WCGAN-GP

*Wasserstein Conditional GAN with Gradient Penalty* or WCGAN-GP for short, is a Generative Adversarial Network model used by [Walia, Tierney and McKeever 2020](https://www.researchgate.net/publication/347437993_Synthesising_Tabular_Data_using_Wasserstein_Conditional_GANs_with_Gradient_Penalty_WCGAN-GP) to create synthetic tabular data.

WCGAN-GP uses *Wasserstein loss* to overcome mode collapse, *gradient penalty* instead of weight clipping to increase stability in training, while also being a *conditional* GAN meaning that it can create data conditioned by input label.

This repo contains the TensorFlow 2 implementation of WCGAN-GP and demonstrates its use by creating synthetic data from [CIC-IDS-2017]() dataset.

---

## Install

```bash
https://github.com/marzekan/WCGAN-GP.git
```

```bash
cd WCGAN-GP
```

```bash
pip install -r requirements.txt
```

or

```
pipenv shell
```

## Run

Checkout the Jupyter Notebook

```bash
pipenv run juypter notebook
```

... and open [wcgan-gp.ipynb]()

or 

use the packaged module: [wcgan.py]()

---

## Synthetic data evaluation results

Synthetic data evaluation was done using the [TableEvaluator](https://pypi.org/project/table-evaluator/) package.

---

<strong style="color: red">Important!</strong>

For demo purposes and to reduce resource usage the original **CIC-IDS-2017 is sampled to 25% of the original dataset size**.

This is sure to implact the results so if you want to get the best possible results you should train the WCGAN-GP on entire CIC-IDS-2017 after running it through [cleaning-cic-ids-2017.ipynb]() and [data-preproc.ipynb]() notebooks.

---

## Sources

[Walia, Manhar & Tierney, Brendan & Mckeever, Susan. (2020). Synthesising Tabular Data using Wasserstein Conditional GANs with Gradient Penalty (WCGAN-GP)](https://www.researchgate.net/publication/347437993_Synthesising_Tabular_Data_using_Wasserstein_Conditional_GANs_with_Gradient_Penalty_WCGAN-GP)

[Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). Improved Training of Wasserstein GANs.](https://arxiv.org/abs/1704.00028)

[https://github.com/eriklindernoren/Keras-GAN](https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py)