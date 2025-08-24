from setuptools import setup, find_packages

setup(
    name='chain_embed',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'torchaudio',
        'scipy',
        'tqdm',
        'moviepy',
        'transformers',
        'diffusers',
        'blobfile',
        'lxml',
        'num2words',
        'spacy',
        'huggingface-hub',
        'wandb',
        'matplotlib',
        'Motion @ git+https://github.com/inbar-2344/Motion.git'
    ],
)
