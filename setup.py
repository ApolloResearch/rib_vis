from setuptools import setup, find_packages

setup(
    name = 'sae-vis',
    version = '0.1',
    packages = find_packages(),
    install_requires = [
        'torch',
        'einops',
        'datasets',
        'jaxtyping',
        'eindex',
    ],
    dependency_links = [
        'git+https://github.com/callummcdougall/eindex.git@a84f40ce5fabdc09c60036c9834d96808c897200#egg=eindex'
    ],
    include_package_data = True,
    author = 'Callum McDougall',
    author_email = 'cal.s.mcdougall@gmail.com',
    description = "Open-source SAE visualizer, based on Anthropic's published visualizer.",
    url = 'https://github.com/callummcdougall/sae_visualizer',
    classifiers = [
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
