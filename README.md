This is a fork of [callummcdougall/sae_vis](https://github.com/callummcdougall/sae_vis) where we
maintain custom modifications required for visualizing our non-SAE features, as well as PR branches
to upstream. This code is likely not great for visualizing SAEs, use
[callummcdougall/sae_vis](https://github.com/callummcdougall/sae_vis) instead.

The main feature in this repo, up to small changes, is the new function `parse_activation_data`
which is a more generic, non-SAE specific version of `_get_feature_data`.
```python
def parse_activation_data(
    tokens: Int[Tensor, "batch n_ctx"],
    feature_acts: Float[Tensor, "batch n_ctx n_feats"],
    final_resid_acts: Float[Tensor, "batch n_ctx d_resid_end"],
    feature_resid_dirs: Float[Tensor, "n_feats d_resid_feat"],
    feature_indices_list: list[int],
    W_U: Float[Tensor, "dim d_vocab"],
    vocab: dict[int, str],
    fvp: FeatureVisParams,
) -> MultiFeatureData:
```
It takes in a list of `tokens` and corresponding feature activations `feature_acts` to produce
a visualization based on dataset examples (`SequenceMultiGroupData`) by calling `get_sequences_data`.

Additionally it uses `final_resid_acts`, `feature_resid_dir`, and `W_U` to produce the logit lens
data (`MiddlePlotsData`).

`vocab` is used for vizualizing the tokens and can either be a dictionary or a Tokenizer object.
We prefer the latter to avoid weird behaviour that the tokenizer.get_vobab() dictionary has.