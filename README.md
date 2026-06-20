# ImGenMaga

Tools for preparing image datasets for FM/SFT experiments.

## CommonCatalog SFT builder

```powershell
uv run prepare-sft-dataset inspect-commoncatalog --limit-shards 20
uv run prepare-sft-dataset run --config configs/commoncatalog_sft.yaml
uv run prepare-sft-dataset validate-output --dataset-root D:\datasets\commoncatalog_sft
```

The implementation lives in `src/data/prepare_sft_dataset`.
