# Data provenance

The retained `data/wikitext2-*.parquet` files are unmodified public WikiText-2 raw-v1 split files from Salesforce/WikiText on Hugging Face. Byte sequences are constructed by joining each split’s `text` strings and encoding UTF-8. Training, validation and test remain separate.

Dataset card: https://huggingface.co/datasets/Salesforce/wikitext/blob/main/README.md

The upstream card metadata lists `cc-by-sa-3.0` and `gfdl`; its licensing section links CC BY-SA 4.0. These upstream notices apply to the data, independently of the BCIR code licenses retained in `vendor-old/LICENSE` and `vendor-new/LICENSE`.

Dataset authors: Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher, *Pointer Sentinel Mixture Models* (2016): https://arxiv.org/abs/1609.07843 . Source articles are credited through the upstream WikiText/Wikipedia dataset.

This experiment uses byte targets and synthetic recall tasks; it does not claim to reproduce the paper’s word-level benchmark.

## train

Source: https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet

Parquet SHA-256: `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`

Joined UTF-8 SHA-256: `fc1dd2874bf42e945db9bd38e856ae18010f5c088e7446c706c5ba2f39310ee3`; 10,914,845 bytes.

## validation

Source: https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/validation-00000-of-00001.parquet

Parquet SHA-256: `204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c`

Joined UTF-8 SHA-256: `8ef749789ca0693435d20b3f81d5638c19edcebc5a68586dcf09bdf47ef9542f`; 1,144,248 bytes.

## test

Source: https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet

Parquet SHA-256: `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`

Joined UTF-8 SHA-256: `bbf94c53a05abe9ee670d3b6343608095822c85e26de37c70b24fc571964574a`; 1,287,656 bytes.
