use std::path::Path;

use tokenizers::Tokenizer;

pub struct LuminaTokenizer {
    tokenizer: Tokenizer,
}

impl LuminaTokenizer {
    pub fn from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e))?;
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32]) -> anyhow::Result<String> {
        let decoded = self
            .tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(decoded)
    }
}
