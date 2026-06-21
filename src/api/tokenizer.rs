use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer;

#[allow(dead_code)]
pub struct LuminaTokenizer {
    tokenizer: Tokenizer,
}

impl LuminaTokenizer {
    #[allow(dead_code)]
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e))?;
        Ok(Self { tokenizer })
    }

    #[allow(dead_code)]
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(encoding.get_ids().to_vec())
    }

    #[allow(dead_code)]
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        let decoded = self
            .tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(decoded)
    }
}
