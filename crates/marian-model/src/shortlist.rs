use std::{fs, path::Path};

const MAGIC: u64 = 17_373_278_592_220_534_773;
const MAX_TARGET_ENTRIES: usize = 4_000_000;

pub struct LexicalShortlist {
    source_vocab: usize,
    target_vocab: usize,
    first_num: usize,
    offsets: Vec<usize>,
    targets: Vec<u32>,
}

impl LexicalShortlist {
    pub fn load(
        path: Option<&Path>,
        source_vocab: usize,
        target_vocab: usize,
    ) -> Result<Self, String> {
        let Some(path) = path else {
            return Ok(Self {
                source_vocab,
                target_vocab,
                first_num: 0,
                offsets: Vec::new(),
                targets: Vec::new(),
            });
        };
        let offset_count_expected = source_vocab
            .checked_add(1)
            .ok_or_else(|| "source vocabulary is too large".to_string())?;
        let maximum_offset_bytes = offset_count_expected
            .checked_mul(8)
            .ok_or_else(|| "shortlist offset table size overflows usize".to_string())?;
        let maximum_bytes = 48usize
            .checked_add(maximum_offset_bytes)
            .and_then(|value| value.checked_add(MAX_TARGET_ENTRIES * 4))
            .ok_or_else(|| "shortlist size bound overflows usize".to_string())?;
        let metadata = fs::metadata(path).map_err(|error| {
            format!(
                "failed to inspect lexical shortlist {}: {error}",
                path.display()
            )
        })?;
        if metadata.len() > maximum_bytes as u64 {
            return Err(format!(
                "lexical shortlist {} has {} bytes; maximum is {maximum_bytes}",
                path.display(),
                metadata.len()
            ));
        }
        let bytes = fs::read(path).map_err(|error| {
            format!(
                "failed to read lexical shortlist {}: {error}",
                path.display()
            )
        })?;
        if bytes.len() < 48 {
            return Err("truncated lexical shortlist header".into());
        }
        let magic = read_u64(&bytes, 0)?;
        let first_num = to_usize(read_u64(&bytes, 16)?, "firstNum")?;
        let offset_count = to_usize(read_u64(&bytes, 32)?, "offset count")?;
        let target_count = to_usize(read_u64(&bytes, 40)?, "target count")?;
        if magic != MAGIC
            || offset_count != offset_count_expected
            || first_num > target_vocab
            || target_count > MAX_TARGET_ENTRIES
        {
            return Err("unsupported lexical shortlist format".into());
        }
        let expected_len = 48usize
            .checked_add(
                offset_count
                    .checked_mul(8)
                    .ok_or_else(|| "shortlist offset table is too large".to_string())?,
            )
            .and_then(|value| value.checked_add(target_count.checked_mul(4)?))
            .ok_or_else(|| "shortlist body is too large".to_string())?;
        if bytes.len() != expected_len {
            return Err(format!(
                "lexical shortlist has {} bytes, expected {expected_len}",
                bytes.len()
            ));
        }

        let mut cursor = 48;
        let mut offsets = Vec::with_capacity(offset_count);
        for _ in 0..offset_count {
            offsets.push(to_usize(read_u64(&bytes, cursor)?, "offset")?);
            cursor += 8;
        }
        if offsets.first() != Some(&0)
            || offsets.last() != Some(&target_count)
            || offsets
                .windows(2)
                .any(|pair| pair[0] > pair[1] || pair[1] > target_count)
        {
            return Err("lexical shortlist offsets are invalid".into());
        }

        let mut targets = Vec::with_capacity(target_count);
        for _ in 0..target_count {
            let target = read_u32(&bytes, cursor)?;
            if target as usize >= target_vocab {
                return Err(format!(
                    "lexical shortlist target {target} exceeds vocabulary {target_vocab}"
                ));
            }
            targets.push(target);
            cursor += 4;
        }
        Ok(Self {
            source_vocab,
            target_vocab,
            first_num,
            offsets,
            targets,
        })
    }

    pub fn candidates(&self, tokens: &[i32]) -> Result<Vec<u32>, String> {
        let mut present = vec![self.offsets.is_empty(); self.target_vocab];
        if !self.offsets.is_empty() {
            present[..self.first_num].fill(true);
            for &token in tokens {
                let token = usize::try_from(token)
                    .map_err(|_| "source token ID is negative".to_string())?;
                if token >= self.source_vocab {
                    return Err(format!(
                        "source token {token} exceeds vocabulary {}",
                        self.source_vocab
                    ));
                }
                for &target in &self.targets[self.offsets[token]..self.offsets[token + 1]] {
                    present[target as usize] = true;
                }
            }
        }

        // Match Marian's generation mask. UNK and byte/control fallbacks must
        // never become generated user text.
        if self.target_vocab > 1 {
            present[1] = false;
        }
        if self.target_vocab > 7 {
            present[7..self.target_vocab.min(39)].fill(false);
        }
        let mut candidates = present
            .iter()
            .enumerate()
            .filter_map(|(id, &selected)| selected.then_some(id as u32))
            .collect::<Vec<_>>();
        for (id, &already_present) in present.iter().enumerate().skip(50) {
            if candidates.len() % 8 == 0 {
                break;
            }
            if !already_present {
                candidates.push(id as u32);
            }
        }
        candidates.sort_unstable();
        if candidates.is_empty() {
            return Err("lexical shortlist produced no candidates".into());
        }
        if candidates.first() != Some(&0) {
            return Err("lexical shortlist does not contain EOS token 0".into());
        }
        Ok(candidates)
    }
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| "truncated lexical shortlist".to_string())?;
    Ok(u64::from_le_bytes(
        value.try_into().map_err(|_| "invalid u64".to_string())?,
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "truncated lexical shortlist".to_string())?;
    Ok(u32::from_le_bytes(
        value.try_into().map_err(|_| "invalid u32".to_string())?,
    ))
}

fn to_usize(value: u64, label: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{label} does not fit usize"))
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf, time::SystemTime};

    use super::{LexicalShortlist, MAGIC, MAX_TARGET_ENTRIES};

    fn temporary_path(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "marian-model-shortlist-{label}-{}-{unique}",
            std::process::id()
        ))
    }

    fn shortlist_bytes(first_num: u64, offsets: &[u64], targets: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for value in [
            MAGIC,
            0,
            first_num,
            0,
            offsets.len() as u64,
            targets.len() as u64,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for &offset in offsets {
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        for &target in targets {
            bytes.extend_from_slice(&target.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn full_vocabulary_masks_control_tokens() {
        let shortlist = LexicalShortlist::load(None, 64, 64).unwrap();
        let candidates = shortlist.candidates(&[0, 63]).unwrap();
        assert!(!candidates.contains(&1));
        assert!((7..=38).all(|id| !candidates.contains(&id)));
        assert!(candidates.contains(&0));
        assert!(candidates.contains(&63));
    }

    #[test]
    fn shortlist_must_supply_eos() {
        let path = temporary_path("missing-eos");
        fs::write(&path, shortlist_bytes(0, &[0, 1], &[2])).unwrap();
        let shortlist = LexicalShortlist::load(Some(&path), 1, 64).unwrap();
        assert_eq!(
            shortlist.candidates(&[0]).unwrap_err(),
            "lexical shortlist does not contain EOS token 0"
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn shortlist_entry_count_is_bounded() {
        let path = temporary_path("entry-bound");
        let mut bytes = shortlist_bytes(0, &[0, 0], &[]);
        bytes[40..48].copy_from_slice(&((MAX_TARGET_ENTRIES as u64) + 1).to_le_bytes());
        fs::write(&path, bytes).unwrap();
        assert!(LexicalShortlist::load(Some(&path), 1, 64).is_err());
        fs::remove_file(path).unwrap();
    }
}
