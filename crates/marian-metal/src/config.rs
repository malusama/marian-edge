//! Process-facing configuration for the direct Metal backend.
//!
//! Environment parsing intentionally stops at this module. The runtime and
//! graph receive explicit values, which keeps tests deterministic and permits
//! multiple differently configured backends in one process.

use marian_model::MAXIMUM_POSITION;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MetalPrecision {
    #[default]
    Fp32,
    MixedF16,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MetalProfile {
    #[default]
    Auto,
    M1,
    M2,
    M3,
    M4,
    Generic,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MetalAttention {
    #[default]
    Auto,
    Classic,
    Flash,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalConfig {
    pub precision: MetalPrecision,
    pub profile: MetalProfile,
    pub attention: MetalAttention,
    pub flash_threshold: usize,
    pub flash_query_tile: Option<usize>,
    pub duplicate_batch_width: Option<usize>,
    pub decode_row_budget: Option<usize>,
    pub decode_maximum_steps: Option<usize>,
    pub decode_selection_threads: Option<usize>,
    pub custom_gemm_maximum_rows: Option<usize>,
}

impl Default for MetalConfig {
    fn default() -> Self {
        Self {
            precision: MetalPrecision::Fp32,
            profile: MetalProfile::Auto,
            attention: MetalAttention::Auto,
            flash_threshold: 1,
            flash_query_tile: None,
            duplicate_batch_width: None,
            decode_row_budget: None,
            decode_maximum_steps: None,
            decode_selection_threads: None,
            custom_gemm_maximum_rows: None,
        }
    }
}

impl MetalConfig {
    pub fn from_env() -> Result<Self, String> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    fn from_lookup(mut lookup: impl FnMut(&str) -> Option<String>) -> Result<Self, String> {
        let precision = match setting("METAL_PRECISION", &mut lookup)?
            .unwrap_or_else(|| "fp32".into())
            .as_str()
        {
            "fp32" => MetalPrecision::Fp32,
            "mixed-f16" => MetalPrecision::MixedF16,
            value => {
                return Err(format!(
                    "unsupported MARIAN_EDGE_METAL_PRECISION {value:?}; expected fp32 or mixed-f16"
                ));
            }
        };
        let profile = match setting("METAL_PROFILE", &mut lookup)?
            .unwrap_or_else(|| "auto".into())
            .as_str()
        {
            "auto" => MetalProfile::Auto,
            "m1" => MetalProfile::M1,
            "m2" => MetalProfile::M2,
            "m3" => MetalProfile::M3,
            "m4" => MetalProfile::M4,
            "generic" => MetalProfile::Generic,
            value => {
                return Err(format!(
                    "unsupported MARIAN_EDGE_METAL_PROFILE {value:?}; expected auto, m1, m2, m3, m4, or generic"
                ));
            }
        };
        let attention = match setting("METAL_ATTENTION", &mut lookup)?
            .unwrap_or_else(|| "auto".into())
            .as_str()
        {
            "auto" => MetalAttention::Auto,
            "classic" => MetalAttention::Classic,
            "flash" => MetalAttention::Flash,
            value => {
                return Err(format!(
                    "unsupported MARIAN_EDGE_METAL_ATTENTION {value:?}; expected auto, classic, or flash"
                ));
            }
        };
        let flash_threshold = positive_value("METAL_FLASH_THRESHOLD", &mut lookup)?.unwrap_or(1);
        if flash_threshold > MAXIMUM_POSITION {
            return Err(format!(
                "MARIAN_EDGE_METAL_FLASH_THRESHOLD must be between 1 and {MAXIMUM_POSITION}"
            ));
        }
        let flash_query_tile = positive_value("METAL_FLASH_QUERY_TILE", &mut lookup)?;
        if flash_query_tile.is_some_and(|tile| !matches!(tile, 1 | 2 | 4)) {
            return Err("MARIAN_EDGE_METAL_FLASH_QUERY_TILE must be one of 1, 2, or 4".into());
        }
        let duplicate_batch_width = positive_value("METAL_DUPLICATE_BATCH_WIDTH", &mut lookup)?;
        let decode_row_budget = positive_value("METAL_DECODE_ROW_BUDGET", &mut lookup)?;
        let decode_maximum_steps = positive_value("METAL_DECODE_MAX_STEPS", &mut lookup)?;
        if decode_maximum_steps.is_some_and(|steps| steps > 8) {
            return Err("MARIAN_EDGE_METAL_DECODE_MAX_STEPS must be between 1 and 8".into());
        }
        let decode_selection_threads =
            positive_value("METAL_DECODE_SELECTION_THREADS", &mut lookup)?;
        if decode_selection_threads.is_some_and(|threads| !matches!(threads, 128 | 256 | 512)) {
            return Err(
                "MARIAN_EDGE_METAL_DECODE_SELECTION_THREADS must be 128, 256, or 512".into(),
            );
        }
        let custom_gemm_maximum_rows =
            non_negative_value("METAL_CUSTOM_GEMM_MAX_ROWS", &mut lookup)?;
        Ok(Self {
            precision,
            profile,
            attention,
            flash_threshold,
            flash_query_tile,
            duplicate_batch_width,
            decode_row_budget,
            decode_maximum_steps,
            decode_selection_threads,
            custom_gemm_maximum_rows,
        })
    }
}

fn positive_value(
    suffix: &str,
    lookup: &mut impl FnMut(&str) -> Option<String>,
) -> Result<Option<usize>, String> {
    let Some(value) = setting(suffix, lookup)? else {
        return Ok(None);
    };
    let name = format!("MARIAN_EDGE_{suffix}");
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("{name} {value:?} is not an integer"))?;
    if parsed == 0 {
        return Err(format!("{name} must be at least 1"));
    }
    Ok(Some(parsed))
}

fn non_negative_value(
    suffix: &str,
    lookup: &mut impl FnMut(&str) -> Option<String>,
) -> Result<Option<usize>, String> {
    let Some(value) = setting(suffix, lookup)? else {
        return Ok(None);
    };
    let name = format!("MARIAN_EDGE_{suffix}");
    value
        .parse::<usize>()
        .map(Some)
        .map_err(|_| format!("{name} {value:?} is not a non-negative integer"))
}

fn setting(
    suffix: &str,
    lookup: &mut impl FnMut(&str) -> Option<String>,
) -> Result<Option<String>, String> {
    let primary_name = format!("MARIAN_EDGE_{suffix}");
    let alias_name = format!("MARIAN_MLX_{suffix}");
    let primary = lookup(&primary_name);
    let alias = lookup(&alias_name);
    match (primary, alias) {
        (Some(primary), Some(alias)) if primary != alias => Err(format!(
            "conflicting settings: {primary_name}={primary:?} and {alias_name}={alias:?}"
        )),
        (Some(primary), _) => Ok(Some(primary)),
        (None, alias) => Ok(alias),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::{MetalAttention, MetalConfig, MetalPrecision, MetalProfile};

    #[test]
    fn default_config_is_stable_and_hardware_independent() {
        let config = MetalConfig::default();
        assert_eq!(config.flash_threshold, 1);
        assert_eq!(config.flash_query_tile, None);
        assert_eq!(config.duplicate_batch_width, None);
        assert_eq!(config.decode_maximum_steps, None);
        assert_eq!(config.decode_selection_threads, None);
    }

    #[test]
    fn explicit_values_parse_without_mutating_process_environment() {
        let values = HashMap::from([
            ("MARIAN_EDGE_METAL_PRECISION", "mixed-f16"),
            ("MARIAN_EDGE_METAL_PROFILE", "m1"),
            ("MARIAN_EDGE_METAL_ATTENTION", "flash"),
            ("MARIAN_EDGE_METAL_FLASH_THRESHOLD", "128"),
            ("MARIAN_EDGE_METAL_FLASH_QUERY_TILE", "4"),
            ("MARIAN_EDGE_METAL_DUPLICATE_BATCH_WIDTH", "9"),
            ("MARIAN_EDGE_METAL_DECODE_ROW_BUDGET", "54"),
            ("MARIAN_EDGE_METAL_DECODE_MAX_STEPS", "6"),
            ("MARIAN_EDGE_METAL_DECODE_SELECTION_THREADS", "256"),
            ("MARIAN_EDGE_METAL_CUSTOM_GEMM_MAX_ROWS", "0"),
        ]);
        let config = MetalConfig::from_lookup(|name| values.get(name).map(ToString::to_string))
            .expect("configuration must parse");

        assert_eq!(config.precision, MetalPrecision::MixedF16);
        assert_eq!(config.profile, MetalProfile::M1);
        assert_eq!(config.attention, MetalAttention::Flash);
        assert_eq!(config.flash_threshold, 128);
        assert_eq!(config.flash_query_tile, Some(4));
        assert_eq!(config.duplicate_batch_width, Some(9));
        assert_eq!(config.decode_row_budget, Some(54));
        assert_eq!(config.decode_maximum_steps, Some(6));
        assert_eq!(config.decode_selection_threads, Some(256));
        assert_eq!(config.custom_gemm_maximum_rows, Some(0));
    }

    #[test]
    fn invalid_values_name_the_failed_setting() {
        let error = MetalConfig::from_lookup(|name| {
            (name == "MARIAN_EDGE_METAL_DECODE_SELECTION_THREADS").then(|| "64".into())
        })
        .expect_err("invalid thread count must fail");
        assert!(error.contains("MARIAN_EDGE_METAL_DECODE_SELECTION_THREADS"));
    }

    #[test]
    fn legacy_prefix_is_supported_but_conflicting_values_fail() {
        let alias = HashMap::from([
            ("MARIAN_MLX_METAL_PRECISION", "mixed-f16"),
            ("MARIAN_MLX_METAL_PROFILE", "m1"),
        ]);
        let config = MetalConfig::from_lookup(|name| alias.get(name).map(ToString::to_string))
            .expect("legacy prefix must remain supported");
        assert_eq!(config.precision, MetalPrecision::MixedF16);
        assert_eq!(config.profile, MetalProfile::M1);

        let conflict = HashMap::from([
            ("MARIAN_EDGE_METAL_PROFILE", "m1"),
            ("MARIAN_MLX_METAL_PROFILE", "m2"),
        ]);
        let error = MetalConfig::from_lookup(|name| conflict.get(name).map(ToString::to_string))
            .expect_err("conflicting prefixes must not choose silently");
        assert!(error.contains("conflicting settings"));
    }
}
