use std::fmt;

use marian_model::MAXIMUM_POSITION;

use crate::{MetalAttention, MetalConfig, MetalProfile};

const FLASH_ATTENTION_MAX_HEAD_DIM: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AttentionTuning {
    mode: MetalAttention,
    threshold: usize,
    query_tile: usize,
}

impl AttentionTuning {
    pub(crate) fn use_flash(self, query_length: usize, key_length: usize, head_dim: usize) -> bool {
        if head_dim > FLASH_ATTENTION_MAX_HEAD_DIM {
            return false;
        }
        match self.mode {
            MetalAttention::Classic => false,
            MetalAttention::Flash => true,
            MetalAttention::Auto => {
                query_length == 1 || query_length == key_length && query_length >= self.threshold
            }
        }
    }

    pub(crate) const fn query_tile(self) -> usize {
        self.query_tile
    }

    fn label(self) -> String {
        match self.mode {
            MetalAttention::Classic => "classic".into(),
            MetalAttention::Flash => format!("flash-q{}", self.query_tile),
            MetalAttention::Auto => {
                format!("flash-q{}-auto@{}", self.query_tile, self.threshold)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DecodeTuning {
    row_budget: usize,
    maximum_steps: usize,
    selection_threads: usize,
}

impl DecodeTuning {
    pub(crate) fn submission_steps(
        self,
        active_rows: usize,
        remaining_steps: usize,
        completion_observed: bool,
    ) -> usize {
        let occupancy_steps = (self.row_budget / active_rows.max(1)).max(1);
        let completion_cap = if completion_observed {
            1
        } else {
            self.maximum_steps
        };
        occupancy_steps
            .min(completion_cap)
            .min(remaining_steps)
            .max(1)
    }

    pub(crate) const fn selection_threads(self) -> usize {
        self.selection_threads
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GemmTuning {
    custom_maximum_rows: usize,
}

impl GemmTuning {
    pub(crate) fn use_custom_fp32(self, rows: usize, columns: usize) -> bool {
        self.custom_maximum_rows > 0 && rows <= self.custom_maximum_rows && columns >= 32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DeviceFamily {
    M1,
    M2,
    M3,
    M4,
    Generic,
}

impl DeviceFamily {
    fn detect(device_name: &str) -> Self {
        let compact = device_name.replace([' ', '-'], "").to_ascii_lowercase();
        if compact.contains("applem4") {
            Self::M4
        } else if compact.contains("applem3") {
            Self::M3
        } else if compact.contains("applem2") {
            Self::M2
        } else if compact.contains("applem1") {
            Self::M1
        } else {
            Self::Generic
        }
    }

    const fn resolve(profile: MetalProfile, detected: Self) -> Self {
        match profile {
            MetalProfile::Auto => detected,
            MetalProfile::M1 => Self::M1,
            MetalProfile::M2 => Self::M2,
            MetalProfile::M3 => Self::M3,
            MetalProfile::M4 => Self::M4,
            MetalProfile::Generic => Self::Generic,
        }
    }
}

impl fmt::Display for DeviceFamily {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::M1 => "m1",
            Self::M2 => "m2",
            Self::M3 => "m3",
            Self::M4 => "m4",
            Self::Generic => "generic",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DeviceDefaults {
    duplicate_batch_width: usize,
    flash_query_tile: usize,
    decode_row_budget: usize,
    decode_maximum_steps: usize,
    decode_selection_threads: usize,
    custom_gemm_maximum_rows: usize,
}

impl DeviceDefaults {
    const fn for_family(family: DeviceFamily) -> Self {
        match family {
            // M1 is qualified on the release host. Later-family profiles are
            // conservative, explicit starting points until checked-in sweeps
            // can replace them without touching inference graph code.
            DeviceFamily::M1 => Self {
                duplicate_batch_width: 9,
                flash_query_tile: 4,
                decode_row_budget: 54,
                decode_maximum_steps: 6,
                decode_selection_threads: 256,
                custom_gemm_maximum_rows: 0,
            },
            DeviceFamily::M2 => Self {
                duplicate_batch_width: 8,
                flash_query_tile: 4,
                decode_row_budget: 48,
                decode_maximum_steps: 3,
                decode_selection_threads: 256,
                custom_gemm_maximum_rows: 0,
            },
            DeviceFamily::M3 => Self {
                duplicate_batch_width: 8,
                flash_query_tile: 4,
                decode_row_budget: 64,
                decode_maximum_steps: 4,
                decode_selection_threads: 512,
                custom_gemm_maximum_rows: 0,
            },
            DeviceFamily::M4 => Self {
                duplicate_batch_width: 8,
                flash_query_tile: 4,
                decode_row_budget: 64,
                decode_maximum_steps: 4,
                decode_selection_threads: 512,
                custom_gemm_maximum_rows: 0,
            },
            DeviceFamily::Generic => Self {
                duplicate_batch_width: 4,
                flash_query_tile: 2,
                decode_row_budget: 24,
                decode_maximum_steps: 2,
                decode_selection_threads: 128,
                custom_gemm_maximum_rows: 0,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MetalTuning {
    family: DeviceFamily,
    pub(crate) attention: AttentionTuning,
    pub(crate) decode: DecodeTuning,
    pub(crate) gemm: GemmTuning,
    duplicate_batch_width: usize,
}

impl MetalTuning {
    pub(crate) fn resolve(device_name: &str, config: &MetalConfig) -> Result<Self, String> {
        let detected = DeviceFamily::detect(device_name);
        let family = DeviceFamily::resolve(config.profile, detected);
        let defaults = DeviceDefaults::for_family(family);
        if !(1..=MAXIMUM_POSITION).contains(&config.flash_threshold) {
            return Err(format!(
                "flash threshold must be between 1 and {MAXIMUM_POSITION}"
            ));
        }
        let query_tile = config.flash_query_tile.unwrap_or(defaults.flash_query_tile);
        if !matches!(query_tile, 1 | 2 | 4) {
            return Err("flash query tile must be one of 1, 2, or 4".into());
        }
        let duplicate_batch_width = config
            .duplicate_batch_width
            .unwrap_or(defaults.duplicate_batch_width);
        let decode_row_budget = config
            .decode_row_budget
            .unwrap_or(defaults.decode_row_budget);
        if duplicate_batch_width == 0 {
            return Err("duplicate batch width must be at least 1".into());
        }
        if decode_row_budget == 0 {
            return Err("decode row budget must be at least 1".into());
        }
        let decode_maximum_steps = config
            .decode_maximum_steps
            .unwrap_or(defaults.decode_maximum_steps);
        if !(1..=8).contains(&decode_maximum_steps) {
            return Err("decode maximum steps must be between 1 and 8".into());
        }
        let custom_gemm_maximum_rows = config
            .custom_gemm_maximum_rows
            .unwrap_or(defaults.custom_gemm_maximum_rows);
        let decode_selection_threads = config
            .decode_selection_threads
            .unwrap_or(defaults.decode_selection_threads);
        if !matches!(decode_selection_threads, 128 | 256 | 512) {
            return Err("decode selection threads must be 128, 256, or 512".into());
        }
        Ok(Self {
            family,
            attention: AttentionTuning {
                mode: config.attention,
                threshold: config.flash_threshold,
                query_tile,
            },
            decode: DecodeTuning {
                row_budget: decode_row_budget,
                maximum_steps: decode_maximum_steps,
                selection_threads: decode_selection_threads,
            },
            gemm: GemmTuning {
                custom_maximum_rows: custom_gemm_maximum_rows,
            },
            duplicate_batch_width,
        })
    }

    pub(crate) fn duplicate_batch_width(&self) -> usize {
        self.duplicate_batch_width
    }

    pub(crate) fn attention_label(&self) -> String {
        self.attention.label()
    }

    pub(crate) fn profile_label(&self) -> String {
        format!(
            "{}(width={},decode-rows={},decode-steps={},select-threads={},custom-gemm-max={})",
            self.family,
            self.duplicate_batch_width,
            self.decode.row_budget,
            self.decode.maximum_steps,
            self.decode.selection_threads,
            self.gemm.custom_maximum_rows,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{AttentionTuning, DecodeTuning, DeviceFamily, MetalTuning};
    use crate::{MetalAttention, MetalConfig, MetalProfile};

    #[test]
    fn attention_dispatch_respects_mode_shape_threshold_and_head_limit() {
        let classic = AttentionTuning {
            mode: MetalAttention::Classic,
            threshold: 1,
            query_tile: 4,
        };
        assert!(!classic.use_flash(128, 128, 48));

        let flash = AttentionTuning {
            mode: MetalAttention::Flash,
            threshold: 4_096,
            query_tile: 4,
        };
        assert!(flash.use_flash(7, 19, 48));
        assert!(!flash.use_flash(7, 19, 65));

        let auto = AttentionTuning {
            mode: MetalAttention::Auto,
            threshold: 128,
            query_tile: 4,
        };
        assert!(auto.use_flash(1, 512, 48));
        assert!(!auto.use_flash(127, 127, 48));
        assert!(auto.use_flash(128, 128, 48));
        assert!(!auto.use_flash(128, 256, 48));
    }

    #[test]
    fn device_family_detection_ignores_product_suffixes() {
        assert_eq!(DeviceFamily::detect("Apple M1"), DeviceFamily::M1);
        assert_eq!(DeviceFamily::detect("Apple M2 Max"), DeviceFamily::M2);
        assert_eq!(DeviceFamily::detect("Apple-M3-Ultra"), DeviceFamily::M3);
        assert_eq!(DeviceFamily::detect("Apple M4 Pro"), DeviceFamily::M4);
        assert_eq!(DeviceFamily::detect("Unknown GPU"), DeviceFamily::Generic);
    }

    #[test]
    fn decode_submission_reacts_to_active_rows_and_completion() {
        let tuning = DecodeTuning {
            row_budget: 32,
            maximum_steps: 3,
            selection_threads: 256,
        };
        assert_eq!(tuning.submission_steps(16, 20, false), 2);
        assert_eq!(tuning.submission_steps(9, 20, false), 3);
        assert_eq!(tuning.submission_steps(1, 2, false), 2);
        assert_eq!(tuning.submission_steps(4, 20, true), 1);
        assert_eq!(tuning.selection_threads(), 256);
    }

    #[test]
    fn auto_m1_profile_resolves_to_qualified_defaults() {
        let tuning = MetalTuning::resolve("Apple M1", &MetalConfig::default()).unwrap();

        assert_eq!(tuning.family, DeviceFamily::M1);
        assert_eq!(tuning.duplicate_batch_width, 9);
        assert_eq!(tuning.attention.query_tile, 4);
        assert_eq!(tuning.decode.row_budget, 54);
        assert_eq!(tuning.decode.maximum_steps, 6);
        assert_eq!(tuning.decode.selection_threads, 256);
        assert_eq!(tuning.gemm.custom_maximum_rows, 0);
        assert_eq!(
            tuning.profile_label(),
            "m1(width=9,decode-rows=54,decode-steps=6,select-threads=256,custom-gemm-max=0)"
        );
    }

    #[test]
    fn explicit_profile_and_knobs_override_detected_family_defaults() {
        let config = MetalConfig {
            profile: MetalProfile::Generic,
            flash_query_tile: Some(1),
            duplicate_batch_width: Some(7),
            decode_row_budget: Some(35),
            decode_maximum_steps: Some(5),
            decode_selection_threads: Some(512),
            custom_gemm_maximum_rows: Some(3),
            ..MetalConfig::default()
        };
        let tuning = MetalTuning::resolve("Apple M1", &config).unwrap();

        assert_eq!(tuning.family, DeviceFamily::Generic);
        assert_eq!(tuning.attention.query_tile, 1);
        assert_eq!(tuning.duplicate_batch_width, 7);
        assert_eq!(tuning.decode.row_budget, 35);
        assert_eq!(tuning.decode.maximum_steps, 5);
        assert_eq!(tuning.decode.selection_threads, 512);
        assert_eq!(tuning.gemm.custom_maximum_rows, 3);
        assert_eq!(
            tuning.profile_label(),
            "generic(width=7,decode-rows=35,decode-steps=5,select-threads=512,custom-gemm-max=3)"
        );
    }
}
