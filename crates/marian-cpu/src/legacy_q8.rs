use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use memmap2::MmapOptions;

use crate::{Q8Error, Q8Linear};

const MARIAN_BINARY_VERSION: u64 = 1;
const TYPE_INT8: u64 = 0x0101;
const TYPE_FLOAT32: u64 = 0x0404;
const TYPE_INTGEMM8: u64 = 0x4101;
const MAXIMUM_FILE_BYTES: usize = 128 * 1024 * 1024;
const MAXIMUM_TENSORS: usize = 4096;
const MAXIMUM_NAME_BYTES: usize = 16 * 1024;
const MAXIMUM_RANK: usize = 8;
const DATA_ALIGNMENT: usize = 256;
const ACTIVATION_SCALE_SUFFIX: &str = "_QuantMultA";

/// Tensor type tags used by Marian binary format v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarianTensorType {
    Int8,
    Float32,
    Intgemm8,
}

impl MarianTensorType {
    fn from_raw(raw: u64) -> Result<Self, Q8Error> {
        match raw {
            TYPE_INT8 => Ok(Self::Int8),
            TYPE_FLOAT32 => Ok(Self::Float32),
            TYPE_INTGEMM8 => Ok(Self::Intgemm8),
            _ => Err(Q8Error::InvalidFormat(format!(
                "unsupported tensor type 0x{raw:04x}"
            ))),
        }
    }
}

/// Logical tensor data decoded from a Marian binary v1 payload.
#[derive(Debug)]
pub enum MarianTensorData {
    Int8(Vec<i8>),
    Float32(Vec<f32>),
    /// Symmetric Q8 values followed on disk by Marian's quantization
    /// multiplier. A real weight is approximately `value / quant_mult`.
    Intgemm8 {
        values: Vec<i8>,
        quant_mult: f32,
    },
}

/// A tensor from a Marian binary v1 model.
#[derive(Debug)]
pub struct MarianTensor {
    name: String,
    tensor_type: MarianTensorType,
    shape: Vec<usize>,
    data_length: usize,
    data: MarianTensorData,
}

impl MarianTensor {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn tensor_type(&self) -> MarianTensorType {
        self.tensor_type
    }

    /// Return Marian's logical header shape.
    ///
    /// For dense intgemm tensors this is `[input, output]`, while the Q8 bytes
    /// are already transposed into `[output, input]`. `*_Wemb` tensors are the
    /// exception: their header and bytes are both `[vocabulary, dimension]`.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Padded byte length recorded in the v1 header.
    pub fn data_length(&self) -> usize {
        self.data_length
    }

    pub fn data(&self) -> &MarianTensorData {
        &self.data
    }

    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn as_float32(&self) -> Result<&[f32], Q8Error> {
        match &self.data {
            MarianTensorData::Float32(values) => Ok(values),
            _ => Err(Q8Error::tensor(
                &self.name,
                format!("expected float32, got {:?}", self.tensor_type),
            )),
        }
    }

    pub fn as_intgemm8(&self) -> Result<(&[i8], f32), Q8Error> {
        match &self.data {
            MarianTensorData::Intgemm8 { values, quant_mult } => Ok((values, *quant_mult)),
            _ => Err(Q8Error::tensor(
                &self.name,
                format!("expected intgemm8, got {:?}", self.tensor_type),
            )),
        }
    }
}

/// Counts returned after validating cross-tensor Q8 invariants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q8ValidationReport {
    pub dense_linears: usize,
    pub embeddings: usize,
    pub activation_scales: usize,
}

/// An owned, strictly parsed Marian binary v1 model.
///
/// The parser copies Q8 bytes but never dequantizes a complete tensor. The
/// owned representation avoids unsafe self-referential mmap structures and is
/// still close to the original model size.
#[derive(Debug)]
pub struct MarianBinaryModel {
    tensors: Vec<MarianTensor>,
    by_name: HashMap<String, usize>,
}

impl MarianBinaryModel {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Q8Error> {
        let path = path.as_ref();
        let metadata = fs::metadata(path).map_err(|source| Q8Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let file_len = usize::try_from(metadata.len()).map_err(|_| {
            Q8Error::InvalidFormat(format!(
                "model {} is too large for this platform",
                path.display()
            ))
        })?;
        if file_len > MAXIMUM_FILE_BYTES {
            return Err(Q8Error::InvalidFormat(format!(
                "model {} is {file_len} bytes; maximum is {MAXIMUM_FILE_BYTES}",
                path.display()
            )));
        }
        let file = fs::File::open(path).map_err(|source| Q8Error::Io {
            path: PathBuf::from(path),
            source,
        })?;
        // SAFETY: The mapping is read-only, its length was bounded above, and
        // parsing completes before `file` or the mapping is dropped. The
        // owned tensor representation never keeps references into the map.
        let bytes = unsafe { MmapOptions::new().map(&file) }.map_err(|source| Q8Error::Io {
            path: PathBuf::from(path),
            source,
        })?;
        debug_assert_eq!(bytes.len(), file_len);
        Self::parse(&bytes)
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, Q8Error> {
        if bytes.len() > MAXIMUM_FILE_BYTES {
            return Err(Q8Error::InvalidFormat(format!(
                "model is {} bytes; maximum is {MAXIMUM_FILE_BYTES}",
                bytes.len()
            )));
        }
        let mut reader = Reader::new(bytes);
        let version = reader.read_u64("binary version")?;
        if version != MARIAN_BINARY_VERSION {
            return Err(Q8Error::InvalidFormat(format!(
                "unsupported version {version}; expected {MARIAN_BINARY_VERSION}"
            )));
        }
        let tensor_count = reader.read_usize_u64("tensor count")?;
        if tensor_count == 0 || tensor_count > MAXIMUM_TENSORS {
            return Err(Q8Error::InvalidFormat(format!(
                "tensor count {tensor_count} is outside 1..={MAXIMUM_TENSORS}"
            )));
        }

        let mut headers = Vec::with_capacity(tensor_count);
        for index in 0..tensor_count {
            let name_length = reader.read_usize_u64(&format!("header {index} name length"))?;
            let tensor_type = MarianTensorType::from_raw(
                reader.read_u64(&format!("header {index} tensor type"))?,
            )?;
            let shape_length = reader.read_usize_u64(&format!("header {index} shape length"))?;
            let data_length = reader.read_usize_u64(&format!("header {index} data length"))?;
            if !(2..=MAXIMUM_NAME_BYTES).contains(&name_length) {
                return Err(Q8Error::InvalidFormat(format!(
                    "header {index} name length {name_length} is outside 2..={MAXIMUM_NAME_BYTES}"
                )));
            }
            if shape_length == 0 || shape_length > MAXIMUM_RANK {
                return Err(Q8Error::InvalidFormat(format!(
                    "header {index} rank {shape_length} is outside 1..={MAXIMUM_RANK}"
                )));
            }
            if data_length == 0 || data_length > MAXIMUM_FILE_BYTES {
                return Err(Q8Error::InvalidFormat(format!(
                    "header {index} has invalid data length {data_length}"
                )));
            }
            headers.push(Header {
                name_length,
                tensor_type,
                shape_length,
                data_length,
            });
        }

        let mut names = Vec::with_capacity(tensor_count);
        let mut by_name = HashMap::with_capacity(tensor_count);
        for (index, header) in headers.iter().enumerate() {
            let raw = reader.take(header.name_length, &format!("tensor {index} name"))?;
            if raw.last() != Some(&0) || raw[..raw.len() - 1].contains(&0) {
                return Err(Q8Error::InvalidFormat(format!(
                    "tensor {index} name is not one NUL-terminated string"
                )));
            }
            let name = std::str::from_utf8(&raw[..raw.len() - 1])
                .map_err(|error| {
                    Q8Error::InvalidFormat(format!("tensor {index} name is not UTF-8: {error}"))
                })?
                .to_owned();
            if name.is_empty() {
                return Err(Q8Error::InvalidFormat(format!(
                    "tensor {index} has an empty name"
                )));
            }
            if by_name.insert(name.clone(), index).is_some() {
                return Err(Q8Error::InvalidFormat(format!(
                    "duplicate tensor name {name}"
                )));
            }
            names.push(name);
        }

        let mut shapes = Vec::with_capacity(tensor_count);
        for (index, header) in headers.iter().enumerate() {
            let mut shape = Vec::with_capacity(header.shape_length);
            for dimension in 0..header.shape_length {
                let value = reader.read_i32(&format!("tensor {index} shape[{dimension}]"))?;
                let value = usize::try_from(value).map_err(|_| {
                    Q8Error::InvalidFormat(format!(
                        "tensor {} ({}) has non-positive shape dimension {value}",
                        index, names[index]
                    ))
                })?;
                if value == 0 {
                    return Err(Q8Error::InvalidFormat(format!(
                        "tensor {} ({}) has a zero shape dimension",
                        index, names[index]
                    )));
                }
                shape.push(value);
            }
            checked_elements(&names[index], &shape)?;
            shapes.push(shape);
        }

        let offset = reader.read_usize_u64("data offset")?;
        let padding = reader.take(offset, "header padding")?;
        if padding.iter().any(|&byte| byte != 0) {
            return Err(Q8Error::InvalidFormat(
                "header alignment padding contains non-zero bytes".into(),
            ));
        }
        if reader.position() % DATA_ALIGNMENT != 0 {
            return Err(Q8Error::InvalidFormat(format!(
                "tensor data begins at {}, which is not {DATA_ALIGNMENT}-byte aligned",
                reader.position()
            )));
        }

        let mut tensors = Vec::with_capacity(tensor_count);
        for index in 0..tensor_count {
            let header = &headers[index];
            let name = &names[index];
            let shape = &shapes[index];
            let elements = checked_elements(name, shape)?;
            let payload = reader.take(header.data_length, &format!("tensor {name} payload"))?;
            let logical_bytes = logical_data_length(name, header.tensor_type, elements)?;
            let expected_data_length = match header.tensor_type {
                // The only raw int8 record in supported Marian Q8 models is the YAML
                // blob, written without allocator padding.
                MarianTensorType::Int8 => logical_bytes,
                MarianTensorType::Float32 | MarianTensorType::Intgemm8 => {
                    align_up(logical_bytes, DATA_ALIGNMENT).ok_or_else(|| {
                        Q8Error::tensor(name, "aligned record length overflows the address space")
                    })?
                }
            };
            if header.data_length != expected_data_length {
                return Err(Q8Error::tensor(
                    name,
                    format!(
                        "record length is {}, expected {expected_data_length} for {logical_bytes} logical bytes",
                        header.data_length
                    ),
                ));
            }
            if payload[logical_bytes..].iter().any(|&byte| byte != 0) {
                return Err(Q8Error::tensor(
                    name,
                    "record padding contains non-zero bytes",
                ));
            }
            let data = match header.tensor_type {
                MarianTensorType::Int8 => {
                    if payload.len() < elements {
                        return Err(Q8Error::tensor(
                            name,
                            format!(
                                "int8 payload has {} bytes, expected at least {elements}",
                                payload.len()
                            ),
                        ));
                    }
                    MarianTensorData::Int8(
                        payload[..elements].iter().map(|&byte| byte as i8).collect(),
                    )
                }
                MarianTensorType::Float32 => {
                    let logical_bytes =
                        elements.checked_mul(size_of::<f32>()).ok_or_else(|| {
                            Q8Error::tensor(name, "float32 byte length overflows the address space")
                        })?;
                    if payload.len() < logical_bytes {
                        return Err(Q8Error::tensor(
                            name,
                            format!(
                                "float32 payload has {} bytes, expected at least {logical_bytes}",
                                payload.len()
                            ),
                        ));
                    }
                    let mut values = Vec::with_capacity(elements);
                    for bytes in payload[..logical_bytes].chunks_exact(4) {
                        let value = f32::from_le_bytes(bytes.try_into().expect("four byte chunk"));
                        if !value.is_finite() {
                            return Err(Q8Error::tensor(name, "float32 payload is not finite"));
                        }
                        values.push(value);
                    }
                    MarianTensorData::Float32(values)
                }
                MarianTensorType::Intgemm8 => {
                    let required = elements.checked_add(size_of::<f32>()).ok_or_else(|| {
                        Q8Error::tensor(name, "intgemm8 byte length overflows the address space")
                    })?;
                    if payload.len() < required {
                        return Err(Q8Error::tensor(
                            name,
                            format!(
                                "intgemm8 payload has {} bytes, expected at least {required}",
                                payload.len()
                            ),
                        ));
                    }
                    let values = payload[..elements]
                        .iter()
                        .map(|&byte| byte as i8)
                        .collect::<Vec<_>>();
                    if values.contains(&i8::MIN) {
                        return Err(Q8Error::tensor(
                            name,
                            "symmetric intgemm8 payload contains -128",
                        ));
                    }
                    let quant_mult = f32::from_le_bytes(
                        payload[elements..elements + 4]
                            .try_into()
                            .expect("four byte scale"),
                    );
                    if !quant_mult.is_finite() || quant_mult <= 0.0 {
                        return Err(Q8Error::tensor(
                            name,
                            format!(
                                "intgemm8 quantization multiplier must be finite and positive, got {quant_mult}"
                            ),
                        ));
                    }
                    MarianTensorData::Intgemm8 { values, quant_mult }
                }
            };
            tensors.push(MarianTensor {
                name: name.clone(),
                tensor_type: header.tensor_type,
                shape: shape.clone(),
                data_length: header.data_length,
                data,
            });
        }

        if reader.remaining() != 0 {
            return Err(Q8Error::InvalidFormat(format!(
                "{} trailing bytes remain after the final tensor",
                reader.remaining()
            )));
        }
        Ok(Self { tensors, by_name })
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn tensors(&self) -> &[MarianTensor] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Result<&MarianTensor, Q8Error> {
        self.by_name
            .get(name)
            .map(|&index| &self.tensors[index])
            .ok_or_else(|| Q8Error::MissingTensor(name.to_owned()))
    }

    /// Validate Q8 tensors and all activation-scale relationships.
    pub fn validate_q8(&self) -> Result<Q8ValidationReport, Q8Error> {
        let mut report = Q8ValidationReport {
            dense_linears: 0,
            embeddings: 0,
            activation_scales: 0,
        };

        for tensor in &self.tensors {
            if let Some(weight_name) = tensor.name.strip_suffix(ACTIVATION_SCALE_SUFFIX) {
                self.scalar_activation_scale(&tensor.name)?;
                let weight = self.tensor(weight_name)?;
                if !matches!(weight.data, MarianTensorData::Intgemm8 { .. }) {
                    return Err(Q8Error::tensor(
                        &tensor.name,
                        format!("activation scale refers to non-Q8 tensor {weight_name}"),
                    ));
                }
                report.activation_scales += 1;
                continue;
            }

            if !matches!(tensor.data, MarianTensorData::Intgemm8 { .. }) {
                continue;
            }
            if tensor.shape.len() != 2 {
                return Err(Q8Error::tensor(
                    &tensor.name,
                    format!("Q8 weight must have rank 2, got shape {:?}", tensor.shape),
                ));
            }
            if tensor.name.ends_with("_Wemb") {
                report.embeddings += 1;
            } else {
                self.activation_scale_for(&tensor.name)?;
                report.dense_linears += 1;
            }
        }

        if report.dense_linears + report.embeddings == 0 {
            return Err(Q8Error::InvalidFormat(
                "model contains no intgemm8 weight tensors".into(),
            ));
        }
        Ok(report)
    }

    /// Resolve the static activation multiplier paired with a Q8 weight.
    ///
    /// Normal alpha tensors are f32 scalars. Some published models accidentally
    /// quantized `decoder_Wemb_QuantMultA`; this method deliberately normalizes
    /// that scalar as `q / quant_mult` at the format boundary.
    pub fn activation_scale_for(&self, weight_name: &str) -> Result<f32, Q8Error> {
        let scale_name = format!("{weight_name}{ACTIVATION_SCALE_SUFFIX}");
        self.scalar_activation_scale(&scale_name)
    }

    /// Build a dense operator from a Marian tensor whose logical header shape
    /// is `[input, output]` and whose bytes are `[output, input]`.
    pub fn dense_linear(
        &self,
        weight_name: &str,
        bias_name: Option<&str>,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<Q8Linear, Q8Error> {
        let weight = self.tensor(weight_name)?;
        if weight.shape != [input_dim, output_dim] {
            return Err(Q8Error::tensor(
                weight_name,
                format!(
                    "dense header shape is {:?}, expected [{input_dim}, {output_dim}]",
                    weight.shape
                ),
            ));
        }
        let (values, weight_quant_mult) = weight.as_intgemm8()?;
        let activation_quant_mult = self.activation_scale_for(weight_name)?;
        let bias = self.load_bias(bias_name, output_dim)?;
        Q8Linear::new(
            weight_name,
            input_dim,
            output_dim,
            values.to_vec(),
            activation_quant_mult,
            weight_quant_mult,
            bias,
        )
    }

    /// Build the tied decoder output projection from row-major Wemb bytes.
    pub fn tied_output_linear(
        &self,
        embedding_name: &str,
        bias_name: Option<&str>,
        model_dim: usize,
        vocabulary_size: usize,
    ) -> Result<Q8Linear, Q8Error> {
        let weight = self.tensor(embedding_name)?;
        if weight.shape != [vocabulary_size, model_dim] {
            return Err(Q8Error::tensor(
                embedding_name,
                format!(
                    "embedding header shape is {:?}, expected [{vocabulary_size}, {model_dim}]",
                    weight.shape
                ),
            ));
        }
        let (values, weight_quant_mult) = weight.as_intgemm8()?;
        let activation_quant_mult = self.activation_scale_for(embedding_name)?;
        let bias = self.load_bias(bias_name, vocabulary_size)?;
        Q8Linear::new(
            embedding_name,
            model_dim,
            vocabulary_size,
            values.to_vec(),
            activation_quant_mult,
            weight_quant_mult,
            bias,
        )
    }

    /// Dequantize exactly one embedding row, never the complete table.
    pub fn embedding_row(&self, name: &str, row: usize) -> Result<Vec<f32>, Q8Error> {
        let tensor = self.tensor(name)?;
        if tensor.shape.len() != 2 || !tensor.name.ends_with("_Wemb") {
            return Err(Q8Error::tensor(
                name,
                format!(
                    "expected a rank-2 *_Wemb tensor, got shape {:?}",
                    tensor.shape
                ),
            ));
        }
        let rows = tensor.shape[0];
        let columns = tensor.shape[1];
        if row >= rows {
            return Err(Q8Error::tensor(
                name,
                format!("embedding row {row} exceeds row count {rows}"),
            ));
        }
        let (values, quant_mult) = tensor.as_intgemm8()?;
        Ok(values[row * columns..(row + 1) * columns]
            .iter()
            .map(|&value| f32::from(value) / quant_mult)
            .collect())
    }

    fn scalar_activation_scale(&self, name: &str) -> Result<f32, Q8Error> {
        let tensor = self.tensor(name)?;
        if tensor.elements() != 1 {
            return Err(Q8Error::tensor(
                name,
                format!(
                    "activation scale must be scalar, got shape {:?}",
                    tensor.shape
                ),
            ));
        }
        let scale = match &tensor.data {
            MarianTensorData::Float32(values) => values[0],
            MarianTensorData::Intgemm8 { values, quant_mult } => f32::from(values[0]) / quant_mult,
            MarianTensorData::Int8(_) => {
                return Err(Q8Error::tensor(
                    name,
                    "activation scale must be float32 or intgemm8",
                ));
            }
        };
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Q8Error::tensor(
                name,
                format!("activation scale must be finite and positive, got {scale}"),
            ));
        }
        Ok(scale)
    }

    fn load_bias(
        &self,
        name: Option<&str>,
        output_dim: usize,
    ) -> Result<Option<Vec<f32>>, Q8Error> {
        let Some(name) = name else {
            return Ok(None);
        };
        let tensor = self.tensor(name)?;
        if tensor.shape != [1, output_dim] && tensor.shape != [output_dim] {
            return Err(Q8Error::tensor(
                name,
                format!(
                    "bias shape is {:?}, expected [1, {output_dim}] or [{output_dim}]",
                    tensor.shape
                ),
            ));
        }
        Ok(Some(tensor.as_float32()?.to_vec()))
    }
}

#[derive(Debug)]
struct Header {
    name_length: usize,
    tensor_type: MarianTensorType,
    shape_length: usize,
    data_length: usize,
}

fn checked_elements(name: &str, shape: &[usize]) -> Result<usize, Q8Error> {
    shape.iter().try_fold(1_usize, |elements, &dimension| {
        elements
            .checked_mul(dimension)
            .ok_or_else(|| Q8Error::tensor(name, "shape element count overflows the address space"))
    })
}

fn logical_data_length(
    name: &str,
    tensor_type: MarianTensorType,
    elements: usize,
) -> Result<usize, Q8Error> {
    match tensor_type {
        MarianTensorType::Int8 => Ok(elements),
        MarianTensorType::Float32 => elements.checked_mul(size_of::<f32>()).ok_or_else(|| {
            Q8Error::tensor(name, "float32 byte length overflows the address space")
        }),
        MarianTensorType::Intgemm8 => elements.checked_add(size_of::<f32>()).ok_or_else(|| {
            Q8Error::tensor(name, "intgemm8 byte length overflows the address space")
        }),
    }
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    fn take(&mut self, length: usize, label: &str) -> Result<&'a [u8], Q8Error> {
        let end = self.position.checked_add(length).ok_or_else(|| {
            Q8Error::InvalidFormat(format!("{label} byte range overflows the address space"))
        })?;
        let result = self.bytes.get(self.position..end).ok_or_else(|| {
            Q8Error::InvalidFormat(format!(
                "truncated {label}: need {length} bytes at {}, only {} remain",
                self.position,
                self.remaining()
            ))
        })?;
        self.position = end;
        Ok(result)
    }

    fn read_u64(&mut self, label: &str) -> Result<u64, Q8Error> {
        Ok(u64::from_le_bytes(
            self.take(8, label)?.try_into().expect("eight byte integer"),
        ))
    }

    fn read_usize_u64(&mut self, label: &str) -> Result<usize, Q8Error> {
        usize::try_from(self.read_u64(label)?).map_err(|_| {
            Q8Error::InvalidFormat(format!("{label} does not fit this platform's usize"))
        })
    }

    fn read_i32(&mut self, label: &str) -> Result<i32, Q8Error> {
        Ok(i32::from_le_bytes(
            self.take(4, label)?.try_into().expect("four byte integer"),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        MarianBinaryModel, MarianTensorData, Q8ValidationReport, TYPE_FLOAT32, TYPE_INT8,
        TYPE_INTGEMM8,
    };
    use crate::quantize_symmetric_u8;

    #[derive(Clone)]
    struct Item {
        name: &'static str,
        tensor_type: u64,
        shape: Vec<i32>,
        payload: Vec<u8>,
    }

    fn f32_payload(values: &[f32], padded: bool) -> Vec<u8> {
        let mut bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        if padded {
            bytes.resize(256, 0);
        }
        bytes
    }

    fn q8_payload(values: &[i8], quant_mult: f32) -> Vec<u8> {
        let mut bytes = values.iter().map(|&value| value as u8).collect::<Vec<_>>();
        bytes.extend_from_slice(&quant_mult.to_le_bytes());
        bytes.resize(bytes.len().div_ceil(256) * 256, 0);
        bytes
    }

    fn build(items: &[Item]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for item in items {
            bytes.extend_from_slice(&((item.name.len() + 1) as u64).to_le_bytes());
            bytes.extend_from_slice(&item.tensor_type.to_le_bytes());
            bytes.extend_from_slice(&(item.shape.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&(item.payload.len() as u64).to_le_bytes());
        }
        for item in items {
            bytes.extend_from_slice(item.name.as_bytes());
            bytes.push(0);
        }
        for item in items {
            for &dimension in &item.shape {
                bytes.extend_from_slice(&dimension.to_le_bytes());
            }
        }
        let next_position = ((bytes.len() + 8) / 256 + 1) * 256;
        let offset = next_position - bytes.len() - 8;
        bytes.extend_from_slice(&(offset as u64).to_le_bytes());
        bytes.resize(next_position, 0);
        for item in items {
            bytes.extend_from_slice(&item.payload);
        }
        bytes
    }

    fn valid_items() -> Vec<Item> {
        vec![
            Item {
                name: "encoder_l1_self_Wq",
                tensor_type: TYPE_INTGEMM8,
                shape: vec![2, 3],
                // Dense bytes are canonical [output=3, input=2].
                payload: q8_payload(&[2, -3, 5, 7, -11, 13], 8.0),
            },
            Item {
                name: "encoder_l1_self_Wq_QuantMultA",
                tensor_type: TYPE_FLOAT32,
                shape: vec![1, 1],
                payload: f32_payload(&[4.0], true),
            },
            Item {
                name: "encoder_l1_self_bq",
                tensor_type: TYPE_FLOAT32,
                shape: vec![1, 3],
                payload: f32_payload(&[0.25, -0.5, 1.0], true),
            },
            Item {
                name: "encoder_Wemb",
                tensor_type: TYPE_INTGEMM8,
                shape: vec![2, 2],
                payload: q8_payload(&[8, -8, 4, -4], 4.0),
            },
            Item {
                name: "decoder_Wemb",
                tensor_type: TYPE_INTGEMM8,
                shape: vec![3, 2],
                payload: q8_payload(&[1, 2, 3, 4, 5, 6], 2.0),
            },
            Item {
                name: "decoder_Wemb_QuantMultA",
                tensor_type: TYPE_INTGEMM8,
                shape: vec![1, 1],
                payload: q8_payload(&[127], 15.339_54),
            },
            Item {
                name: "decoder_ff_logit_out_b",
                tensor_type: TYPE_FLOAT32,
                shape: vec![1, 3],
                payload: f32_payload(&[0.0, 0.25, -0.25], true),
            },
            Item {
                name: "special:model.yml",
                tensor_type: TYPE_INT8,
                shape: vec![3],
                payload: b"abc".to_vec(),
            },
        ]
    }

    #[test]
    fn parses_v1_and_validates_q8_roles() {
        let model = MarianBinaryModel::parse(&build(&valid_items())).unwrap();
        assert_eq!(model.len(), 8);
        assert_eq!(
            model.validate_q8().unwrap(),
            Q8ValidationReport {
                dense_linears: 1,
                embeddings: 2,
                activation_scales: 2,
            }
        );
        let weight = model.tensor("encoder_l1_self_Wq").unwrap();
        assert_eq!(weight.shape(), [2, 3]);
        assert_eq!(weight.data_length(), 256);
        let (values, quant_mult) = weight.as_intgemm8().unwrap();
        assert_eq!(values, [2, -3, 5, 7, -11, 13]);
        assert_eq!(quant_mult, 8.0);
        assert!(matches!(
            model.tensor("special:model.yml").unwrap().data(),
            MarianTensorData::Int8(values) if values == &[97, 98, 99]
        ));
    }

    #[test]
    fn normalizes_accidentally_quantized_decoder_alpha() {
        let model = MarianBinaryModel::parse(&build(&valid_items())).unwrap();
        let alpha = model.activation_scale_for("decoder_Wemb").unwrap();
        assert!((alpha - 8.279_258).abs() < 1.0e-5);

        let projection = model
            .tied_output_linear("decoder_Wemb", Some("decoder_ff_logit_out_b"), 2, 3)
            .unwrap();
        assert_eq!(projection.input_dim(), 2);
        assert_eq!(projection.output_dim(), 3);
    }

    #[test]
    fn constructs_dense_linear_without_dequantizing_weights() {
        let model = MarianBinaryModel::parse(&build(&valid_items())).unwrap();
        let linear = model
            .dense_linear("encoder_l1_self_Wq", Some("encoder_l1_self_bq"), 2, 3)
            .unwrap();
        assert_eq!(linear.weights(), [2, -3, 5, 7, -11, 13]);
        assert_eq!(linear.run(&[0.25, -0.5], 1).unwrap().len(), 3);
        assert_eq!(model.embedding_row("encoder_Wemb", 1).unwrap(), [1.0, -1.0]);
        assert!(model.embedding_row("encoder_Wemb", 2).is_err());
    }

    #[test]
    fn rejects_truncation_trailing_data_and_wrong_version() {
        let bytes = build(&valid_items());
        assert!(MarianBinaryModel::parse(&bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(MarianBinaryModel::parse(&trailing).is_err());
        let mut wrong_version = bytes;
        wrong_version[..8].copy_from_slice(&2_u64.to_le_bytes());
        assert!(MarianBinaryModel::parse(&wrong_version).is_err());
    }

    #[test]
    fn rejects_invalid_names_shapes_scales_and_q8_range() {
        let mut invalid_shape = valid_items();
        invalid_shape[0].shape[0] = -1;
        assert!(MarianBinaryModel::parse(&build(&invalid_shape)).is_err());

        let mut invalid_scale = valid_items();
        invalid_scale[0].payload = q8_payload(&[2, -3, 5, 7, -11, 13], f32::NAN);
        assert!(MarianBinaryModel::parse(&build(&invalid_scale)).is_err());

        let mut invalid_q8 = valid_items();
        invalid_q8[0].payload = q8_payload(&[i8::MIN, -3, 5, 7, -11, 13], 8.0);
        assert!(MarianBinaryModel::parse(&build(&invalid_q8)).is_err());

        let duplicate = vec![valid_items()[0].clone(), valid_items()[0].clone()];
        assert!(MarianBinaryModel::parse(&build(&duplicate)).is_err());
    }

    #[test]
    fn validation_rejects_missing_or_orphan_activation_scales() {
        let mut missing = valid_items();
        missing.remove(1);
        let model = MarianBinaryModel::parse(&build(&missing)).unwrap();
        assert!(model.validate_q8().is_err());

        let mut orphan = valid_items();
        orphan[1].name = "absent_W_QuantMultA";
        let model = MarianBinaryModel::parse(&build(&orphan)).unwrap();
        assert!(model.validate_q8().is_err());
    }

    #[test]
    #[ignore = "set MARIAN_Q8_MODEL to a real Marian intgemm.alphas.bin artifact"]
    fn validates_real_marian_q8_artifact_inventory() {
        let path = std::env::var("MARIAN_Q8_MODEL").expect("MARIAN_Q8_MODEL");
        let model = MarianBinaryModel::open(path).unwrap();
        assert_eq!(model.len(), 253);
        assert_eq!(
            model.validate_q8().unwrap(),
            Q8ValidationReport {
                dense_linears: 68,
                embeddings: 2,
                activation_scales: 69,
            }
        );

        // Exercise both the real Q8 payload and the reusable packed-B path.
        let linear = model
            .dense_linear("encoder_l1_self_Wq", Some("encoder_l1_self_bq"), 384, 384)
            .unwrap();
        let input = (0..768)
            .map(|index| ((index % 23) as f32 - 11.0) / 32.0)
            .collect::<Vec<_>>();
        let actual = linear.run(&input, 2).unwrap();
        let quantized = quantize_symmetric_u8(&input, linear.activation_quant_mult()).unwrap();
        let bias = model
            .tensor("encoder_l1_self_bq")
            .unwrap()
            .as_float32()
            .unwrap();
        let inverse_scale = 1.0 / (linear.activation_quant_mult() * linear.weight_quant_mult());
        let mut expected = Vec::with_capacity(actual.len());
        for row in quantized.chunks_exact(384) {
            for (column, &bias_value) in bias.iter().enumerate() {
                let accumulator = row
                    .iter()
                    .zip(&linear.weights()[column * 384..(column + 1) * 384])
                    .map(|(&activation, &weight)| (i32::from(activation) - 127) * i32::from(weight))
                    .sum::<i32>();
                expected.push(accumulator as f32 * inverse_scale + bias_value);
            }
        }
        assert_eq!(actual, expected);
    }
}
