//! Stable wrappers used by the repository's allocation-free microbenchmark driver.
//!
//! This module is deliberately feature-gated. It exposes fixed synthetic
//! workloads without making the internal tensor representation part of the
//! production API.

use crate::tensor::{Matrix, attention, residual_layer_norm, select_token, ssru_update_layer_norm};

pub fn attention_384(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    sequence: usize,
) -> Result<Vec<f32>, String> {
    attention(
        query,
        key,
        value,
        &[sequence],
        1,
        sequence,
        sequence,
        384,
        8,
    )
}

pub struct LayerNorm384 {
    scale: Matrix,
    bias: Matrix,
}

impl LayerNorm384 {
    pub fn new() -> Self {
        Self {
            scale: Matrix::new(vec![1.0; 384], 1, 384).unwrap(),
            bias: Matrix::new(vec![0.0; 384], 1, 384).unwrap(),
        }
    }

    pub fn residual(&self, input: &[f32], residual: &[f32]) -> Result<Vec<f32>, String> {
        residual_layer_norm(
            input,
            residual,
            &self.scale,
            &self.bias,
            input.len() / 384,
            384,
        )
    }

    pub fn ssru(
        &self,
        candidate: &[f32],
        forget: &[f32],
        state: &mut [f32],
        residual: &[f32],
    ) -> Result<Vec<f32>, String> {
        ssru_update_layer_norm(
            candidate,
            forget,
            state,
            residual,
            &self.scale,
            &self.bias,
            candidate.len() / 384,
            384,
        )
    }
}

impl Default for LayerNorm384 {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Shortlist384 {
    embedding: Matrix,
    bias: Matrix,
}

impl Shortlist384 {
    pub fn new(rows: usize) -> Result<Self, String> {
        let values = (0..rows * 384)
            .map(|index| ((index * 17 % 257) as f32 - 128.0) / 128.0)
            .collect();
        Ok(Self {
            embedding: Matrix::new(values, rows, 384)?,
            bias: Matrix::new(vec![0.0; rows], 1, rows)?,
        })
    }

    pub fn score(&self, decoder: &[f32], candidates: &[u32]) -> Result<u32, String> {
        select_token(decoder, &self.embedding, &self.bias, candidates)
    }
}
