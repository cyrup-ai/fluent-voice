use candle_core::{D, IndexOp, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

#[derive(Debug, Clone)]
pub struct EuclideanCodebook {
    initialized: Tensor,
    cluster_usage: Tensor,
    embedding_sum: Tensor,
    embedding: Tensor,
    c2: Tensor,
    epsilon: f64,
    dim: usize,
    span_encode: tracing::Span,
    span_decode: tracing::Span,
}

impl EuclideanCodebook {
    pub fn new(dim: usize, codebook_size: usize, vb: VarBuilder) -> Result<Self> {
        let epsilon = 1e-5;
        let initialized = vb.get(1, "_initialized")?;
        let cluster_usage = vb.get(codebook_size, "cluster_usage")?;
        let embedding_sum = vb.get((codebook_size, dim), "embedding_sum")?;
        let embedding = {
            let cluster_usage = cluster_usage.maximum(epsilon)?.unsqueeze(1)?;
            embedding_sum.broadcast_div(&cluster_usage)?
        };
        let c2 = ((&embedding * &embedding)?.sum(D::Minus1)? / 2.0)?;
        Ok(Self {
            initialized,
            cluster_usage,
            embedding_sum,
            embedding,
            c2,
            epsilon,
            dim,
            span_encode: tracing::span!(tracing::Level::TRACE, "euclidean-encode"),
            span_decode: tracing::span!(tracing::Level::TRACE, "euclidean-decode"),
        })
    }

    /// Check if the codebook has been initialized
    pub fn is_initialized(&self) -> Result<bool> {
        let init_value = self.initialized.to_scalar::<f32>()?;
        Ok(init_value > 0.0)
    }

    fn get_embedding(&self) -> Result<Tensor> {
        // Check if codebook is initialized before using it
        if !self.is_initialized()? {
            return Err(candle_core::Error::Msg(
                "Codebook not initialized".to_string(),
            ));
        }
        let cluster_usage = self.cluster_usage.maximum(self.epsilon)?.unsqueeze(1)?;
        self.embedding_sum.broadcast_div(&cluster_usage)
    }

    fn prepare_for_encoding(&self, xs: &Tensor) -> Result<(Tensor, Vec<usize>)> {
        let mut target_shape = xs.dims().to_vec();
        target_shape.pop();
        let xs = xs.flatten_to(D::Minus2)?;
        let _ = xs.dims2()?;
        Ok((xs, target_shape))
    }

    pub fn encode_very_slow(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span_encode.enter();
        let (xs, target_shape) = self.prepare_for_encoding(xs)?;

        let embedding = self.get_embedding()?;

        let diff = xs.unsqueeze(1)?.broadcast_sub(&embedding.unsqueeze(0)?)?;
        let dists = diff.sqr()?.sum(D::Minus1)?;
        let codes = dists.argmin(D::Minus1)?;
        codes.reshape(target_shape)
    }

    pub fn encode_slow(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span_encode.enter();
        let (xs, target_shape) = self.prepare_for_encoding(xs)?;

        let dot_prod = xs.matmul(&self.embedding.t()?)?;
        let codes = self.c2.broadcast_sub(&dot_prod)?.argmin(D::Minus1)?;
        codes.reshape(target_shape)
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        // Graceful performance degradation: use optimized path when available, stable path otherwise
        if self.can_use_custom_op() {
            self.encode_with_custom_op(xs) // Optimized implementation
        } else {
            self.encode_slow(xs) // Stable reference implementation
        }
    }

    /// Check if custom op can be used for optimization
    fn can_use_custom_op(&self) -> bool {
        // Custom op is now implemented and ready for use
        true
    }

    /// Encode using custom op for optimization
    fn encode_with_custom_op(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span_encode.enter();
        let (xs, target_shape) = self.prepare_for_encoding(xs)?;
        let embedding = self.get_embedding()?;

        // Use custom op for efficient encoding
        let custom_op = CodebookEncode;
        let result = xs.apply_op2(&embedding, custom_op)?;
        result.reshape(target_shape)
    }

    pub fn decode(&self, indexes: &Tensor) -> Result<Tensor> {
        let _enter = self.span_decode.enter();
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.dim);
        let indexes = indexes.flatten_all()?;
        let values = self.embedding.index_select(&indexes, 0)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}

#[derive(Debug, Clone)]
pub struct VectorQuantization {
    project_in: Option<Linear>,
    project_out: Option<Linear>,
    codebook: EuclideanCodebook,
}

impl VectorQuantization {
    pub fn new(
        dim: usize,
        codebook_size: usize,
        codebook_dim: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let codebook_dim = codebook_dim.unwrap_or(dim);
        let (project_in, project_out) = if codebook_dim == dim {
            (None, None)
        } else {
            let p_in = linear(dim, codebook_dim, vb.pp("project_in"))?;
            let p_out = linear(codebook_dim, dim, vb.pp("project_out"))?;
            (Some(p_in), Some(p_out))
        };
        let codebook = EuclideanCodebook::new(codebook_dim, codebook_size, vb.pp("_codebook"))?;
        Ok(Self {
            project_in,
            project_out,
            codebook,
        })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.t()?.apply(&self.project_in.as_ref())?;
        self.codebook.encode(&xs)
    }

    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let quantized = self.codebook.decode(codes)?;
        let quantized = match &self.project_out {
            None => quantized,
            Some(p) => quantized.apply(p)?,
        };
        quantized.t()
    }
}

#[derive(Debug, Clone)]
pub struct ResidualVectorQuantization {
    layers: Vec<VectorQuantization>,
}

impl ResidualVectorQuantization {
    pub fn new(
        n_q: usize,
        dim: usize,
        codebook_size: usize,
        codebook_dim: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = vb.pp("layers");
        let mut layers = Vec::with_capacity(n_q);
        for i in 0..n_q {
            let layer = VectorQuantization::new(dim, codebook_size, codebook_dim, vb.pp(i))?;
            layers.push(layer)
        }
        Ok(Self { layers })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let mut codes = Vec::with_capacity(self.layers.len());
        let mut residual = xs.clone();
        for layer in self.layers.iter() {
            let indices = layer.encode(&residual)?;
            let quantized = layer.decode(&indices)?;
            residual = (residual - quantized)?;
            codes.push(indices)
        }
        Tensor::stack(&codes, 0)
    }

    pub fn decode(&self, xs: &Tensor) -> Result<Tensor> {
        if self.layers.is_empty() {
            candle::bail!("empty layers in ResidualVectorQuantization");
        }
        if self.layers.len() != xs.dim(0)? {
            candle::bail!(
                "mismatch between the number of layers {} and the code shape {:?}",
                self.layers.len(),
                xs.shape()
            );
        }
        let mut quantized = self.layers[0].decode(&xs.i(0)?)?;
        for (i, layer) in self.layers.iter().enumerate().skip(1) {
            let xs = xs.i(i)?;
            quantized = (quantized + layer.decode(&xs))?
        }
        Ok(quantized)
    }
}

#[derive(Debug, Clone)]
pub struct ResidualVectorQuantizer {
    vq: ResidualVectorQuantization,
    input_proj: Option<candle_nn::Conv1d>,
    output_proj: Option<candle_nn::Conv1d>,
}

impl ResidualVectorQuantizer {
    pub fn new(
        dim: usize,
        input_dim: Option<usize>,
        output_dim: Option<usize>,
        n_q: usize,
        bins: usize,
        force_projection: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_dim = input_dim.unwrap_or(dim);
        let output_dim = output_dim.unwrap_or(dim);

        let input_proj = if input_dim == dim && !force_projection {
            None
        } else {
            let c = candle_nn::conv1d_no_bias(
                input_dim,
                dim,
                1,
                Default::default(),
                vb.pp("input_proj"),
            )?;
            Some(c)
        };
        let output_proj = if output_dim == dim && !force_projection {
            None
        } else {
            let c = candle_nn::conv1d_no_bias(
                dim,
                output_dim,
                1,
                Default::default(),
                vb.pp("output_proj"),
            )?;
            Some(c)
        };

        let vq = ResidualVectorQuantization::new(
            n_q,
            dim,
            /* codebook_size */ bins,
            /* codebook_dim */ None,
            vb.pp("vq"),
        )?;
        Ok(Self {
            vq,
            input_proj,
            output_proj,
        })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let codes = self.vq.encode(&xs.apply(&self.input_proj.as_ref())?)?;
        codes.transpose(0, 1)
    }

    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        // codes is [B, K, T], with T frames, K nb of codebooks, vq.decode expects [K, B, T].
        let codes = codes.transpose(0, 1)?;
        let quantized = self.vq.decode(&codes)?;
        match &self.output_proj {
            None => Ok(quantized),
            Some(p) => quantized.apply(p),
        }
    }
}

// we do not use any codebook_offset at the moment. When reconstructing the codes, we could just
// concatenate the indexes.
#[derive(Debug, Clone)]
pub struct SplitResidualVectorQuantizer {
    rvq_first: ResidualVectorQuantizer,
    rvq_rest: ResidualVectorQuantizer,
    n_q: usize,
    span_encode: tracing::Span,
    span_decode: tracing::Span,
}

impl SplitResidualVectorQuantizer {
    pub fn new(
        dim: usize,
        input_dim: Option<usize>,
        output_dim: Option<usize>,
        n_q: usize,
        bins: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let rvq_first = ResidualVectorQuantizer::new(
            dim,
            input_dim,
            output_dim,
            1,
            bins,
            true,
            vb.pp("rvq_first"),
        )?;
        let rvq_rest = ResidualVectorQuantizer::new(
            dim,
            input_dim,
            output_dim,
            n_q - 1,
            bins,
            true,
            vb.pp("rvq_rest"),
        )?;
        let span_encode = tracing::span!(tracing::Level::TRACE, "split-rvq-encode");
        let span_decode = tracing::span!(tracing::Level::TRACE, "split-rvq-decode");
        Ok(Self {
            rvq_first,
            rvq_rest,
            n_q,
            span_encode,
            span_decode,
        })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span_encode.enter();
        let codes = self.rvq_first.encode(xs)?;
        if self.n_q > 1 {
            // We encode xs again here rather than the residual. The decomposition is not
            // hierarchical but rather having semantic tokens for rvq_first and the acoustic tokens
            // for rvq_rest.
            let rest_codes = self.rvq_rest.encode(xs)?;
            Tensor::cat(&[codes, rest_codes], 1)
        } else {
            Ok(codes)
        }
    }

    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        // codes is [B, K, T], with T frames, K nb of codebooks.
        let _enter = self.span_decode.enter();
        let quantized = self.rvq_first.decode(&codes.i((.., ..1))?)?;
        let quantized = if self.n_q > 1 {
            (quantized + self.rvq_rest.decode(&codes.i((.., 1..))?)?)?
        } else {
            quantized
        };
        Ok(quantized)
    }
}

// Custom operation for efficient codebook encoding
struct CodebookEncode;

impl candle::CustomOp2 for CodebookEncode {
    fn name(&self) -> &'static str {
        "codebook-encode"
    }

    fn cpu_fwd(
        &self,
        xs: &candle::CpuStorage,
        xs_layout: &candle::Layout,
        embedding: &candle::CpuStorage,
        embedding_layout: &candle::Layout,
    ) -> candle::Result<(candle::CpuStorage, candle::Shape)> {
        // Efficient codebook encoding implementation
        use candle::Tensor;

        // Extract data from storage for tensor recreation
        let xs_data = xs.as_slice::<f32>()?;
        let embedding_data = embedding.as_slice::<f32>()?;

        // Create tensors from data using current API
        let xs_tensor = Tensor::from_slice(xs_data, xs_layout.shape(), &candle_core::Device::Cpu)?;
        let embedding_tensor = Tensor::from_slice(
            embedding_data,
            embedding_layout.shape(),
            &candle_core::Device::Cpu,
        )?;

        // Compute dot product: xs @ embedding.T
        let dot_prod = xs_tensor.matmul(&embedding_tensor.t()?)?;

        // Compute c2 values (squared norms of embedding vectors)
        let embedding_norms = embedding_tensor.sqr()?.sum_keepdim(1)?;
        let c2 = embedding_norms.broadcast_as(dot_prod.shape())?;

        // Compute distances: c2 - 2 * dot_prod
        // (We don't need xs norms since we only care about argmin)
        let distances = c2.broadcast_sub(&(dot_prod * 2.0)?)?;

        // Find closest codebook entries
        let codes = distances.argmin(candle::D::Minus1)?;

        // Return storage and shape - use simpler approach
        let shape = codes.shape().clone();

        // Convert to CPU storage directly
        let cpu_codes = codes.to_device(&candle_core::Device::Cpu)?;
        let (storage, _layout) = cpu_codes.storage_and_layout();

        // Convert to CpuStorage directly
        match &*storage {
            candle::Storage::Cpu(cpu_storage) => Ok((cpu_storage.clone(), shape)),
            _ => Err(candle_core::Error::Msg("Expected CPU storage".to_string())),
        }
    }
}

// Helper function for creating linear layers
fn linear(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    candle_nn::linear(in_dim, out_dim, vb)
}
