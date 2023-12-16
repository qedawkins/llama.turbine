from pathlib import Path

import torch

from shark_turbine.aot import *
from iree.compiler.ir import Context

from .model import *


ENABLE_DEBUG = True

def debug(*args):
    if ENABLE_DEBUG:
        print(*args)

BATCH_SIZE = 1
MAX_STEP_SEQ = 4095

class CompiledLlamaCPP(HParamsModule, CompiledModule):
    def __new__(cls, hp: HParams, *args, **kwargs):
        self = super(CompiledLlamaCPP, cls).__new__(cls, *args, **kwargs)
        self.__init__(hp)

    def __init__(self, hp: HParams):
        HParamsModule.__init__(self, hp)

        # Attention hyper-params.
        self.embedding_length = int(self.hp["llama.embedding_length"])
        self.max_seqlen = int(self.hp["llama.context_length"])
        self.transformer_block_count = int(self.hp["llama.block_count"])
        self.attention_layer_norm_rms_epsilon = self.hp[
            "llama.attention.layer_norm_rms_epsilon"
        ]
        self.attention_head_dim = int(self.hp["llama.rope.dimension_count"])
        self.attention_head_count = int(self.hp["llama.attention.head_count"])
        self.attention_head_count_kv = int(self.hp["llama.attention.head_count_kv"])

        assert (
            self.attention_head_count * self.attention_head_dim == self.embedding_length
        )
        assert (
            self.attention_head_count_kv * self.attention_head_dim
            == self.embedding_length
        )

        # Initialize the rope.
        if "llama.rope.dimension_count" in self.hp:
            scaling_factor = None
            if "llama.rope.scale_linear" in self.hp:
                scaling_factor = int(self.hp["llama.rope.scale_linear"])
            self.rope_dimension_count = self.hp["llama.rope.dimension_count"]
            self.rotary_embed_table = create_rotary_embed_table(
                max_seqlen=self.max_seqlen,
                dim=self.rope_dimension_count,
            )
        else:
            raise ValueError("Unsupported rotary embedding")

        # Initialize the KV cache.
        cache_k = torch.empty(
            (
                self.transformer_block_count,
                self.max_seqlen,
                self.hp.bs,
                self.attention_head_count,
                self.attention_head_dim,
            ),
            dtype=self.hp.dtype,
        )
        cache_v = torch.empty(
            (
                self.transformer_block_count,
                self.max_seqlen,
                self.hp.bs,
                self.attention_head_count,
                self.attention_head_dim,
            ),
            dtype=self.hp.dtype,
        )

        self.global_seq_step = export_global(AbstractIndex, mutable=True)
        self.global_cache_k = export_global(
            abstractify(cache_k), uninitialized=True, mutable=True
        )
        self.global_cache_v = export_global(
            abstractify(cache_v), uninitialized=True, mutable=True
        )

    def run_initialize(self, x=AbstractTensor(BATCH_SIZE, None, dtype=torch.int64)):
        token = self.initialize(x)
        self.global_seq_step = x.dynamic_dim(1) + 1
        return token
    
    def run_forward(self, x=AbstractTensor(1, 1, dtype=torch.int64)):
        token = self.forward(x, self.global_seq_step)
        self.global_seq_step = self.global_seq_step + 1
        return token

    @jittable
    def initialize(tokens: torch.Tensor):
        return inner_forward(tokens, 0)
    
    @jittable
    def forward(token0: torch.Tensor, start_index: int):
        return inner_forward(token0, start_index)

    # def extract_slice(cache, i, seq_step, batch_size):
    #     heads = self.attention_head_count
    #     hidden_dim = self.attention_head_dim
    #     sliced = IREE.tensor_slice(
    #         cache, i, (0, seq_step), (0, batch_size), (0, heads), (0, hidden_dim)
    #     )
    #     return IREE.tensor_reshape(sliced, seq_step, batch_size, heads, hidden_dim)

    @jittable
    def inner_forward(
        tokens: torch.Tensor, start_index: int, return_logits: bool = False
    ):
        bs, sl = tokens.shape
        assert bs == hp.bs, "Batch size mismatch vs params"
        h = tok_embeddings(tokens)
        
        # Compute attention mask.
        attention_mask = None
        if sl > 1:
            # Use the smallest value like HF as opposed to -inf like original.
            # A little bit easier for some systems.
            attention_mask = torch.full(
                (1, 1, sl, sl), torch.finfo(hp.dtype).min, dtype=hp.dtype
            )
            attention_mask = torch.triu(
                attention_mask, diagonal=start_index + 1
            ).type_as(h)

        # Transformer blocks.
        for block_idx in range(transformer_block_count):
            transformer_theta = theta("blk", block_idx)
            # Attention.
            attention_output = attention(
                transformer_theta,
                h,
                block_idx=block_idx,
                start_index=start_index,
                attention_mask=attention_mask,
            )
            h = h + attention_output

            # Feed-forward network.
            ff_input = rms_norm(
                transformer_theta("ffn_norm"),
                h,
                eps=attention_layer_norm_rms_epsilon,
            )
            ff_gate = F.silu(
                linear(
                    transformer_theta("ffn_gate"),
                    ff_input,
                    stored_transposed=True,
                )
            )
            ff_up = linear(
                transformer_theta("ffn_up"), ff_input, stored_transposed=True
            )
            ff_down = linear(
                transformer_theta("ffn_down"), ff_gate * ff_up, stored_transposed=True
            )
            h = h + ff_down

        # Output norm.
        h = rms_norm(
            theta("output_norm"),
            h,
            eps=attention_layer_norm_rms_epsilon,
        )

        # Output LM head.
        logits = linear(theta("output"), h, stored_transposed=True)

        # Return logits or token.
        # Shape: bs, sl, logits
        if return_logits:
            return h
        else:
            last_step = logits[:, -1, :]
            token = torch.argmax(last_step, dim=1)
            return token

    @jittable
    def tok_embeddings(tokens, stored_transposed=True):
        w, qw = p("token_embd", "weight")
        if qw is not None:
            w = qw.unpack().dequant(hp.dtype)
        w_shape = w.shape
        if stored_transposed:
            w = w.view(w_shape[1], w_shape[0])
        return F.embedding(tokens, w)

    @jittable
    def attention(
        theta: Theta,
        x: torch.Tensor,
        *,
        block_idx: int,
        start_index: int,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        x = rms_norm(
            theta("attn_norm"), x, eps=attention_layer_norm_rms_epsilon
        )

        bs, q_len, feature_dim = x.shape
        kv_seq_len = start_index + q_len
        assert feature_dim == attention_head_count * attention_head_dim

        xq = linear(theta("attn_q"), x)
        xk = linear(theta("attn_k"), x)
        xv = linear(theta("attn_v"), x)

        xq = xq.view(bs, q_len, attention_head_count, attention_head_dim)
        xk = xk.view(bs, q_len, attention_head_count_kv, attention_head_dim)
        xv = xv.view(bs, q_len, attention_head_count_kv, attention_head_dim)

        offset_rotary_embed_table = rotary_embed_table[start_index:kv_seq_len, :]
        xq, xk = apply_rotary_embed(xq, xk, offset_rotary_embed_table)

        # TODO: Some model variants do some form of kv repetition to expand the
        # count of kv heads to the count of attention heads used by the q.
        # Here we assert they are the same.
        assert (
            attention_head_count == attention_head_count_kv
        ), "NYI: KV expansion"


        # Update our positions in the cache.

        # Transpose into [sl, bs, heads, dim]
        xkt = xk.transpose(0, 1)
        xvt = xv.transpose(0, 1)

        heads = attention_head_count
        hidden_dim = attention_head_dim
        cache_k[block_idx, start_index:kv_seq_len] = xkt
        # cache_k[:bs, start_index:kv_seq_len] = xk
        # Hopefully
        # slice_of_k_state = IREE.tensor_reshape(
        #     xk, 1, batch_size, kv_seq_len - start_index, heads, hidden_dim
        # )
        # cache_k = IREE.tensor_update(
        #     cache_k, slice_of_state, block_idx, 0, start_index, 0, 0
        # )
        cache_v[block_idx, start_index:kv_seq_len] = xvt
        # cache_v[:bs, start_index:kv_seq_len] = xv
        # Hopefully
        # slice_of_v_state = IREE.tensor_reshape(
        #     xv, 1, batch_size, kv_seq_len - start_index, heads, hidden_dim
        # )
        # cache_v = IREE.tensor_update(
        #     cache_v, slice_of_v_state, block_idx, 0, start_index, 0, 0
        # )

        # Derive keys/values from the entirety of the available sequence.
        # cache_k = cache_k[block_idx, seq_len, bs, ...]
        # keys = cache_k[:kv_seq_len, :bs]
        # or hopefully
        # keys = extract_slice(cache_k, block_idx, kv_seq_len, bs)
        keys = cache_k[block_idx, kv_seq_len, bs, ...]
        # cache_v = cache_v[block_idx, seq_len, bs, ...]
        # values = cache_v[:kv_seq_len, :bs]
        # or hopefully
        # values = extract_slice(cache_v, block_idx, kv_seq_len, bs)
        values = cache_v[block_idx, kv_seq_len, bs, ...]

        # Transpose into [bs, sl, heads, dim]
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Tranpose into [bs, heads, sl, dim]
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_weights = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(
            attention_head_dim
        )

        # Apply attention mask.
        if attention_mask is not None:
            expected_mask_shape = (bs, 1, q_len, kv_seq_len)
            assert (
                attention_mask.shape == expected_mask_shape
            ), f"Attention mask should be of size {expected_mask_shape}, but is {attention_mask.shape}"
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(xq)
        output = torch.matmul(attn_weights, values)  # (bs, heads, slen, head_dim)
        output = output.transpose(1, 2).reshape(bs, q_len, -1)

        output = linear(theta("attn_output"), output)
        return output

    @jittable
    def apply_rotary_embed(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        # xq_, xk_ shape: bs, sl, _, dim
        # freqs_cis shape: max_sl, dim
        xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
        _, sl, _, dim = xq_.shape

        assert freqs_cis.shape[-1] == dim
        assert freqs_cis.shape[0] >= sl, "Sequence length longer than embedding table"
        bounded_freqs_cis = freqs_cis[None, 0:sl, None, :]

        xq_out = torch.view_as_real(xq_ * bounded_freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * bounded_freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    @jittable
    def rms_norm(theta: Theta, x: torch.Tensor, *, eps: float = 1e-6):
        w, qw = theta.p("weight")
        if qw is not None:
            w = qw.unpack().dequant(hp.dtype)
        variance = x.pow(2).mean(-1, keepdim=True)
        output = x * torch.rsqrt(variance + eps)
        output = output * w
        return output

    @jittable
    def linear(
        theta: Theta,
        x: torch.Tensor,
        *,
        transpose_weights=True,
        stored_transposed=False,
    ):
        w, qw = theta.p("weight")
        if qw is not None:
            w = qw.unpack().dequant(hp.dtype)
        if stored_transposed:
            w = w.reshape(w.shape[1], w.shape[0])
        if transpose_weights:
            w = w.T
        return torch.matmul(x, w)


def create_rotary_embed_table(max_seqlen: int, dim: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seqlen, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

if __name__ == "__main__":
    torch.no_grad().__enter__()
    path = Path("/home/quinn/llama.cpp/models/3B")
    hp = HParams(path / "ggml-model-q8_0.gguf")
    # print(hp)
    detokenizer = Detokenizer(hp)
    inner_module = LlamaCPP(hp)

    # class StateUpdateModule(CompiledModule):
    #     global_seq_step = export_global(AbstractIndex, mutable=True)
    #     params = export_parameters(inner_module)
    # 
    #     def run_initialize(self, x=AbstractTensor(BATCH_SIZE, None, dtype=torch.int64)):
    #         token = self.initialize(x)
    #         self.global_seq_step = x.dynamic_dim(1) + 1
    #         return token
    # 
    #     def run_forward(self, x=AbstractTensor(1, 1, dtype=torch.int64)):
    #         token = self.forward(x, self.global_seq_step)
    #         self.global_seq_step = self.global_seq_step + 1
    #         return token

    #     @jittable
    #     def initialize(tokens: torch.Tensor):
    #         return inner_module.forward(tokens, 0)
    # 
    #     @jittable
    #     def forward(token0: torch.Tensor, start_index: int):
    #         return inner_module.forward(token0, start_index)

    inst = CompiledLlamaCPP(hp, context=Context(), import_to="IMPORT")
    debug(inst)
