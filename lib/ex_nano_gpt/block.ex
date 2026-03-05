defmodule ExNanoGPT.Block do
  @moduledoc """
  Transformer block with pre-norm residual connections.

  Mirrors nanoGPT's Block class (model.py lines 94-106).

  Architecture:
    x = x + attn(ln_1(x))
    x = x + mlp(ln_2(x))

  This is "pre-norm" style (LayerNorm before attention/MLP, not after).
  The residual connections (`x + ...`) allow gradients to flow directly
  through the network during backpropagation.
  """

  import Nx.Defn

  alias ExNanoGPT.{Attention, LayerNorm, MLP}

  @typedoc """
  Transformer block parameters.
  - `:ln_1` - LayerNorm params before attention
  - `:ln_2` - LayerNorm params before MLP
  - `:attn` - CausalSelfAttention params
  - `:mlp` - MLP params
  """
  @type params :: %{
          ln_1: LayerNorm.params(),
          ln_2: LayerNorm.params(),
          attn: Attention.params(),
          mlp: MLP.params()
        }

  @doc """
  Initialize transformer block parameters.

  ## Options
    * `:bias` - whether to include bias terms (default: true)
    * `:n_layer` - total number of layers, for c_proj init scaling (default: 1)
  """
  @spec init_params(pos_integer(), pos_integer(), Nx.Tensor.t(), keyword()) :: params()
  def init_params(n_embd, n_head, key, opts \\ []) do
    bias? = Keyword.get(opts, :bias, true)

    keys = Nx.Random.split(key, parts: 2)

    %{
      ln_1: LayerNorm.init_params(n_embd, bias: bias?),
      ln_2: LayerNorm.init_params(n_embd, bias: bias?),
      attn: Attention.init_params(n_embd, n_head, keys[0], opts),
      mlp: MLP.init_params(n_embd, keys[1], opts)
    }
  end

  @doc """
  Forward pass: pre-norm attention + residual, then pre-norm MLP + residual.

  ## Inputs
    * `x` - input tensor, shape `{batch, seq_len, n_embd}`
    * `params` - block params from `init_params/4`
    * `key` - PRNG key for dropout

  ## Options
    * `:n_head` - number of attention heads (required)
    * `:dropout_rate` - dropout probability (default: 0.0)
    * `:training` - whether in training mode (default: false)

  ## Returns
  Tensor of shape `{batch, seq_len, n_embd}`.
  """
  defn forward(x, params, key, opts \\ []) do
    # x = x + attn(ln_1(x))
    residual = x
    x = LayerNorm.forward(x, params.ln_1)
    x = Attention.forward(x, params.attn, key, opts)
    x = residual + x

    # x = x + mlp(ln_2(x))
    residual = x
    x = LayerNorm.forward(x, params.ln_2)
    x = MLP.forward(x, params.mlp, key, opts)
    x = residual + x

    x
  end
end
