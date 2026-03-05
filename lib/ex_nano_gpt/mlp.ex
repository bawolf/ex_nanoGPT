defmodule ExNanoGPT.MLP do
  @moduledoc """
  Feed-forward network (MLP) used in each transformer block.

  Mirrors nanoGPT's MLP class (model.py lines 78-92).

  Architecture: n_embd -> 4*n_embd -> GELU -> 4*n_embd -> n_embd -> dropout
  The 4x expansion is standard in GPT-2.
  """

  import Nx.Defn

  @typedoc """
  MLP parameters.
  - `:c_fc_weight` - first linear, shape `{n_embd, 4 * n_embd}`
  - `:c_fc_bias` - first linear bias, shape `{4 * n_embd}` (optional)
  - `:c_proj_weight` - second linear, shape `{4 * n_embd, n_embd}`
  - `:c_proj_bias` - second linear bias, shape `{n_embd}` (optional)
  """
  @type params :: %{
          c_fc_weight: Nx.Tensor.t(),
          c_proj_weight: Nx.Tensor.t(),
          c_fc_bias: Nx.Tensor.t() | nil,
          c_proj_bias: Nx.Tensor.t() | nil
        }

  @doc """
  Initialize MLP parameters.

  ## Options
    * `:bias` - whether to include bias terms (default: true)
    * `:n_layer` - number of layers, for scaling c_proj init (default: 1)
  """
  @spec init_params(pos_integer(), Nx.Tensor.t(), keyword()) :: params()
  def init_params(n_embd, key, opts \\ []) do
    bias? = Keyword.get(opts, :bias, true)
    n_layer = Keyword.get(opts, :n_layer, 1)

    keys = Nx.Random.split(key, parts: 2)

    {c_fc_weight, _} = Nx.Random.normal(keys[0], 0.0, 0.02, shape: {n_embd, 4 * n_embd})
    proj_std = 0.02 / :math.sqrt(2 * n_layer)
    {c_proj_weight, _} = Nx.Random.normal(keys[1], 0.0, proj_std, shape: {4 * n_embd, n_embd})

    params = %{c_fc_weight: c_fc_weight, c_proj_weight: c_proj_weight}

    if bias? do
      params
      |> Map.put(:c_fc_bias, Nx.broadcast(0.0, {4 * n_embd}))
      |> Map.put(:c_proj_bias, Nx.broadcast(0.0, {n_embd}))
    else
      params
    end
  end

  @doc """
  Forward pass: linear -> GELU -> linear -> dropout.

  ## Inputs
    * `x` - input tensor, shape `{batch, seq_len, n_embd}`
    * `params` - MLP params from `init_params/3`
    * `key` - PRNG key for dropout

  ## Options
    * `:dropout_rate` - dropout probability (default: 0.0)
    * `:training` - whether in training mode (default: false)

  ## Returns
  Tensor of shape `{batch, seq_len, n_embd}`.
  """
  defn forward(x, params, key, opts \\ []) do
    dropout_rate = opts[:dropout_rate]
    training = opts[:training]

    x = linear(x, params.c_fc_weight, params[:c_fc_bias])
    x = gelu(x)
    x = linear(x, params.c_proj_weight, params[:c_proj_bias])

    if training do
      {x, _} = apply_dropout(x, key, dropout_rate)
      x
    else
      x
    end
  end

  # GELU activation matching PyTorch's nn.GELU() (approximate='none')
  # gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
  defnp gelu(x) do
    x * 0.5 * (1.0 + Nx.erf(x / Nx.sqrt(Nx.tensor(2.0, type: :f32))))
  end

  defnp linear(x, weight, bias) do
    result = Nx.dot(x, [-1], weight, [0])
    maybe_add_bias(result, bias)
  end

  deftransform maybe_add_bias(result, nil), do: result
  deftransform maybe_add_bias(result, bias), do: Nx.add(result, bias)

  defnp apply_dropout(x, key, rate) do
    if rate > 0.0 do
      {mask, new_key} = Nx.Random.uniform(key, shape: Nx.shape(x))
      keep = Nx.greater(mask, rate)
      {x * keep / (1.0 - rate), new_key}
    else
      {x, key}
    end
  end
end
