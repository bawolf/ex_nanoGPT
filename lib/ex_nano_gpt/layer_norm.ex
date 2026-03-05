defmodule ExNanoGPT.LayerNorm do
  @moduledoc """
  Layer Normalization with optional bias.

  Mirrors nanoGPT's LayerNorm class (model.py lines 18-27).
  Uses eps=1e-5 to match PyTorch's F.layer_norm default.

  Normalizes over the last dimension:
    output = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
  """

  import Nx.Defn

  @typedoc "Layer norm parameters: scale weight and optional bias, both shape {n_embd}"
  @type params :: %{weight: Nx.Tensor.t(), bias: Nx.Tensor.t()} | %{weight: Nx.Tensor.t()}

  @doc """
  Initialize layer norm parameters.

  ## Options
    * `:bias` - whether to include a bias term (default: true)

  ## Returns
  A params map with `:weight` (shape `{n_embd}`, ones) and
  optionally `:bias` (shape `{n_embd}`, zeros).
  """
  @spec init_params(pos_integer(), keyword()) :: params()
  def init_params(n_embd, opts \\ []) do
    bias? = Keyword.get(opts, :bias, true)

    params = %{weight: Nx.broadcast(1.0, {n_embd})}

    if bias? do
      Map.put(params, :bias, Nx.broadcast(0.0, {n_embd}))
    else
      params
    end
  end

  @doc """
  Apply layer normalization.

  ## Inputs
    * `x` - input tensor, any shape; normalization is over the last axis
    * `params` - layer norm params from `init_params/2`

  ## Returns
  Normalized tensor, same shape as `x`.
  """
  defn forward(x, params) do
    eps = 1.0e-5

    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    variance = Nx.variance(x, axes: [-1], keep_axes: true)

    normalized = (x - mean) / Nx.sqrt(variance + eps)

    result = normalized * params.weight
    maybe_add_bias(result, params)
  end

  deftransform maybe_add_bias(result, params) do
    if Map.has_key?(params, :bias) do
      Nx.add(result, params.bias)
    else
      result
    end
  end
end
