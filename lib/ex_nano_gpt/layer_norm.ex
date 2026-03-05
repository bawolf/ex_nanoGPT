defmodule ExNanoGPT.LayerNorm do
  @moduledoc """
  Layer Normalization with optional bias.

  Mirrors nanoGPT's LayerNorm class (model.py lines 18-27).
  Uses eps=1e-5 to match PyTorch's F.layer_norm default.

  Normalizes over the last dimension:
    output = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
  """

  import Nx.Defn

  @doc """
  Initialize layer norm parameters.

  Returns a map with:
  - :weight - scale parameter, shape {n_embd}, initialized to ones
  - :bias - shift parameter, shape {n_embd}, initialized to zeros (or nil if bias=false)
  """
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

  Normalizes over the last axis of the input tensor.
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
