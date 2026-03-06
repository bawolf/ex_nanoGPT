defmodule ExNanoGPT.V2.WeightLoader do
  @moduledoc """
  Load nanochat pre-trained weights into the v2 model.

  Expects weights to be pre-converted from PyTorch .pt to individual .npy files
  using `scripts/convert_checkpoint.py`. The directory should contain:
    - metadata.json (with config and parameter shapes)
    - *.npy files (one per parameter tensor)

  Key mapping from nanochat's parameter names to our struct:
    transformer.wte.weight         -> params.wte
    lm_head.weight                 -> params.lm_head
    transformer.resid_lambdas      -> params.resid_lambdas
    transformer.x0_lambdas         -> params.x0_lambdas
    transformer.h.{i}.attn.c_q.weight   -> blocks[i].c_q (transposed)
    transformer.h.{i}.attn.c_k.weight   -> blocks[i].c_k (transposed)
    transformer.h.{i}.attn.c_v.weight   -> blocks[i].c_v (transposed)
    transformer.h.{i}.attn.c_proj.weight -> blocks[i].c_proj (transposed)
    transformer.h.{i}.attn.ve_gate.weight -> blocks[i].ve_gate (transposed)
    transformer.h.{i}.mlp.c_fc.weight    -> blocks[i].c_fc (transposed)
    transformer.h.{i}.mlp.c_proj.weight  -> blocks[i].c_proj_mlp (transposed)
    transformer.value_embeds.{i}.weight  -> value_embeds[i]
  """

  alias ExNanoGPT.V2.Model
  alias ExNanoGPT.Test.Npy

  @doc """
  Load converted nanochat weights from a directory of .npy files.

  Returns `{params, config}` where params is ready for `Model.forward/3`.

  ## Options
    * `:dtype` - tensor type for all weights (default: `:f32`, use `:f16` to save memory)
  """
  def load(dir, opts \\ []) do
    dtype = Keyword.get(opts, :dtype, :f32)

    metadata = read_metadata(dir)
    config = build_config(metadata)

    n_kv_head = config.n_kv_head

    wte = load_npy(dir, "transformer.wte.weight", dtype)
    lm_head = load_npy(dir, "lm_head.weight", dtype)

    resid_lambdas = load_npy(dir, "transformer.resid_lambdas", dtype)
    x0_lambdas = load_npy(dir, "transformer.x0_lambdas", dtype)

    blocks =
      for i <- 0..(config.n_layer - 1) do
        %{
          c_q: load_linear(dir, "transformer.h.#{i}.attn.c_q.weight", dtype),
          c_k: load_linear(dir, "transformer.h.#{i}.attn.c_k.weight", dtype),
          c_v: load_linear(dir, "transformer.h.#{i}.attn.c_v.weight", dtype),
          c_proj: load_linear(dir, "transformer.h.#{i}.attn.c_proj.weight", dtype),
          ve_gate: load_linear_optional(dir, "transformer.h.#{i}.attn.ve_gate.weight", dtype,
                     {Model.ve_gate_channels(), n_kv_head}),
          c_fc: load_linear(dir, "transformer.h.#{i}.mlp.c_fc.weight", dtype),
          c_proj_mlp: load_linear(dir, "transformer.h.#{i}.mlp.c_proj.weight", dtype)
        }
      end
      |> List.to_tuple()

    value_embeds =
      for i <- 0..(config.n_layer - 1) do
        key = "transformer.value_embeds.#{i}.weight"
        if has_param?(metadata, key) do
          load_npy(dir, key, dtype)
        else
          :none
        end
      end
      |> List.to_tuple()

    params = %{
      wte: wte,
      lm_head: lm_head,
      resid_lambdas: resid_lambdas,
      x0_lambdas: x0_lambdas,
      blocks: blocks,
      value_embeds: value_embeds
    }

    {params, config}
  end

  defp read_metadata(dir) do
    path = Path.join(dir, "metadata.json")
    path |> File.read!() |> Jason.decode!()
  end

  defp build_config(metadata) do
    cfg = metadata["config"] || %{}
    %Model{
      sequence_len: Map.get(cfg, "sequence_len", 2048),
      vocab_size: Map.get(cfg, "vocab_size", 32768),
      n_layer: Map.get(cfg, "n_layer", 12),
      n_head: Map.get(cfg, "n_head", 6),
      n_kv_head: Map.get(cfg, "n_kv_head", 6),
      n_embd: Map.get(cfg, "n_embd", 768),
      window_pattern: Map.get(cfg, "window_pattern", "SSSL")
    }
  end

  defp has_param?(metadata, key), do: Map.has_key?(metadata["params"] || %{}, key)

  defp load_npy(dir, key, dtype) do
    filename = String.replace(key, "/", "_") <> ".npy"
    path = Path.join(dir, filename)
    Npy.load!(path) |> Nx.as_type(dtype)
  end

  defp load_linear(dir, key, dtype) do
    # PyTorch Linear stores weights as (out_features, in_features)
    # We need (in_features, out_features) for Nx.dot(x, [-1], w, [0])
    load_npy(dir, key, dtype) |> Nx.transpose()
  end

  defp load_linear_optional(dir, key, dtype, default_shape) do
    filename = String.replace(key, "/", "_") <> ".npy"
    path = Path.join(dir, filename)
    if File.exists?(path) do
      load_npy(dir, key, dtype) |> Nx.transpose()
    else
      Nx.broadcast(Nx.tensor(0.0, type: dtype), default_shape)
    end
  end
end
